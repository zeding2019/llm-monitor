# vllm-monitor 实现计划

一个轻量、零侵入、以猴子补丁方式为 vLLM 提供请求级端到端监控的库。灵感对齐点评美团 CAT。

---

## 一、设计原则

1. **零侵入**:vLLM 源码不动。通过环境变量 `VLLM_MONITOR_ENABLE=1` + `sitecustomize.py` 机制或用户显式 `import vllm_monitor` 自动装载 patch。
2. **低开销**:热路径只做 `perf_counter()` + `deque.append` + `contextvars` 读写。序列化、聚合、GPU 采样、DB 写入全部在**后台守护线程**里做,通过 lock-free 队列与热路径解耦。
3. **不抢主机资源**:
   - GPU 采样默认 1s 一次,可调
   - 环形缓冲有上限(默认 Transaction 10000 条、Heartbeat 3600 点),满则丢老的
   - Web 服务单线程 uvicorn,绑定 127.0.0.1(可配),不占 vLLM 的 GPU/CPU 大头
4. **可插拔**:GPU 采集器、Patcher、Exporter(SQLite/Prometheus) 都做成插件式,注册即用。

---

## 二、目录结构

```
dashboard/
├── .gitignore
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml                 # 打包配置,提供 `vllm-monitor` CLI
├── .env.example
│
├── src/vllm_monitor/
│   ├── __init__.py                # 入口:import 时按 env 自动 install()
│   ├── bootstrap.py               # install() / uninstall() 主入口
│   ├── config.py                  # 配置(env + 可选 toml),含所有开关/阈值
│   │
│   ├── core/                      # 内核:与厂商无关
│   │   ├── models.py              # Transaction/Event/Heartbeat/Metric 数据类
│   │   ├── context.py             # TreeContext(contextvars),消息树构建
│   │   ├── clock.py               # 单调时钟封装
│   │   ├── ringbuf.py             # 线程安全环形缓冲
│   │   ├── aggregator.py          # 分钟聚合器(count/avg/p50/p95/p99/max)
│   │   └── api.py                 # 对外记录 API:transaction()/event()/metric()/heartbeat()
│   │
│   ├── sampler/                   # 后台采样线程
│   │   ├── runner.py              # SamplerThread(周期调度所有 collector)
│   │   ├── host.py                # CPU / 内存 / 磁盘 / 网络(psutil)
│   │   ├── gpu_base.py            # GpuCollector 协议 + 注册表 + 自动探测
│   │   └── gpu/
│   │       ├── nvidia.py          # pynvml 实现(第一版)
│   │       ├── hygon.py           # 海光 hy-smi/rocm-smi(占位,subprocess 兜底)
│   │       ├── cambricon.py       # 寒武纪 cnmon(占位)
│   │       └── kunlun.py          # 昆仑芯 xpu-smi(占位)
│   │
│   ├── patch/                     # 猴子补丁
│   │   ├── registry.py            # patch 注册/卸载框架
│   │   ├── vllm_engine.py         # AsyncLLMEngine.generate / LLMEngine.step
│   │   ├── vllm_scheduler.py      # Scheduler.schedule(排队/抢占)
│   │   └── vllm_openai.py         # OpenAI server 入口(可选)
│   │
│   ├── store/                     # 存储层
│   │   ├── memory.py              # 环形缓冲(热数据)
│   │   ├── sqlite.py              # 分钟聚合落盘 + 慢事务采样落盘
│   │   ├── schema.sql             # 建表 DDL
│   │   └── writer.py              # WriterThread(批量 flush)
│   │
│   ├── web/                       # Web 服务
│   │   ├── server.py              # FastAPI + 后台线程 uvicorn
│   │   ├── routes/
│   │   │   ├── dashboard.py       # 概览
│   │   │   ├── transactions.py    # 列表/详情/消息树
│   │   │   ├── events.py
│   │   │   ├── heartbeat.py       # 时序曲线(GPU/CPU/内存)
│   │   │   ├── metrics.py         # 业务指标
│   │   │   └── prometheus.py      # /metrics 出口
│   │   └── static/                # 单页 HTML + Chart.js(不引入前端构建)
│   │
│   └── util/
│       ├── logger.py
│       └── percentile.py          # T-Digest 或简单 reservoir
│
├── tests/
│   ├── test_ringbuf.py
│   ├── test_context_tree.py
│   ├── test_aggregator.py
│   ├── test_sampler_host.py
│   └── test_patch_dummy.py        # 用假 vllm module 验证 patch 逻辑
│
└── examples/
    ├── run_with_vllm_server.sh    # 演示 env 触发
    └── run_python_api.py          # 演示 import 触发
```

---

## 三、四种指标模型(对齐 CAT)

```python
# core/models.py 关键字段(伪代码)

@dataclass
class Transaction:
    type: str          # e.g. "vllm.generate"
    name: str          # e.g. request_id
    start_ns: int
    duration_ns: int
    status: str        # "0" ok / 错误码
    tags: dict[str, str]
    data: dict         # prompt_tokens、gen_tokens、TTFT 等
    children: list["Transaction"]   # 构成消息树
    parent_id: str | None

@dataclass
class Event:
    type: str; name: str; ts_ns: int; status: str; data: dict

@dataclass
class Heartbeat:
    ts_ns: int; source: str        # "host" / "gpu:0"
    values: dict[str, float]       # cpu_pct / mem_pct / gpu_util / vram_used_mb ...

@dataclass
class Metric:
    name: str; ts_ns: int
    count: int; sum: float          # avg = sum/count
```

**Transaction 树**:`TreeContext` 存在 `contextvars.ContextVar`,同一个协程/请求内 `with transaction("prefill"):` 自动挂到当前父节点下。请求结束(顶层 __exit__)把整棵树塞进环形缓冲。

---

## 四、猴子补丁点(初步)

| 目标 | Patch 目的 | 产出 |
|---|---|---|
| `AsyncLLMEngine.generate` (async gen) | 顶层 Transaction,记录 TTFT / e2e | `vllm.generate` 事务 |
| `LLMEngine.add_request` | 入队时间戳 | 埋在请求上下文 |
| `LLMEngine.step` | 每步 decode 耗时、batch size | 子 Transaction `vllm.step` + Metric |
| `Scheduler.schedule` | 排队时长、抢占次数 | `vllm.schedule` 子事务 + Event(preempt) |
| OpenAI server route | HTTP 入口 request_id 关联 | 顶层事务 tag |

每个 patch 都实现 `apply()` / `revert()`,便于测试和热卸载。`patch/registry.py` 在 import vllm 之后延迟绑定;若 vllm 未安装则整体跳过(便于单元测试)。

---

## 五、GPU 多厂商适配(核心难点)

```python
# sampler/gpu_base.py
class GpuCollector(Protocol):
    vendor: str
    def available(self) -> bool: ...
    def device_count(self) -> int: ...
    def sample(self) -> list[GpuSample]: ...    # 一卡一条
    def close(self) -> None: ...

REGISTRY: list[type[GpuCollector]] = []
def register(cls): REGISTRY.append(cls); return cls

def autodetect() -> list[GpuCollector]:
    picked = []
    for cls in REGISTRY:
        c = cls()
        if c.available(): picked.append(c)
    return picked
```

- **第一版实现 NVIDIA**(pynvml):利用率、显存(used/total)、温度、功耗、SM 时钟
- **其他厂商占位**:`available()` 返回 False,内部有 TODO 注释和 subprocess 兜底范式(解析 `hy-smi`/`cnmon`/`xpu-smi` 输出),便于后续按现场设备补齐
- 环境变量 `VLLM_MONITOR_GPU=nvidia,hygon` 可强制指定,跳过 autodetect

---

## 六、存储:内存 + SQLite

- **热数据**:全内存环形缓冲(`store/memory.py`)。查询近 N 分钟走内存。
- **冷数据**:WriterThread 每 60s 把上一分钟的聚合结果 + 采样保留的慢/异常事务写入 SQLite。查询长时间范围走 SQLite。
- **SQLite 配置**:WAL 模式、synchronous=NORMAL,单写线程,避免锁竞争。
- **保留策略**:默认 SQLite 保留 7 天,后台任务每小时清理(可配)。

表设计(粗):
```
minute_agg (metric_type, name, minute_ts, count, sum, p50, p95, p99, max, tags_json)
transactions (id, type, name, start_ts, duration_ns, status, tree_json, sampled_reason)
events (id, type, name, ts, status, data_json)
heartbeat (source, ts, values_json)
```

---

## 七、Web 层

- **FastAPI + uvicorn**,后台线程 `uvicorn.Server.serve()`(不 fork,不干扰 vLLM 主进程)
- 绑定默认 `127.0.0.1:9109`,可通过 env 改
- 页面用**单文件 HTML + Chart.js CDN**(不引入 npm/webpack,零构建)
- 路由:
  - `/` 仪表盘(实时曲线:GPU 利用率/显存/请求 QPS/P95)
  - `/transactions` 列表 → 详情(展示消息树 tree view)
  - `/events` `/heartbeat` `/metrics` 分栏
  - `/metrics/prom` Prometheus 格式导出
  - `/api/*` JSON,前端拉取

---

## 八、配置(env 优先,可选 toml)

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `VLLM_MONITOR_ENABLE` | `0` | 主开关 |
| `VLLM_MONITOR_HOST` | `127.0.0.1` | Web 绑定 |
| `VLLM_MONITOR_PORT` | `9109` | Web 端口 |
| `VLLM_MONITOR_GPU` | `auto` | `auto` / `nvidia,hygon,...` |
| `VLLM_MONITOR_SAMPLE_INTERVAL_MS` | `1000` | Heartbeat 采样周期 |
| `VLLM_MONITOR_DB_PATH` | `./vllm_monitor.db` | SQLite 路径,空表示纯内存 |
| `VLLM_MONITOR_RETENTION_DAYS` | `7` | SQLite 保留天数 |
| `VLLM_MONITOR_TRANSACTION_BUF` | `10000` | 内存环形容量 |

---

## 九、依赖

**运行时(requirements.txt)**
```
fastapi>=0.110
uvicorn[standard]>=0.29
psutil>=5.9
pynvml>=11.5        # NVIDIA GPU,其他厂商懒加载
```
> 说明:不硬依赖 vllm(便于纯测试环境跑单测);GPU 各厂商 SDK 全部**软依赖**,`try import` 失败则该 collector 不可用。

**开发(requirements-dev.txt)** 已有 ruff/pytest,追加 `pytest-asyncio`、`httpx`。

---

## 十、实现分期

**Phase 1(骨架,可跑通)**
1. 目录 + `pyproject.toml` + CLI 入口
2. `core/`(models、ringbuf、context、api、aggregator)+ 单测
3. `sampler/host.py`(psutil)+ `SamplerThread`
4. `store/memory.py` + `store/sqlite.py`(WriterThread)
5. `web/` 最小骨架:一个仪表盘页面显示 CPU/内存实时曲线
6. `bootstrap.install()`:import vllm_monitor 即启动上述所有后台线程

**Phase 2(vLLM 集成)**
7. `patch/vllm_engine.py`:generate/step 打点 → 消息树
8. `patch/vllm_scheduler.py`:排队/抢占
9. Web 增加 Transaction 列表/详情/树视图
10. `sampler/gpu/nvidia.py` 打通

**Phase 3(扩展)**
11. Prometheus 出口
12. 海光/寒武纪/昆仑芯 collector 占位实现 + 文档
13. 采样保留策略(慢事务、异常事务优先保留)
14. 认证/多用户(可选)

---

## 十一、本次(第一步)将要做的事

本次只落 **Phase 1 的骨架**:

1. 创建 `pyproject.toml`(替代/并存 requirements),声明包 `vllm-monitor`,提供 `vllm-monitor` CLI
2. 更新 `requirements.txt` / `requirements-dev.txt` 补运行时依赖
3. 更新 `README.md` 加入本项目定位、架构图、使用方法
4. 建 `src/vllm_monitor/` 全部目录 + `__init__.py`(空实现,导入不报错)
5. 落 **`core/models.py`、`core/context.py`、`core/ringbuf.py`、`core/api.py`** 可用版本
6. 落 **`sampler/host.py` + `sampler/runner.py`** 可用版本
7. 落 **`bootstrap.install()`** + env 触发
8. 落 **`web/server.py`** 最小版本(一个 `/` 返回 JSON 显示当前 host 指标)
9. 写 3~5 个核心单测(ringbuf、context 树、host 采样)
10. `examples/run_python_api.py` 演示:`import vllm_monitor` 后访问 `http://127.0.0.1:9109`

vLLM patch、GPU 采集、SQLite、完整 Web UI 留到后续 Phase。

---

## 十二、待你确认的点

若你同意本计划,回 "同意" 或直接 approve,我按 Phase 1 落地。
若某处想调整(例如目录扁平化、Web 换 Flask、SQLite 换 DuckDB、要不要现在就 stub vLLM patch),告诉我改哪里再走。
