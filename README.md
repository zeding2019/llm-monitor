# llm-monitor

轻量、零侵入的 LLM 推理引擎请求级监控（支持 vLLM 和 SGLang）。通过猴子补丁挂载,与推理引擎同进程部署,提供:

- **请求级端到端 Transaction**:含消息树、TTFT、e2e 耗时
- **主机健康**:CPU / 内存 / 磁盘 / 网络带宽 / GPU 显存 & 利用率
- **多厂商 GPU 适配**:NVIDIA(pynvml)、海光、寒武纪、昆仑芯(插件式,按需启用)
- **四种指标模型**:Transaction / Event / Heartbeat / Metric
- **Web 实时可视化** + **Prometheus 出口**
- **内存 + SQLite** 两层存储,历史可查

## 设计原则

1. 零侵入:vLLM/SGLang 源码不动,`LLM_MONITOR_ENABLE=1` 或 `import llm_monitor` 触发
2. 低开销:热路径只做 `perf_counter` + append,聚合/GPU 采样在后台线程
3. 不抢主机资源:GPU 用 NVML(不占 CUDA context),Web 单线程 uvicorn 绑本地

## 快速开始

### 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,nvidia]"
```

### 使用方式一:vLLM OpenAI server

```bash
export LLM_MONITOR_ENABLE=1
export LLM_MONITOR_PORT=9109
python -m vllm.entrypoints.openai.api_server --model xxx
# 打开 http://127.0.0.1:9109
```

### 使用方式二:Python API

```python
import llm_monitor  # 自动 install
from vllm import LLM
llm = LLM(model="xxx")
llm.generate(["hello"])
```

### 使用方式三:自定义 FastAPI(手动挂中间件)

若你自己包了一层 FastAPI 网关,想把 HTTP 请求也作为顶层 Transaction:

```python
import llm_monitor
from llm_monitor.patch import LlmMonitorMiddleware
from fastapi import FastAPI

app = FastAPI()
app.add_middleware(LlmMonitorMiddleware)
```

## 四种指标模型

| 类型 | 用途 | 举例 |
|---|---|---|
| Transaction | 有耗时段,可嵌套成树 | 一次 generate 请求、prefill、decode |
| Event | 一次性事件计数 | OOM、请求取消、异常 |
| Heartbeat | 周期采样值 | GPU 显存、CPU、带宽 |
| Metric | 业务指标 count/sum/avg | tokens/s、prompt/gen tokens |

## 配置(环境变量)

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_MONITOR_ENABLE` | `0` | 主开关 |
| `LLM_MONITOR_HOST` | `127.0.0.1` | Web 绑定 |
| `LLM_MONITOR_PORT` | `9109` | Web 端口 |
| `LLM_MONITOR_GPU` | `auto` | `auto` / `nvidia,hygon,...` |
| `LLM_MONITOR_SAMPLE_INTERVAL_MS` | `1000` | Host 采样周期(CPU/内存/网络) |
| `LLM_MONITOR_GPU_SAMPLE_INTERVAL_MS` | `5000` | GPU 采样周期(独立线程,不阻塞 host) |
| `LLM_MONITOR_HYGON_TIMEOUT` | `15` | hy-smi/rocm-smi 命令超时(秒) |
| `LLM_MONITOR_DB_PATH` | `./llm_monitor.db` | SQLite 路径,空则纯内存 |
| `LLM_MONITOR_RETENTION_DAYS` | `7` | SQLite 保留天数 |
| `LLM_MONITOR_TRANSACTION_BUF` | `10000` | 内存环形容量 |

## 开发

```bash
pip install -e ".[dev]"
pytest
ruff check .
ruff format .
```

## 项目结构

```
src/llm_monitor/
├── bootstrap.py     install() / uninstall()
├── config.py        环境变量与默认值
├── core/            与厂商无关的内核(models/context/ringbuf/api/aggregator)
├── sampler/         后台采样(host + gpu/*)
├── patch/           vLLM 和 SGLang 猴子补丁
├── store/           memory + sqlite
└── web/             FastAPI + 静态页
```

## License

MIT
