"""启动阶段统计:双引擎(vLLM/SGLang)统一阶段模型。

阶段编号(引擎侧与前端共用同一张表):
  0 参数解析/配置      param_config
  1 环境/依赖初始化    env_init
  2 加载 Tokenizer 与模型配置  tokenizer_config
  3 加载模型权重       weights
  4 分布式/KV 初始化   dist_init
  5 总启动时长         total

记录方式:每个测量点 append 一条 heartbeat(source='startup'),values 全为数值:
  eng=引擎码(1=vllm 2=sglang) ph=阶段码 dur_ms pid
heartbeat 已由每进程 DbWriter 落库并跨进程可见,前端按 source='startup'
取到全部进程的阶段耗时;同 pid 同阶段只保留最后一次标记。
"""
from __future__ import annotations

import os
import time

ENGINE_VLLM = 1
ENGINE_SGLANG = 2

PHASE_CODES = {
    "param_config": 0,
    "env_init": 1,
    "tokenizer_config": 2,
    "weights": 3,
    "dist_init": 4,
    "total": 5,
    "cuda_graph": 6,
}
CODE_PHASE = {v: k for k, v in PHASE_CODES.items()}

# 跨引擎统一的中文标签(前端也可复用这份顺序)
PHASE_LABELS = [
    "参数解析/配置",
    "环境/依赖初始化",
    "加载 Tokenizer 与模型配置",
    "加载模型权重",
    "分布式/KV 初始化",
    "总启动时长",
    "CUDA Graph 编译",
]
ENGINE_LABELS = {ENGINE_VLLM: "vLLM", ENGINE_SGLANG: "SGLang"}

_PID = os.getpid()
# LOCAL_RANK: vLLM/SGLang 各 executor 后端均会在 Worker 进程里设此环境变量。
# 主进程无此变量时默认 -1，方便前端区分"主进程记录"与"GPU Worker 记录"。
_RANK = int(os.environ.get("LOCAL_RANK", -1))


def record(engine: int, phase: str, dur_ms: float) -> None:
    """记录一个阶段耗时。由各引擎补丁在测量点调用;失败静默。"""
    try:
        from ..core.models import Heartbeat
        from ..core.registry import get_stores
        code = PHASE_CODES.get(phase)
        if code is None:
            return
        get_stores().heartbeats.append(Heartbeat(
            ts_ns=time.time_ns(),
            source="startup",
            values={
                "eng": float(engine),
                "ph": float(code),
                "dur_ms": round(float(dur_ms), 1),
                "pid": float(_PID),
                "rank": float(_RANK),
            },
        ))
    except Exception:  # noqa: BLE001
        pass


class Timer:
    """计时器:begin()/end(phase) 或 with 块;begin 后可多次 end 不同阶段。"""

    __slots__ = ("engine", "_t0", "_mark")

    def __init__(self, engine: int, mark: bool = True) -> None:
        self.engine = engine
        self._t0: float | None = None
        self._mark = mark

    def begin(self) -> float:
        self._t0 = time.perf_counter()
        return self._t0

    def end(self, phase: str) -> float | None:
        """记 phase 耗时(相对 begin)。返回 ms,未 begin 返回 None。"""
        if self._t0 is None:
            return None
        ms = (time.perf_counter() - self._t0) * 1000.0
        if self._mark:
            record(self.engine, phase, ms)
        return ms

    def __enter__(self) -> Timer:
        self.begin()
        return self

    def __exit__(self, *exc) -> None:
        pass
