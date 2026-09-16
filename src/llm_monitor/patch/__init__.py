"""vLLM 猴子补丁的统一入口。

Phase 2:延迟绑定,vLLM 未安装时静默跳过。
"""
from __future__ import annotations

from . import (
    sglang,        # noqa: F401 SGLang 补丁
    vllm_engine,   # noqa: F401 触发 when_imported 注册
    vllm_kvcache,  # noqa: F401 KV cache 命中率
    vllm_openai,   # noqa: F401
    vllm_perf,     # noqa: F401 GPU 算子 vs 框架耗时拆分
    vllm_startup,  # noqa: F401 启动阶段追踪
)
from .registry import apply_all_now
from .vllm_openai import LlmMonitorMiddleware

__all__ = ["install_all", "LlmMonitorMiddleware"]


def install_all() -> None:
    """由 bootstrap.install() 调用。已注册的 when_imported hook 会:
    - 若目标模块已在 sys.modules,立即打;
    - 否则等其被 import 时自动打。
    """
    apply_all_now()
