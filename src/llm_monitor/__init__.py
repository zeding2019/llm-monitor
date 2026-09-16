"""llm-monitor:轻量、零侵入的 LLM 推理引擎请求级监控(支持 vLLM 和 SGLang)。

- import llm_monitor 时,若 LLM_MONITOR_ENABLE=1,自动 install()
- 也可显式:from llm_monitor import install; install()
"""
from __future__ import annotations

from .bootstrap import install, uninstall
from .config import Config
from .core.api import event, heartbeat, metric, transaction

__version__ = "0.1.0"

__all__ = [
    "Config",
    "install",
    "uninstall",
    "transaction",
    "event",
    "metric",
    "heartbeat",
]

# 自动装载
Config.from_env().auto_install()
