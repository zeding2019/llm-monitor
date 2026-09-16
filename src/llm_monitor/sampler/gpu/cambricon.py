"""寒武纪 MLU 采集器占位。

TODO:
- 优先使用 `cndev` Python 绑定
- 兜底:subprocess `cnmon info --json`
"""
from __future__ import annotations

from ..gpu_base import GpuSample, register


@register
class CambriconCollector:
    vendor = "cambricon"

    def available(self) -> bool:
        return False

    def device_count(self) -> int:
        return 0

    def sample(self) -> list[GpuSample]:
        return []

    def close(self) -> None:
        return None
