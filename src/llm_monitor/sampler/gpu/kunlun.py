"""昆仑芯 XPU 采集器占位。

TODO:
- subprocess `xpu-smi discovery --dump 1,2,4,26,27` 或 JSON 输出解析
"""
from __future__ import annotations

from ..gpu_base import GpuSample, register


@register
class KunlunCollector:
    vendor = "kunlun"

    def available(self) -> bool:
        return False

    def device_count(self) -> int:
        return 0

    def sample(self) -> list[GpuSample]:
        return []

    def close(self) -> None:
        return None
