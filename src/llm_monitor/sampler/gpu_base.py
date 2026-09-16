"""GpuCollector 协议与厂商注册表。

第一版只实现 NVIDIA(nvidia.py)。其他厂商的 collector 应实现同一协议,
在 sampler/gpu/__init__.py 中 import 即完成注册。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class GpuSample:
    vendor: str
    index: int
    name: str = ""
    util_pct: float = 0.0
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    temperature_c: float = 0.0
    power_w: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class GpuCollector(Protocol):
    vendor: str

    def available(self) -> bool: ...
    def device_count(self) -> int: ...
    def sample(self) -> list[GpuSample]: ...
    def close(self) -> None: ...


REGISTRY: list[type] = []


def register(cls):
    REGISTRY.append(cls)
    return cls


def autodetect(backends: list[str]) -> list[GpuCollector]:
    """按 backends 过滤。['auto'] 表示尝试所有已注册且 available() 的。"""
    picked: list[GpuCollector] = []
    want_all = backends == ["auto"] or not backends
    wanted = None if want_all else set(backends)

    for cls in REGISTRY:
        instance = cls()
        if wanted is not None and getattr(instance, "vendor", "") not in wanted:
            continue
        try:
            if instance.available():
                picked.append(instance)
        except Exception:
            continue
    return picked
