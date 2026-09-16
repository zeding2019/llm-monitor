"""NVIDIA GPU 采集,基于 pynvml(NVML C 库,不占用 CUDA context)。"""
from __future__ import annotations

import logging

from ..gpu_base import GpuSample, register

log = logging.getLogger("llm_monitor.gpu.nvidia")


@register
class NvidiaCollector:
    vendor = "nvidia"

    def __init__(self) -> None:
        self._nvml = None
        self._handles: list = []

    def available(self) -> bool:
        try:
            import pynvml  # type: ignore
        except ImportError:
            return False
        try:
            pynvml.nvmlInit()
        except Exception as e:  # noqa: BLE001
            log.debug("nvmlInit failed: %s", e)
            return False
        self._nvml = pynvml
        try:
            n = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
        except Exception as e:  # noqa: BLE001
            log.debug("nvmlDeviceGetCount failed: %s", e)
            return False
        return True

    def device_count(self) -> int:
        return len(self._handles)

    def sample(self) -> list[GpuSample]:
        if not self._nvml:
            return []
        p = self._nvml
        out: list[GpuSample] = []
        for i, h in enumerate(self._handles):
            try:
                name = p.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                mem = p.nvmlDeviceGetMemoryInfo(h)
                util = p.nvmlDeviceGetUtilizationRates(h)
                try:
                    temp = p.nvmlDeviceGetTemperature(h, p.NVML_TEMPERATURE_GPU)
                except Exception:  # noqa: BLE001
                    temp = 0
                try:
                    power = p.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception:  # noqa: BLE001
                    power = 0.0

                extra: dict[str, float] = {
                    "mem_bw_util_pct": float(util.memory),  # 显存带宽利用率
                }
                # PCIe 吞吐 (KB/s → MB/s)
                try:
                    tx_kbps = p.nvmlDeviceGetPcieThroughput(h, 0)  # TX = H2D
                    rx_kbps = p.nvmlDeviceGetPcieThroughput(h, 1)  # RX = D2H
                    extra["pcie_tx_mbps"] = tx_kbps / 1024.0
                    extra["pcie_rx_mbps"] = rx_kbps / 1024.0
                except Exception:  # noqa: BLE001
                    pass
                # PCIe 链路信息 + 理论带宽 + 利用率
                try:
                    gen = p.nvmlDeviceGetCurrPcieLinkGeneration(h)
                    width = p.nvmlDeviceGetCurrPcieLinkWidth(h)
                    extra["pcie_link_gen"] = float(gen)
                    extra["pcie_link_width"] = float(width)
                    max_mbps = _pcie_max_mbps(gen, width)
                    if max_mbps > 0:
                        extra["pcie_max_mbps"] = float(max_mbps)
                        # 单向理论上限,tx+rx 都以此为分母
                        cur = extra.get("pcie_tx_mbps", 0) + extra.get("pcie_rx_mbps", 0)
                        extra["pcie_util_pct"] = round(min(cur / max_mbps * 100, 100), 2)
                except Exception:  # noqa: BLE001
                    pass

                out.append(
                    GpuSample(
                        vendor=self.vendor,
                        index=i,
                        name=name,
                        util_pct=float(util.gpu),
                        vram_used_mb=mem.used / 1024 / 1024,
                        vram_total_mb=mem.total / 1024 / 1024,
                        temperature_c=float(temp),
                        power_w=float(power),
                        extra=extra,
                    )
                )
            except Exception as e:  # noqa: BLE001
                log.debug("sample gpu %d failed: %s", i, e)
        return out

    def close(self) -> None:
        if self._nvml:
            import contextlib
            with contextlib.suppress(Exception):
                self._nvml.nvmlShutdown()


# PCIe 单向理论峰值带宽(MB/s):按 gen 每 lane 有效速率 × lane 数
# Gen1:250 MB/s, Gen2:500, Gen3:985, Gen4:1969, Gen5:3938 (每 lane)
_PCIE_PER_LANE_MBPS = {1: 250, 2: 500, 3: 985, 4: 1969, 5: 3938, 6: 7877}


def _pcie_max_mbps(gen: int, width: int) -> float:
    per_lane = _PCIE_PER_LANE_MBPS.get(int(gen), 0)
    return per_lane * int(width)
