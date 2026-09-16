"""Host 采样:CPU / 内存 / 磁盘 / 网络。基于 psutil。"""
from __future__ import annotations

import time

import psutil

from ..core.api import heartbeat

# 首次调用需要基线,保存上次网络计数
_last_net: dict[str, tuple[int, int, float]] = {}


def sample() -> None:
    now = time.monotonic()

    cpu_pct = psutil.cpu_percent(interval=None)  # 非阻塞
    vm = psutil.virtual_memory()
    values: dict[str, float] = {
        "cpu_pct": float(cpu_pct),
        "mem_used_mb": vm.used / 1024 / 1024,
        "mem_total_mb": vm.total / 1024 / 1024,
        "mem_pct": float(vm.percent),
    }

    # 网络带宽:字节/秒(所有接口合计)
    net = psutil.net_io_counters()
    prev = _last_net.get("all")
    if prev is not None:
        p_sent, p_recv, p_ts = prev
        dt = max(now - p_ts, 1e-6)
        values["net_tx_bps"] = (net.bytes_sent - p_sent) * 8 / dt
        values["net_rx_bps"] = (net.bytes_recv - p_recv) * 8 / dt
    _last_net["all"] = (net.bytes_sent, net.bytes_recv, now)

    heartbeat("host", values)
