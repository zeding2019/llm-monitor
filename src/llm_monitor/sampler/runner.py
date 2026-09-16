"""后台采样线程:周期驱动 host + gpu collectors。
额外职责:
- 每 tick 从 transactions 环形缓冲计算 QPS,发 "vllm.throughput" 心跳
- 每 tick re-emit 最新 KV cache 状态,保证曲线连续

GPU 采样单独一个 GpuSamplerThread:
- hy-smi/rocm-smi 这类 subprocess 可能慢到 5-15 秒,不能拖累 host 采样
- 独立间隔,默认 5 秒
- 单卡挂掉/命令超时不影响 host 和 QPS 曲线
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time

from ..core.api import heartbeat
from ..core.registry import get_stores
from . import (
    gpu,  # noqa: F401  触发注册
    host,
)
from .gpu_base import autodetect

log = logging.getLogger("llm_monitor.sampler")


class SamplerThread(threading.Thread):
    """Host 采样 + QPS / KV cache re-emit。GPU 采样已拆到 GpuSamplerThread。"""

    def __init__(self, interval_ms: int = 1000) -> None:
        super().__init__(name="llm-monitor-sampler", daemon=True)
        self.interval = max(interval_ms, 100) / 1000.0
        self._stop = threading.Event()
        self._last_tick_ns = 0

    def run(self) -> None:
        try:
            import psutil
            psutil.cpu_percent(interval=None)
        except Exception:  # noqa: BLE001
            pass

        while not self._stop.is_set():
            t0 = time.monotonic()
            now_ns = time.time_ns()
            try:
                host.sample()
            except Exception as e:  # noqa: BLE001
                log.warning("host sample failed: %s", e)

            # -------- QPS / 请求吞吐时序 --------
            try:
                self._emit_throughput(now_ns)
            except Exception as e:  # noqa: BLE001
                log.warning("throughput sample failed: %s", e)

            # -------- KV cache 状态 re-emit --------
            try:
                self._reemit_kv_cache(now_ns)
            except Exception as e:  # noqa: BLE001
                log.warning("kv_cache re-emit failed: %s", e)

            self._last_tick_ns = now_ns
            elapsed = time.monotonic() - t0
            self._stop.wait(max(0.0, self.interval - elapsed))

    def _emit_throughput(self, now_ns: int) -> None:
        """从 transactions 环形缓冲扫最近一个 tick 内完成的请求,算 QPS 和平均耗时。"""
        if self._last_tick_ns == 0:
            return
        window_ns = now_ns - self._last_tick_ns
        if window_ns <= 0:
            return
        cutoff = self._last_tick_ns
        want = {"http.request", "vllm.generate", "sglang.batch"}
        completed = 0
        total_dur_ns = 0
        for t in get_stores().transactions.snapshot():
            if t.type not in want:
                continue
            end_ns = t.start_ns + t.duration_ns
            if cutoff <= end_ns < now_ns:
                completed += 1
                total_dur_ns += t.duration_ns
        qps = completed / (window_ns / 1e9)
        avg_ms = (total_dur_ns / completed / 1e6) if completed else 0.0
        heartbeat("vllm.throughput", {
            "qps": round(qps, 3),
            "completed": float(completed),
            "avg_latency_ms": round(avg_ms, 2),
        })

    def _reemit_kv_cache(self, now_ns: int) -> None:
        """KV cache 状态原本只在请求触发时才有;此处 re-emit 保证曲线连续。"""
        # vLLM
        try:
            from ..patch import vllm_kvcache
            state = getattr(vllm_kvcache, "_last_kv_state", None)
            if state:
                heartbeat("vllm.kv_cache", dict(state))
        except Exception:  # noqa: BLE001
            pass
        # SGLang
        try:
            from ..patch import sglang
            state = getattr(sglang, "_last_sglang_kv_state", None)
            if state:
                heartbeat("sglang.kv_cache", dict(state))
        except Exception:  # noqa: BLE001
            pass

    def stop(self) -> None:
        self._stop.set()


class GpuSamplerThread(threading.Thread):
    """独立 GPU 采样线程。

    为什么单独一个线程:
    - hy-smi / rocm-smi subprocess 慢时(5-15秒),会阻塞整个采样循环
    - GPU 状态变化本来就慢,采样频率可以低于 host(默认 5 秒)
    - 一个 collector 卡死不影响其他 collector 和 host 采样

    单个 collector 内的 sample() 用超时保护:超时就跳过本次,不阻塞下一次。
    """

    def __init__(
        self,
        interval_ms: int = 5000,
        gpu_backends: list[str] | None = None,
    ) -> None:
        super().__init__(name="llm-monitor-gpu-sampler", daemon=True)
        self.interval = max(interval_ms, 500) / 1000.0
        self._stop = threading.Event()
        self._collectors = autodetect(gpu_backends or ["auto"])
        if self._collectors:
            summary = ", ".join(f"{c.vendor}×{c.device_count()}" for c in self._collectors)
            log.info("GPU collectors active: %s", summary)
            print(f"[llm-monitor] GPU collectors: {summary} (interval={self.interval:.1f}s)",
                  flush=True)
        else:
            log.info("no GPU collector available")
            print("[llm-monitor] no GPU collector available", flush=True)

    def run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            for c in self._collectors:
                try:
                    for s in c.sample():
                        heartbeat(
                            f"gpu:{c.vendor}:{s.index}",
                            {
                                "util_pct": s.util_pct,
                                "vram_used_mb": s.vram_used_mb,
                                "vram_total_mb": s.vram_total_mb,
                                "temperature_c": s.temperature_c,
                                "power_w": s.power_w,
                                **s.extra,
                            },
                        )
                except Exception as e:  # noqa: BLE001
                    log.warning("gpu(%s) sample failed: %s", c.vendor, e)
            elapsed = time.monotonic() - t0
            self._stop.wait(max(0.0, self.interval - elapsed))

    def stop(self) -> None:
        self._stop.set()
        for c in self._collectors:
            with contextlib.suppress(Exception):
                c.close()
