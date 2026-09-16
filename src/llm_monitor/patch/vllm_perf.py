"""细粒度性能拆分:GPU 算子耗时 vs 框架耗时。

原理:
- Python 侧 perf_counter 测的是墙钟时间(wall time)
- CUDA Event 测的是 GPU stream 上 kernel 的实际执行时间(gpu time)
- 框架耗时 = wall - gpu (Python 调度 / 内存管理 / KV cache 索引 / 采样后处理 / ...)

覆盖点:model_runner.execute_model —— vLLM 每个 step 都会调它,
里面就是"框架准备输入 → GPU forward → 框架后处理"。

每个 step 记一条 Transaction,data 里写:
  wall_ms         墙钟总耗时
  gpu_ms          纯 GPU 算子耗时
  framework_ms    框架侧耗时(wall - gpu)
tags 里写 gpu_pct(GPU 占比),点开消息树一眼看出瓶颈在哪。
"""
from __future__ import annotations

import logging
import time

from ..core.api import transaction
from .registry import when_imported
from .util import is_coroutine, wrap_method

log = logging.getLogger("llm_monitor.patch.perf")

_torch = None
_cuda_ok: bool | None = None


def _cuda_available() -> bool:
    """惰性检测:torch 未装 / 无 GPU 时静默降级为只测 wall time。"""
    global _torch, _cuda_ok
    if _cuda_ok is not None:
        return _cuda_ok
    try:
        import torch
        _torch = torch
        _cuda_ok = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        _cuda_ok = False
    return _cuda_ok


def _measure_and_record(tx, wall_start_ns: int, gpu_start, gpu_end) -> None:
    """在 finally 里调用:算出 wall/gpu/framework,写进 tx.data 和 tx.tags。"""
    wall_ms = (time.perf_counter_ns() - wall_start_ns) / 1e6
    tx.data["wall_ms"] = round(wall_ms, 3)

    if gpu_start is not None and gpu_end is not None:
        try:
            # synchronize 会等 GPU 干完再返回,通常这时 kernel 已跑完,开销 ~0
            gpu_end.synchronize()
            gpu_ms = gpu_start.elapsed_time(gpu_end)  # 单位:毫秒
            fw_ms = max(wall_ms - gpu_ms, 0.0)
            tx.data["gpu_ms"] = round(gpu_ms, 3)
            tx.data["framework_ms"] = round(fw_ms, 3)
            if wall_ms > 0:
                tx.tags["gpu_pct"] = f"{gpu_ms / wall_ms * 100:.1f}"
                tx.tags["bottleneck"] = "gpu" if gpu_ms > fw_ms else "framework"
        except Exception:  # noqa: BLE001
            pass


def _wrap_execute_model(orig):
    """支持 sync / async 两种 execute_model 签名。"""

    if is_coroutine(orig):
        async def new_async(self, *args, **kwargs):
            with transaction("vllm.forward", "execute_model") as tx:
                gpu_start = gpu_end = None
                if _cuda_available():
                    gpu_start = _torch.cuda.Event(enable_timing=True)
                    gpu_end = _torch.cuda.Event(enable_timing=True)
                    gpu_start.record()
                wall_start = time.perf_counter_ns()
                try:
                    result = await orig(self, *args, **kwargs)
                    if gpu_end is not None:
                        gpu_end.record()
                    return result
                finally:
                    _measure_and_record(tx, wall_start, gpu_start, gpu_end)
        return new_async

    def new_sync(self, *args, **kwargs):
        with transaction("vllm.forward", "execute_model") as tx:
            gpu_start = gpu_end = None
            if _cuda_available():
                gpu_start = _torch.cuda.Event(enable_timing=True)
                gpu_end = _torch.cuda.Event(enable_timing=True)
                gpu_start.record()
            wall_start = time.perf_counter_ns()
            try:
                result = orig(self, *args, **kwargs)
                if gpu_end is not None:
                    gpu_end.record()
                return result
            finally:
                _measure_and_record(tx, wall_start, gpu_start, gpu_end)
    return new_sync


# ---- 各版本 vLLM 的挂载点 -------------------------------------------

def _try_wrap(module_path: str, class_name: str, method: str) -> None:
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name, None)
        if cls is None or not hasattr(cls, method):
            return
        wrap_method(cls, method, _wrap_execute_model)
        log.info("perf patch: %s.%s.%s", module_path, class_name, method)
    except Exception as e:  # noqa: BLE001
        log.debug("skip perf patch %s.%s.%s: %s", module_path, class_name, method, e)


@when_imported("vllm.worker.model_runner")
def _patch_v0_model_runner():
    for cls in ("ModelRunner", "GPUModelRunnerBase", "GPUModelRunner"):
        _try_wrap("vllm.worker.model_runner", cls, "execute_model")


@when_imported("vllm.worker.worker")
def _patch_v0_worker():
    # Worker.execute_model 是更外层,如果 model_runner 没命中,这里兜底
    _try_wrap("vllm.worker.worker", "Worker", "execute_model")


@when_imported("vllm.v1.worker.gpu_model_runner")
def _patch_v1_model_runner():
    _try_wrap("vllm.v1.worker.gpu_model_runner", "GPUModelRunner", "execute_model")


@when_imported("vllm.v1.worker.gpu_worker")
def _patch_v1_worker():
    _try_wrap("vllm.v1.worker.gpu_worker", "Worker", "execute_model")
