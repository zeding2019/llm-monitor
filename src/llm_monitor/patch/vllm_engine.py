"""vLLM 引擎补丁:AsyncLLMEngine.generate / LLMEngine.step / Scheduler.schedule。

设计要点:
- vLLM 版本差异大(v0 SyncEngine、v1 AsyncEngine、v1 core),用多个 when_imported 覆盖
- generate 通常是 async generator(流式产 output),我们要:
    * 建顶层 Transaction(端到端时长)
    * 首个 yield 记 TTFT
    * 每个 step 单独一个 child Transaction
- 若目标方法/类不存在,静默跳过(不同版本 vLLM)
"""
from __future__ import annotations

import logging
import time

from ..core.api import event, metric, transaction
from ..core.models import Heartbeat
from ..core.registry import get_stores
from .registry import when_imported
from .util import is_async_gen, is_coroutine, wrap_method

log = logging.getLogger("llm_monitor.patch.engine")


# ---- AsyncLLMEngine.generate (v0 / v1 通用位置试探) --------------------

def _wrap_generate(orig):
    """orig 可能是 async def 也可能是 async generator。"""

    if is_async_gen(orig):
        async def new_gen(self, *args, **kwargs):
            req_id = kwargs.get("request_id") or (args[2] if len(args) > 2 else "unknown")
            with transaction("vllm.generate", str(req_id)) as tx:
                first = True
                token_count = 0
                start = time.perf_counter_ns()
                try:
                    async for out in orig(self, *args, **kwargs):
                        if first:
                            tx.data["ttft_ns"] = time.perf_counter_ns() - start
                            first = False
                        token_count += 1
                        yield out
                    tx.data["output_events"] = token_count
                    metric("vllm.requests", 1.0)
                except BaseException:
                    event("vllm.generate", "error", status="1", request_id=str(req_id))
                    raise
        return new_gen

    if is_coroutine(orig):
        async def new_coro(self, *args, **kwargs):
            req_id = kwargs.get("request_id") or (args[2] if len(args) > 2 else "unknown")
            with transaction("vllm.generate", str(req_id)):
                try:
                    return await orig(self, *args, **kwargs)
                except BaseException:
                    event("vllm.generate", "error", status="1", request_id=str(req_id))
                    raise
        return new_coro

    def new_sync(self, *args, **kwargs):
        req_id = kwargs.get("request_id", "unknown")
        with transaction("vllm.generate", str(req_id)):
            try:
                return orig(self, *args, **kwargs)
            except BaseException:
                event("vllm.generate", "error", status="1", request_id=str(req_id))
                raise
    return new_sync


def _wrap_step(orig):
    if is_coroutine(orig):
        async def new_async(self, *args, **kwargs):
            with transaction("vllm.step", "step"):
                result = await orig(self, *args, **kwargs)
            _record_step_metrics(result)
            return result
        return new_async

    def new_sync(self, *args, **kwargs):
        with transaction("vllm.step", "step"):
            result = orig(self, *args, **kwargs)
        _record_step_metrics(result)
        return result
    return new_sync


def _record_step_metrics(result) -> None:
    # 尽量宽容:vLLM 不同版本 step 返回结构不同
    try:
        if isinstance(result, list | tuple):
            metric("vllm.step.outputs", float(len(result)))
    except Exception:  # noqa: BLE001
        pass


def _wrap_schedule(orig):
    def new_sync(self, *args, **kwargs):
        with transaction("vllm.schedule", "schedule") as tx:
            result = orig(self, *args, **kwargs)
        try:
            # SchedulerOutputs 常见字段
            preempted = getattr(result, "preempted", None) or getattr(result, "preemption_reason", None)
            if preempted:
                event("vllm.schedule", "preempt")
                tx.tags["preempt"] = "1"
        except Exception:  # noqa: BLE001
            pass
        # 采集队列积压:waiting / running / swapped
        try:
            import time as _time
            waiting = len(getattr(self, "waiting", []))
            running = len(getattr(self, "running", []))
            swapped = len(getattr(self, "swapped", []))
            hb = Heartbeat(
                ts_ns=_time.time_ns(),
                source="vllm.queue",
                values={
                    "waiting": float(waiting),
                    "running": float(running),
                    "swapped": float(swapped),
                },
            )
            get_stores().heartbeats.append(hb)
        except Exception:  # noqa: BLE001
            pass
        # 分类 prefill / decode 并累计,按秒 emit stage heartbeat
        try:
            pt, dt_, pr, dr = _classify_step(result)
            _stage_add(pt, dt_, pr, dr)
        except Exception:  # noqa: BLE001
            pass
        return result
    return new_sync


# ---- prefill / decode 分类 + 速率累计 --------------------------------

def _classify_step(sched_out) -> tuple[int, int, int, int]:
    """从 scheduler_output 拆出 (prefill_tokens, decode_tokens, prefill_reqs, decode_reqs)。
    兼容 v0(SchedulerOutputs) 和 v1(SchedulerOutput)。
    """
    pt = dt = pr = dr = 0
    # v1: SchedulerOutput
    num_scheduled = getattr(sched_out, "num_scheduled_tokens", None)
    new_reqs = getattr(sched_out, "scheduled_new_reqs", None)
    if isinstance(num_scheduled, dict) and new_reqs is not None:
        new_ids = set()
        for r in new_reqs:
            rid = getattr(r, "req_id", None) or getattr(r, "request_id", None)
            if rid is not None:
                new_ids.add(rid)
        for rid, ntok in num_scheduled.items():
            n = int(ntok or 0)
            if rid in new_ids or n > 1:
                pt += n
                pr += 1
            else:
                dt += 1
                dr += 1
        return pt, dt, pr, dr

    # v0: SchedulerOutputs.scheduled_seq_groups
    sgs = getattr(sched_out, "scheduled_seq_groups", None)
    if sgs is not None:
        for sg in sgs:
            is_prompt = getattr(sg, "is_prompt", None)
            token_chunk = int(getattr(sg, "token_chunk_size", 1) or 1)
            if is_prompt or token_chunk > 1:
                pt += token_chunk
                pr += 1
            else:
                dt += token_chunk or 1
                dr += 1
        return pt, dt, pr, dr

    return 0, 0, 0, 0


import threading as _threading  # noqa: E402
import time as _time_mod         # noqa: E402

_stage_lock = _threading.Lock()
_stage_pt = 0             # 累计 prefill tokens
_stage_dt = 0             # 累计 decode tokens
_stage_pr = 0             # 累计 prefill reqs (batch 里的次数)
_stage_dr = 0             # 累计 decode reqs
_stage_steps = 0
_stage_last_emit = 0.0


def _stage_add(pt: int, dt: int, pr: int, dr: int) -> None:
    """schedule 后调用:累加计数,按秒 emit heartbeat 转换成速率。"""
    global _stage_pt, _stage_dt, _stage_pr, _stage_dr, _stage_steps, _stage_last_emit
    with _stage_lock:
        _stage_pt += pt
        _stage_dt += dt
        _stage_pr += pr
        _stage_dr += dr
        _stage_steps += 1
        now = _time_mod.monotonic()
        if _stage_last_emit == 0.0:
            _stage_last_emit = now
            return
        dt_sec = now - _stage_last_emit
        if dt_sec < 1.0:
            return
        pt_rate = _stage_pt / dt_sec
        dt_rate = _stage_dt / dt_sec
        step_rate = _stage_steps / dt_sec
        avg_pf_batch = _stage_pr / max(_stage_steps, 1)
        avg_dc_batch = _stage_dr / max(_stage_steps, 1)
        # 分成两个 source,前端展示时更直观
        get_stores().heartbeats.append(Heartbeat(
            ts_ns=_time_mod.time_ns(),
            source="vllm.prefill",
            values={
                "tokens_per_sec": round(pt_rate, 2),
                "avg_batch_reqs": round(avg_pf_batch, 3),
                "tokens_per_step": round(_stage_pt / max(_stage_steps, 1), 2),
            },
        ))
        get_stores().heartbeats.append(Heartbeat(
            ts_ns=_time_mod.time_ns(),
            source="vllm.decode",
            values={
                "tokens_per_sec": round(dt_rate, 2),
                "avg_batch_reqs": round(avg_dc_batch, 3),
                "steps_per_sec": round(step_rate, 2),
            },
        ))
        # reset
        _stage_pt = _stage_dt = _stage_pr = _stage_dr = _stage_steps = 0
        _stage_last_emit = now


# ---- 各版本模块的挂载点 -------------------------------------------------

def _try_wrap(module_path: str, class_name: str, method: str, wrapper) -> None:
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name, None)
        if cls is None or not hasattr(cls, method):
            return
        wrap_method(cls, method, wrapper)
        log.info("patched %s.%s.%s", module_path, class_name, method)
    except Exception as e:  # noqa: BLE001
        log.debug("skip %s.%s.%s: %s", module_path, class_name, method, e)


@when_imported("vllm.engine.async_llm_engine")
def _patch_async_engine_v0():
    _try_wrap("vllm.engine.async_llm_engine", "AsyncLLMEngine", "generate", _wrap_generate)


@when_imported("vllm.engine.llm_engine")
def _patch_sync_engine_v0():
    _try_wrap("vllm.engine.llm_engine", "LLMEngine", "step", _wrap_step)


@when_imported("vllm.v1.engine.async_llm")
def _patch_async_engine_v1():
    _try_wrap("vllm.v1.engine.async_llm", "AsyncLLM", "generate", _wrap_generate)


@when_imported("vllm.v1.engine.llm_engine")
def _patch_sync_engine_v1():
    _try_wrap("vllm.v1.engine.llm_engine", "LLMEngine", "step", _wrap_step)


@when_imported("vllm.core.scheduler")
def _patch_scheduler_v0():
    _try_wrap("vllm.core.scheduler", "Scheduler", "schedule", _wrap_schedule)


@when_imported("vllm.v1.core.sched.scheduler")
def _patch_scheduler_v1():
    _try_wrap("vllm.v1.core.sched.scheduler", "Scheduler", "schedule", _wrap_schedule)
