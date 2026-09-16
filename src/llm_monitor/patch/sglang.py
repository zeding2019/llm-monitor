"""SGLang 补丁。

覆盖的关键路径:
  1. HTTP:sglang.srt.entrypoints.http_server 的 FastAPI app + 老版本 srt.server
  2. Scheduler:sglang.srt.managers.scheduler.Scheduler
     - __init__:启动耗时
     - run_batch / process_batch:每 batch tx + prefill/decode 分类
  3. ModelRunner:sglang.srt.model_executor.model_runner.ModelRunner.forward
     CUDA event 分离 GPU 算子 vs 框架耗时

SGLang 的 ForwardMode 枚举:PREFILL / DECODE / EXTEND / MIXED / IDLE
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
import time as _time_mod

from ..core.api import emit_request_trace, emit_sched_step, event, metric, transaction  # noqa: F401
from ..core.models import Heartbeat, RequestTrace, SchedStep
from ..core.registry import get_stores
from .registry import when_imported
from .startup_phases import ENGINE_SGLANG
from .startup_phases import record as _sp_record
from .util import _ORIG_ATTR, is_coroutine, wrap_method
from .vllm_openai import LlmMonitorMiddleware  # 通用 ASGI middleware,复用

log = logging.getLogger("llm_monitor.patch.sglang")
_PID = os.getpid()

# ---- 启动阶段计时(共享 startup_phases 阶段模型)----
_import_t0 = time.perf_counter()   # 本进程 patch 载入时刻(≈进程启动早期)
_cli_import_t = None               # cli.main 载入时刻


def _on_cli_main_import():
    """主进程 CLI 入口模块载入 → 环境/依赖初始化结束。"""
    global _cli_import_t
    _sp_record(ENGINE_SGLANG, "env_init", (time.perf_counter() - _import_t0) * 1000.0)
    _cli_import_t = time.perf_counter()


def _on_cli_serve_call():
    """serve() 被调用 → 参数解析/配置已结束。"""
    if _cli_import_t is None:
        _sp_record(ENGINE_SGLANG, "param_config", 0.0)
        return
    _sp_record(ENGINE_SGLANG, "param_config",
               (time.perf_counter() - _cli_import_t) * 1000.0)


# ==================== 通用工具 ====================

def _try_wrap(mod_path: str, cls_name: str, method: str, wrapper) -> None:
    import importlib
    try:
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name, None)
        if cls is None or not hasattr(cls, method):
            return
        wrap_method(cls, method, wrapper)
        log.info("sglang patch: %s.%s.%s", mod_path, cls_name, method)
        print(f"[llm-monitor] patched sglang: {cls_name}.{method}", flush=True)
    except Exception as e:  # noqa: BLE001
        # 诊断期:失败直接打印完整 traceback,不静默吞 —— 之前 Scheduler 包装失败
        # 全被 log.debug 隐藏,看不到真实原因。
        import traceback as _tb
        print(f"[llm-monitor] PATCH-FAIL {mod_path}.{cls_name}.{method}: {e}", flush=True)
        _tb.print_exc()
        log.warning("sglang patch %s.%s.%s failed: %s", mod_path, cls_name, method, e)


# ==================== HTTP:挂 FastAPI middleware ====================

def _attach_middleware(module_path: str) -> None:
    import importlib
    try:
        mod = importlib.import_module(module_path)
        app = getattr(mod, "app", None)
        if app is not None and callable(getattr(app, "add_middleware", None)):
            app.add_middleware(LlmMonitorMiddleware)
            log.info("MonitorMiddleware added to %s.app", module_path)
            print(f"[llm-monitor] patched sglang HTTP: {module_path}", flush=True)
    except Exception as e:  # noqa: BLE001
        log.debug("attach middleware %s failed: %s", module_path, e)


@when_imported("sglang.srt.entrypoints.http_server")
def _patch_sglang_http_server():
    _attach_middleware("sglang.srt.entrypoints.http_server")


@when_imported("sglang.srt.server")
def _patch_sglang_server_legacy():
    _attach_middleware("sglang.srt.server")


@when_imported("sglang.srt.openai_api.adapter")
def _patch_sglang_openai_adapter():
    # 老版本 openai 兼容层的 handler,先只 log,不 wrap 具体函数
    log.info("sglang.srt.openai_api.adapter detected")


# ==================== Scheduler.run_batch: 每 batch tx ====================

def _get_forward_mode(batch) -> str | None:
    fm = getattr(batch, "forward_mode", None)
    if fm is None:
        return None
    name = getattr(fm, "name", None) or str(fm)
    return str(name).lower()


def _get_batch_size(batch) -> int:
    reqs = getattr(batch, "reqs", None)
    if reqs is not None:
        try:
            return len(reqs)
        except Exception:  # noqa: BLE001
            pass
    return 0


def _get_batch_tokens(batch) -> int:
    """尝试从 batch 拿 total tokens。不同版本 SGLang 字段不一样,都试试。"""
    for attr in ("extend_num_tokens", "input_ids"):
        v = getattr(batch, attr, None)
        if v is None:
            continue
        try:
            if hasattr(v, "shape"):  # torch.Tensor
                return int(v.shape[-1])
            if hasattr(v, "__len__"):
                return len(v)
            if isinstance(v, int):
                return v
        except Exception:  # noqa: BLE001
            continue
    for attr in ("seq_lens", "extend_lens"):
        v = getattr(batch, attr, None)
        if v is None:
            continue
        try:
            if hasattr(v, "sum"):
                return int(v.sum())
            return sum(v)
        except Exception:  # noqa: BLE001
            continue
    return 0


def _wrap_run_batch(orig):
    if is_coroutine(orig):
        async def new_async(self, batch, *args, **kwargs):
            mode = _get_forward_mode(batch)
            _t0 = time.perf_counter_ns()
            with transaction("sglang.batch", mode or "batch") as tx:
                tx.tags["mode"] = mode or "unknown"
                nreqs = _get_batch_size(batch)
                if nreqs:
                    tx.tags["batch_size"] = str(nreqs)
                result = await orig(self, batch, *args, **kwargs)
            wall_ms = (time.perf_counter_ns() - _t0) / 1e6
            _record_stage(mode, batch)
            _record_sched_step(mode, batch, wall_ms)
            _track_requests(mode, batch, wall_ms)
            _sample_sglang_queue(self)
            _sample_sglang_kv_pool(self)
            return result
        return new_async

    def new_sync(self, batch, *args, **kwargs):
        mode = _get_forward_mode(batch)
        _t0 = time.perf_counter_ns()
        with transaction("sglang.batch", mode or "batch") as tx:
            tx.tags["mode"] = mode or "unknown"
            nreqs = _get_batch_size(batch)
            if nreqs:
                tx.tags["batch_size"] = str(nreqs)
            result = orig(self, batch, *args, **kwargs)
        wall_ms = (time.perf_counter_ns() - _t0) / 1e6
        _record_stage(mode, batch)
        _record_sched_step(mode, batch, wall_ms)
        _track_requests(mode, batch, wall_ms)
        _sample_sglang_queue(self)
        _sample_sglang_kv_pool(self)
        return result
    return new_sync


def _record_sched_step(mode: str | None, batch, wall_ms: float) -> None:
    """把一次真实执行的 batch(prefill/decode 等)逐条记下,供引擎事务明细表。"""
    if not mode:
        return
    if mode not in ("prefill", "extend", "mixed", "decode"):
        return
    try:
        from ..core.registry import get_stores as _gs
        _gs().scheduler_steps.append(SchedStep(
            ts_ns=_time_mod.time_ns(),
            pid=_PID,
            mode=mode,
            batch_reqs=_get_batch_size(batch),
            batch_tokens=_get_batch_tokens(batch),
            dur_ms=round(wall_ms, 3),
        ))
    except Exception:  # noqa: BLE001
        pass


# ==================== Prefill/Decode 速率累计 ====================
# 和 vllm_engine 里同样的思路:累加,按秒 emit heartbeat 转成速率

_stage_lock = threading.Lock()
_stage_pt = 0
_stage_dt = 0
_stage_pr = 0
_stage_dr = 0
_stage_steps = 0
_stage_last_emit = 0.0


def _record_stage(mode: str | None, batch) -> None:
    if not mode:
        return
    is_prefill = mode in ("prefill", "extend", "mixed")
    is_decode = mode == "decode"
    if not (is_prefill or is_decode):
        return
    nreqs = _get_batch_size(batch)
    ntokens = _get_batch_tokens(batch)
    global _stage_pt, _stage_dt, _stage_pr, _stage_dr, _stage_steps, _stage_last_emit
    with _stage_lock:
        if is_prefill:
            _stage_pt += ntokens or nreqs
            _stage_pr += nreqs
        else:
            _stage_dt += ntokens or nreqs  # decode 通常 1 token/req
            _stage_dr += nreqs
        _stage_steps += 1
        now = _time_mod.monotonic()
        if _stage_last_emit == 0.0:
            _stage_last_emit = now
            return
        dt_sec = now - _stage_last_emit
        if dt_sec < 1.0:
            return
        get_stores().heartbeats.append(Heartbeat(
            ts_ns=_time_mod.time_ns(),
            source="sglang.prefill",
            values={
                "tokens_per_sec": round(_stage_pt / dt_sec, 2),
                "avg_batch_reqs": round(_stage_pr / max(_stage_steps, 1), 3),
            },
        ))
        get_stores().heartbeats.append(Heartbeat(
            ts_ns=_time_mod.time_ns(),
            source="sglang.decode",
            values={
                "tokens_per_sec": round(_stage_dt / dt_sec, 2),
                "avg_batch_reqs": round(_stage_dr / max(_stage_steps, 1), 3),
                "steps_per_sec": round(_stage_steps / dt_sec, 2),
            },
        ))
        _stage_pt = _stage_dt = _stage_pr = _stage_dr = _stage_steps = 0
        _stage_last_emit = now


# ==================== 单请求生命周期追踪(按 rid)====================
# 目标:到达 → prefill(TTFT)→ 各 decode step(TPOT)→ 完成,每个 rid 聚合成一行。
# 在每个 Scheduler 进程里累计,完成时投递到排空队列,由 DbWriter 落共享库。
# 全程防御式 getattr:字段名跨 SGLang 版本会变,取不到就跳过该字段,绝不抛错。

_req_lock = threading.Lock()
_req_state: dict[str, dict] = {}  # rid -> 累计状态
_REQ_STALE_S = 300.0  # 超过这么久没更新的 state 认为漏检完成,兜底 flush
_REQ_IDLE_S = 10.0    # 已开始产出的 rid 连续这么久没再进 batch → 判定完成
_last_sweep = 0.0     # 上次兜底扫描的 monotonic 秒(节流用)


def _req_rid(req) -> str | None:
    for attr in ("rid", "request_id"):
        v = getattr(req, attr, None)
        if v:
            return str(v)
    return None


def _seq_len(v) -> int:
    if v is None:
        return 0
    try:
        if hasattr(v, "shape"):
            return int(v.shape[-1])
        return len(v)
    except Exception:  # noqa: BLE001
        return 0


def _req_prompt_tokens(req) -> int:
    for attr in ("origin_input_ids", "input_ids", "prompt_ids"):
        n = _seq_len(getattr(req, attr, None))
        if n:
            return n
    return 0


def _req_output_tokens(req) -> int:
    for attr in ("output_ids", "output_token_ids"):
        n = _seq_len(getattr(req, attr, None))
        if n:
            return n
    return 0


def _req_finished(req) -> bool:
    """兼容 finished() 方法 / finished_reason 属性 / finished 布尔。"""
    fn = getattr(req, "finished", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:  # noqa: BLE001
            return False
    if isinstance(fn, bool):
        return fn
    return getattr(req, "finished_reason", None) is not None


def _iter_reqs(batch):
    """拿到 batch 里的 Req 列表。不同版本字段不同:有的 .reqs,有的只留 .decoding_reqs。"""
    for attr in ("reqs", "decoding_reqs"):
        reqs = getattr(batch, attr, None)
        if reqs is None:
            continue
        try:
            out = list(reqs)
            if out:
                return out
        except Exception:  # noqa: BLE001
            continue
    return []


# 运行时探针:每 ~10s 往 DB 写一条 __diag__ 请求追踪,并打日志。
# 用途:确认在真正跑 batch 的进程里 run_batch 包装是否生效、batch 里有没有
# Req、rid 取不取得出来 —— 通过监控页 /api/traces 直接观察,不用猜。
_diag_lock = threading.Lock()
_last_diag_print = 0.0


def _probe_batch(mode: str, batch, reqs: list, rid_count: int) -> None:
    global _last_diag_print
    if "pytest" in sys.modules:  # 单测里不写探针,避免污染断言
        return
    now = _time_mod.monotonic()
    if now - _last_diag_print < 10.0:
        return
    _last_diag_print = now
    n = len(reqs)
    print(f"[llm-monitor] probe mode={mode} batch_reqs={n} with_rid={rid_count} "
          f"attrs={[a for a in ('reqs','decoding_reqs') if getattr(batch, a, None) is not None]}",
          flush=True)
    try:
        from ..core.models import RequestTrace
        rid = f"__diag__:{_PID}"
        emit_request_trace(RequestTrace(
            rid=rid,
            arrival_ts=_time_mod.time_ns(),
            finish_ts=_time_mod.time_ns(),
            status="diag",
            tags={"mode": str(mode), "nreqs": str(n), "with_rid": str(rid_count),
                  "has": ",".join(a for a in ("reqs", "decoding_reqs") if getattr(batch, a, None) is not None)},
        ))
    except Exception as e:  # noqa: BLE001
        log.debug("probe emit failed: %s", e)


def _track_requests(mode: str | None, batch, wall_ms: float) -> None:
    """一个 batch 跑完后,更新其中每个 req 的生命周期状态;完成的 flush 落库。

    完成判定不依赖完成瞬间的 req 状态(SGLang 完成时请求已不在 batch,读不到):
    - 快路径:req.finished() / finished_reason 存在就直接定案;
    - 主路径:某个已开始产出的 rid 连续 _REQ_IDLE_S 秒不再出现在 batch 里,
      认为已结束,用"最后一次出现时间"做 finish_ts —— e2e 不受空闲窗影响。
    token 数在每帧出现时累计,避免结束时请求已消失读不到。
    """
    if not mode:
        return
    is_prefill = mode in ("prefill", "extend", "mixed")
    is_decode = mode == "decode"
    if not (is_prefill or is_decode):
        return
    now = _time_mod.time_ns()
    reqs_all = _iter_reqs(batch)
    emit_now: list[tuple[str, dict]] = []
    with _req_lock:
        for req in reqs_all:
            rid = _req_rid(req)
            if rid is None:
                continue
            st = _req_state.get(rid)
            if st is None:
                st = {
                    "arrival_ts": now, "first_token_ts": None,
                    "prefill_ms": 0.0, "decode_ms": 0.0, "decode_steps": 0,
                    "prompt_tokens": 0, "output_tokens": 0,
                    "last_seen": now, "last_update": now,
                }
                _req_state[rid] = st
            st["last_seen"] = now
            st["last_update"] = now
            if is_prefill:
                # 首个 prefill batch 的耗时近似为该 req 的 prefill 时间
                if st["prefill_ms"] == 0.0:
                    st["prefill_ms"] = round(wall_ms, 3)
                if st["prompt_tokens"] == 0:
                    st["prompt_tokens"] = _req_prompt_tokens(req)
            else:  # decode
                st["decode_steps"] += 1
                st["decode_ms"] += wall_ms
                if st["first_token_ts"] is None:
                    st["first_token_ts"] = now  # 首个 decode step 产出首 token
                ot = _req_output_tokens(req)
                if ot > st["output_tokens"]:
                    st["output_tokens"] = ot
                if st["prompt_tokens"] == 0:
                    st["prompt_tokens"] = _req_prompt_tokens(req)
            if _req_finished(req):
                st["finish_ts"] = st["last_seen"]
                st["status"] = str(getattr(req, "finished_reason", "") or "0")
                _req_state.pop(rid, None)
                emit_now.append((rid, st))
    for rid, st in emit_now:
        _emit_trace(rid, st)
    # 已开始产出的 rid 若长时间没再进 batch → 认为结束(不依赖 req.finished)
    for rid, st in _finalize_idle(now):
        _emit_trace(rid, st)
    # 兜底:长时间无更新的僵尸 state(从未产出的也清掉)
    _sweep_stale(now)


def _finalize_idle(now_ns: int) -> list[tuple[str, dict]]:
    """把已开始产出、但连续 _REQ_IDLE_S 秒没再进 batch 的 rid 定案为完成。

    单独抽出来,以便:batch 路径 + 独立心跳线程都调 —— scheduler 空闲时没有新
    batch,若只靠 _track_requests 触发,最后一个请求永远不会被定案落库。
    """
    emit: list[tuple[str, dict]] = []
    idle_ns = int(_REQ_IDLE_S * 1e9)
    with _req_lock:
        for rid, st in list(_req_state.items()):
            if st.get("first_token_ts") is None:
                continue  # 还在排队等 prefill,不判定
            if now_ns - st.get("last_seen", now_ns) > idle_ns:
                st["finish_ts"] = st["last_seen"]
                st.setdefault("status", "0")
                _req_state.pop(rid, None)
                emit.append((rid, st))
    return emit


def _start_trace_ticker() -> None:
    """独立心跳:每 ~3s 定案一次空闲完成的请求。scheduler 空闲也照常落库。"""
    if "pytest" in sys.modules:
        return

    def _tick():
        while True:
            try:
                now = _time_mod.time_ns()
                for rid, st in _finalize_idle(now):
                    _emit_trace(rid, st)
                _sweep_stale(now)
            except Exception:  # noqa: BLE001
                pass
            _time_mod.sleep(3.0)

    threading.Thread(target=_tick, name="llm-monitor-trace-ticker", daemon=True).start()


def _emit_trace(rid: str, st: dict) -> None:
    arrival = st["arrival_ts"]
    first_tok = st.get("first_token_ts")
    finish = st.get("finish_ts") or _time_mod.time_ns()
    decode_steps = st.get("decode_steps", 0)
    decode_ms = st.get("decode_ms", 0.0)
    ttft_ms = ((first_tok - arrival) / 1e6) if first_tok else 0.0
    tpot_ms = (decode_ms / decode_steps) if decode_steps else 0.0
    e2e_ms = (finish - arrival) / 1e6
    emit_request_trace(RequestTrace(
        rid=rid,
        arrival_ts=arrival,
        first_token_ts=first_tok,
        finish_ts=finish,
        prefill_ms=round(st.get("prefill_ms", 0.0), 3),
        decode_ms=round(decode_ms, 3),
        decode_steps=decode_steps,
        prompt_tokens=st.get("prompt_tokens", 0),
        output_tokens=st.get("output_tokens", 0),
        ttft_ms=round(ttft_ms, 3),
        tpot_ms=round(tpot_ms, 3),
        e2e_ms=round(e2e_ms, 3),
        status=st.get("status", "0"),
        pid=_PID,
    ))


def _sweep_stale(now_ns: int) -> None:
    """完成检测万一漏判,超时的 state 也 flush 掉,避免内存泄漏。
    每 batch 都被调用,故节流到最多每 ~30s 扫一次,避免高吞吐下的锁竞争。
    """
    global _last_sweep
    mono = _time_mod.monotonic()
    if mono - _last_sweep < 30.0:
        return
    _last_sweep = mono
    stale: list[tuple[str, dict]] = []
    cutoff = now_ns - int(_REQ_STALE_S * 1e9)
    with _req_lock:
        for rid, st in list(_req_state.items()):
            if st.get("last_update", now_ns) < cutoff:
                st.setdefault("status", "stale")
                _req_state.pop(rid, None)
                stale.append((rid, st))
    for rid, st in stale:
        _emit_trace(rid, st)


# ==================== ModelRunner.forward: GPU vs 框架 ====================

_torch = None
_cuda_ok: bool | None = None


def _cuda_available() -> bool:
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


def _wrap_model_forward(orig):
    def new_fn(self, *args, **kwargs):
        with transaction("sglang.forward", "model.forward") as tx:
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
                wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
                tx.data["wall_ms"] = round(wall_ms, 3)
                if gpu_start is not None and gpu_end is not None:
                    try:
                        gpu_end.synchronize()
                        gpu_ms = gpu_start.elapsed_time(gpu_end)
                        fw_ms = max(wall_ms - gpu_ms, 0.0)
                        tx.data["gpu_ms"] = round(gpu_ms, 3)
                        tx.data["framework_ms"] = round(fw_ms, 3)
                        if wall_ms > 0:
                            tx.tags["gpu_pct"] = f"{gpu_ms / wall_ms * 100:.1f}"
                            tx.tags["bottleneck"] = "gpu" if gpu_ms > fw_ms else "framework"
                    except Exception:  # noqa: BLE001
                        pass
    return new_fn


# ==================== 启动阶段追踪 ====================

def _wrap_startup(orig):
    """顶层 Transaction，记 Engine.__init__ 总耗时。"""
    def new_fn(self, *args, **kwargs):
        cls_name = type(self).__name__
        print(f"[llm-monitor] sglang startup: {cls_name}.__init__ begin", flush=True)
        with transaction("sglang.startup", f"{cls_name}.__init__") as tx:
            tx.tags["phase"] = "startup"
            try:
                start = time.perf_counter_ns()
                orig(self, *args, **kwargs)
                elapsed_ms = round((time.perf_counter_ns() - start) / 1e6, 1)
                tx.data["total_ms"] = elapsed_ms
                print(f"[llm-monitor] sglang startup: {cls_name}.__init__ done in {elapsed_ms} ms",
                      flush=True)
            except BaseException as exc:
                event("sglang.startup", "init_failed", status="1",
                      exc=exc.__class__.__name__)
                raise
    return new_fn


# ── 声明式启动阶段补丁表 ─────────────────────────────────────────────────────
# 格式：(module_path, class_name, method_name, kind)
# kind = "_top" → _wrap_startup（建顶层 Transaction）
# kind = 阶段名  → _wrap_phase(phase)（只计时，写 startup_phases）
#
# 适配新版本：在对应版本区域增行；方法改名则改 method_name 或加新行。
# _try_wrap 会静默跳过当前版本里不存在的方法。
# 同时驱动 when_imported 注册 和 polling 兜底，二者共享同一份数据。
_STARTUP_SPECS: list[tuple[str, str, str, str]] = [
    # ── SGLang 0.4.x – 0.5.x ────────────────────────────────────────────────
    ("sglang.srt.entrypoints.engine",          "Engine",           "__init__",               "_top"),
    ("sglang.srt.model_executor.model_runner", "ModelRunner",      "load_model",             "weights"),
    ("sglang.srt.model_executor.model_runner", "ModelRunner",      "init_torch_distributed", "dist_init"),
    ("sglang.srt.managers.tokenizer_manager",  "TokenizerManager", "__init__",               "tokenizer_config"),
]


def _make_startup_wrapper(kind: str):
    if kind == "_top":
        return _wrap_startup
    return _wrap_phase(kind)


def _register_startup_patches() -> None:
    """按 module 分组 _STARTUP_SPECS，自动注册 when_imported 回调。"""
    from collections import defaultdict
    by_module: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for spec in _STARTUP_SPECS:
        by_module[spec[0]].append(spec)

    for mod, specs in by_module.items():
        def _make_patch(specs: list[tuple[str, str, str, str]] = specs):
            def patch_fn() -> None:
                for module, cls, method, kind in specs:
                    _try_wrap(module, cls, method, _make_startup_wrapper(kind))
            return patch_fn
        when_imported(mod)(_make_patch())


# ==================== 挂载点 ====================

@when_imported("sglang.srt.managers.scheduler")
def _patch_scheduler_module():
    # batch 处理:两个可能的方法名
    _try_wrap("sglang.srt.managers.scheduler", "Scheduler", "run_batch", _wrap_run_batch)
    _try_wrap("sglang.srt.managers.scheduler", "Scheduler", "process_batch", _wrap_run_batch)
    # 启动 + 初始化后给真实 tree_cache 挂 KV 命中钩子
    _try_wrap("sglang.srt.managers.scheduler", "Scheduler", "__init__",
              _wrap_scheduler_init)


# ---- 启动阶段计时器 wrapper(共享阶段模型)----

def _wrap_phase(phase: str):
    """包一个方法,执行期间计入 startup_phases 的指定阶段。"""
    def wrapper(orig):
        if is_coroutine(orig):
            async def new_async(self, *a, **k):
                t0 = time.perf_counter()
                try:
                    return await orig(self, *a, **k)
                finally:
                    _sp_record(ENGINE_SGLANG, phase, (time.perf_counter() - t0) * 1000.0)
            return new_async

        def new_sync(self, *a, **k):
            t0 = time.perf_counter()
            try:
                return orig(self, *a, **k)
            finally:
                _sp_record(ENGINE_SGLANG, phase, (time.perf_counter() - t0) * 1000.0)
        return new_sync
    return wrapper


@when_imported("sglang.cli.main")
def _patch_cli_main_phase():
    _on_cli_main_import()


@when_imported("sglang.cli.serve")
def _patch_cli_serve_phase():
    """参数解析/配置阶段终点:serve() 被调用时(参数已解析完)。"""
    import importlib
    try:
        mod = importlib.import_module("sglang.cli.serve")
        fn = getattr(mod, "serve", None)
        if not callable(fn):
            return
        def _marker(*a, **k):
            _on_cli_serve_call()
            return fn(*a, **k)
        mod.serve = _marker
    except Exception:  # noqa: BLE001
        pass


_register_startup_patches()


# ==================== KV cache: RadixCache 命中率 + 池占用 ====================
# SGLang 用 RadixAttention:一棵 radix tree 存 KV block 前缀。
# match_prefix(key) 返回命中的 token 数,插入时同步更新。
# 池占用:Scheduler 里有 token_to_kv_pool,读它的 available_size / size。

class _KVCounters:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.hit_tokens = 0
        self.total_tokens = 0
        self.requests = 0

    def add(self, hit: int, total: int) -> None:
        with self.lock:
            self.hit_tokens += hit
            self.total_tokens += total
            self.requests += 1

    def snapshot(self) -> tuple[int, int, int]:
        with self.lock:
            return self.hit_tokens, self.total_tokens, self.requests


_kv_counters = _KVCounters()
_last_sglang_kv_state: dict = {}


def _record_sglang_kv_hit(hit_tokens: int, prompt_tokens: int) -> None:
    if prompt_tokens <= 0:
        return
    _kv_counters.add(hit_tokens, prompt_tokens)
    hit, total, _ = _kv_counters.snapshot()
    ratio = hit / total if total else 0.0
    values = {
        "hit_ratio": round(ratio * 100, 2),
        "hit_tokens_total": float(hit),
        "prompt_tokens_total": float(total),
        "recent_hit_tokens": float(hit_tokens),
        "recent_prompt_tokens": float(prompt_tokens),
    }
    get_stores().heartbeats.append(Heartbeat(
        ts_ns=_time_mod.time_ns(), source="sglang.kv_cache", values=values,
    ))
    _last_sglang_kv_state.update({
        "hit_ratio": values["hit_ratio"],
        "hit_tokens_total": values["hit_tokens_total"],
        "prompt_tokens_total": values["prompt_tokens_total"],
    })


def _record_prefix_metrics(args, kwargs, result) -> None:
    """从 match_prefix 调用里提取 (命中, prompt总长) → 累计命中率。
    兼容新旧 API:新版 params:MatchPrefixParams→MatchResult(host_hit_length),
    旧版 key→list 长度。失败静默,绝不影响推理。"""
    try:
        hit = None
        for attr in ("host_hit_length", "device_indices"):
            v = getattr(result, attr, None)
            if v is None:
                continue
            if isinstance(v, int):
                hit = v
            else:
                n = _seq_len(v)
                if n:
                    hit = n
            if hit is not None:
                break
        if hit is None:
            candidate = result
            if isinstance(result, tuple) and result:
                candidate = result[0]
            if hasattr(candidate, "__len__"):
                hit = len(candidate)
            elif isinstance(candidate, int):
                hit = candidate
            else:
                hit = 0

        total = 0
        params = args[0] if args else kwargs.get("params")
        req = getattr(params, "req", None)
        if req is not None:
            for attr in ("origin_input_ids", "input_ids", "prompt_ids"):
                n = _seq_len(getattr(req, attr, None))
                if n:
                    total = n
                    break
        if not total:
            key = getattr(params, "key", None)
            if key is None and kwargs.get("key") is not None:
                key = kwargs["key"]
            if key is not None:
                if hasattr(key, "__len__") and not isinstance(key, (str, bytes)):
                    total = len(key)
                else:
                    for attr in ("token_ids", "ids", "key"):
                        n = _seq_len(getattr(key, attr, None))
                        if n:
                            total = n
                            break
        if total > 0 and (hit or 0) >= 0:
            _record_sglang_kv_hit(int(hit or 0), total)
    except Exception:  # noqa: BLE001
        pass


def _wrap_match_prefix(orig):
    """RadixCache.match_prefix 类级钩子(兼容旧路径)。0.5 主路径见 _patch_tree_cache_hook。"""
    def new_fn(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        _record_prefix_metrics(args, kwargs, result)
        return result
    return new_fn


def _patch_tree_cache_hook(scheduler) -> None:
    """Scheduler 实际用的 tree_cache 可能是 HiRadixCache / HiMambaRadixCache /
    MambaRadixCache / UnifiedRadixCache / RadixCache 等任一子类 —— 0.5 起
    RadixCache 不是唯一实现,类级钩子会漏。故在 __init__ 后对实例动态挂
    match_prefix 命中钩子(实例属性覆盖类方法,命中任何子类)。"""
    cache = getattr(scheduler, "tree_cache", None)
    if cache is None:
        cache = getattr(scheduler, "radix_cache", None)
    if cache is None:
        return
    name = "match_prefix"
    if getattr(cache, "_llm_mon_hooked", False):
        return
    meth = getattr(cache, name, None)
    if meth is None:
        return
    cache._llm_mon_hooked = True
    orig_bound = meth  # 已绑定的方法(含 self)

    def hooked(params, *a, **k):
        result = orig_bound(params, *a, **k)
        _record_prefix_metrics((params,), k, result)
        return result

    try:
        setattr(cache, name, hooked)
    except Exception:  # noqa: BLE001
        cache._llm_mon_hooked = False


def _wrap_scheduler_init(orig):
    """Scheduler.__init__:保留原逻辑,并在实例就绪后给真实 tree_cache 挂命中钩子。"""
    import contextlib as _ctx

    def new_init(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        with _ctx.suppress(Exception):
            _patch_tree_cache_hook(self)
            # 该进程从 import 到调度器就绪的总时长(近似整机启动尾部)
            _sp_record(ENGINE_SGLANG, "total",
                       (time.perf_counter() - _import_t0) * 1000.0)
        return result
    return new_init


def _sample_sglang_kv_pool(scheduler) -> None:
    """从 Scheduler 读 token_to_kv_pool 的容量与已用。"""
    try:
        pool = (getattr(scheduler, "token_to_kv_pool_allocator", None)
                or getattr(scheduler, "token_to_kv_pool", None))
        if pool is None:
            return
        # 兼容多种字段
        size = int(getattr(pool, "size", 0) or 0)
        if size == 0:
            size = int(getattr(pool, "total_size", 0) or 0)
        available = getattr(pool, "available_size", None)
        if callable(available):
            available = available()
        elif available is None:
            available = getattr(pool, "free_size", None)
            if callable(available):
                available = available()
        if size > 0 and available is not None:
            available = int(available)
            used = size - available
            usage_pct = used / size * 100
            values = {
                "gpu_cache_usage_pct": round(usage_pct, 2),
                "gpu_blocks_used": float(used),
                "gpu_blocks_total": float(size),
            }
            get_stores().heartbeats.append(Heartbeat(
                ts_ns=_time_mod.time_ns(), source="sglang.kv_cache", values=values,
            ))
            _last_sglang_kv_state.update(values)
    except Exception:  # noqa: BLE001
        pass


# ==================== 请求队列长度监控 ====================
# 对齐 vllm.queue 的 waiting/running/swapped 三态。SGLang 的 Scheduler 里:
#   - waiting_queue:   list[Req]   等待调度(还没进 running batch)
#   - running_batch:   ScheduleBatch  当前正在跑的 batch(等价于 running)
#   - being_chunked_req / cur_batch:  chunked prefill 中间态
#   - retracted / swapped:            少数版本有 preemption/retract 队列
# 字段名跨版本有变,都用 getattr 兜底。

def _q_len(obj) -> int:
    if obj is None:
        return 0
    reqs = getattr(obj, "reqs", None)
    if reqs is not None:
        try:
            return len(reqs)
        except Exception:  # noqa: BLE001
            return 0
    try:
        return len(obj)
    except Exception:  # noqa: BLE001
        return 0


def _sample_sglang_queue(scheduler) -> None:
    """从 Scheduler 读队列长度,emit 一条 sglang.queue heartbeat。"""
    try:
        waiting = _q_len(getattr(scheduler, "waiting_queue", None))
        running = _q_len(
            getattr(scheduler, "running_batch", None)
            or getattr(scheduler, "cur_batch", None)
        )
        chunked_obj = (getattr(scheduler, "being_chunked_req", None)
                       or getattr(scheduler, "chunked_req", None)
                       or getattr(scheduler, "being_chunked_reqs", None))
        if chunked_obj is None:
            chunked = 0
        elif hasattr(chunked_obj, "__len__"):
            chunked = len(chunked_obj)
        else:
            chunked = 1
        swapped = _q_len(
            getattr(scheduler, "retracted_queue", None)
            or getattr(scheduler, "swapped", None)
        )
        grammar_pending = _q_len(getattr(scheduler, "grammar_queue", None))

        values = {
            "waiting": float(waiting),
            "running": float(running),
            "chunked": float(chunked),
            "swapped": float(swapped),
        }
        if grammar_pending:
            values["grammar_pending"] = float(grammar_pending)

        get_stores().heartbeats.append(Heartbeat(
            ts_ns=_time_mod.time_ns(),
            source="sglang.queue",
            values=values,
        ))
    except Exception:  # noqa: BLE001
        pass


@when_imported("sglang.srt.mem_cache.radix_cache")
def _patch_radix_cache():
    _try_wrap("sglang.srt.mem_cache.radix_cache", "RadixCache", "match_prefix",
              _wrap_match_prefix)


@when_imported("sglang.srt.mem_cache.chunk_cache")
def _patch_chunk_cache():
    # 老一点的版本可能用 ChunkCache;接口类似
    _try_wrap("sglang.srt.mem_cache.chunk_cache", "ChunkCache", "match_prefix",
              _wrap_match_prefix)


# ==================== 兜底:轮询补丁 Scheduler ====================
# 实测(0.5.12):import 钩子在 SGLang 复杂 import 图下对 scheduler 分发不可靠
# (注册了、模块也加载了,但 _run_hooks_for 不触发),而直接 _try_wrap 一次成功。
# 故在 import 本模块的每个进程里起一个轮询线程:前 5 分钟内每 2 秒幂等补一次,
# 一旦 Scheduler 进 sys.modules 就打上 run_batch/__init__。带守卫,不会叠层。

_poller_started = False


def _method_patched(cls, method: str) -> bool:
    fn = getattr(cls, method, None)
    return bool(fn is not None and hasattr(fn, _ORIG_ATTR))


def _patch_scheduler_best_effort() -> bool:
    try:
        if "sglang.srt.managers.scheduler" not in sys.modules:
            return False
        mod = __import__("sglang.srt.managers.scheduler", fromlist=["Scheduler"])
        cls = getattr(mod, "Scheduler", None)
        if cls is None:
            return False
        if not _method_patched(cls, "run_batch"):
            _try_wrap("sglang.srt.managers.scheduler", "Scheduler", "run_batch", _wrap_run_batch)
        if not _method_patched(cls, "__init__"):
            _try_wrap("sglang.srt.managers.scheduler", "Scheduler", "__init__",
                      _wrap_scheduler_init)
        return _method_patched(cls, "run_batch") and _method_patched(cls, "__init__")
    except Exception:  # noqa: BLE001
        return False


def _start_scheduler_poller() -> None:
    global _poller_started
    if _poller_started:
        return
    if "pytest" in sys.modules:
        return
    _poller_started = True

    def _run():
        deadline = _time_mod.monotonic() + 300.0  # 最多轮询 5 分钟
        while _time_mod.monotonic() < deadline:
            try:
                if _patch_scheduler_best_effort():
                    return
            except Exception:  # noqa: BLE001
                pass
            _time_mod.sleep(2.0)

    threading.Thread(target=_run, name="llm-monitor-sglang-poller", daemon=True).start()


def _start_startup_poller() -> None:
    """轮询兜底：when_imported 对某些模块未触发时，主动检查并补打 _STARTUP_SPECS。"""
    if "pytest" in sys.modules:
        return

    def _run():
        deadline = _time_mod.monotonic() + 300.0
        while _time_mod.monotonic() < deadline:
            for module, cls_name, method, kind in _STARTUP_SPECS:
                try:
                    if module not in sys.modules:
                        continue
                    mod = __import__(module, fromlist=[cls_name])
                    cls = getattr(mod, cls_name, None)
                    if cls is None or _method_patched(cls, method):
                        continue
                    _try_wrap(module, cls_name, method, _make_startup_wrapper(kind))
                except Exception:  # noqa: BLE001
                    pass
            _time_mod.sleep(2.0)

    threading.Thread(target=_run, name="llm-monitor-startup-poller", daemon=True).start()


_start_scheduler_poller()
_start_trace_ticker()
_start_startup_poller()
