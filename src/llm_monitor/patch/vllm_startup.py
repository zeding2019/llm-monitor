"""追踪 vLLM 启动阶段各步骤耗时。

设计原则：
  - _SPECS 表是唯一需要随 vLLM 版本变化而修改的地方。
  - 每行 (module, class, method, kind)，kind 为阶段名或顶层 Transaction 标记。
  - _try_wrap 遇到不存在的模块/类/方法静默跳过，天然版本兼容。
  - when_imported 按 module 分组自动注册，无需手写每个 @when_imported。

新增 vLLM 版本适配：在 _SPECS 中增加对应行即可，不需要改任何逻辑代码。
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict

from ..core.api import event, transaction
from .registry import when_imported
from .startup_phases import ENGINE_VLLM as _EVLLM
from .startup_phases import record as _sp_rec
from .util import is_coroutine, wrap_method

log = logging.getLogger("llm_monitor.patch.startup")

# wrapper kind 常量
_TOP = "top"            # 同步顶层 Transaction
_TOP_ASYNC = "top_async"  # 自动判断 sync/async 的顶层 Transaction

# ── 声明式补丁表 ────────────────────────────────────────────────────────────
# 格式：(module_path, class_name, method_name, kind)
# kind = _TOP | _TOP_ASYNC | 阶段名(str，对应 startup_phases.PHASE_CODES)
#
# 适配新版本：只需在对应版本区域增行；方法改名则改 method_name 或加新行。
# _try_wrap 会静默跳过当前 vLLM 版本里不存在的方法，不会报错。
_SPECS: list[tuple[str, str, str, str]] = [
    # ── vLLM v0 ─────────────────────────────────────────────────────────────
    ("vllm.engine.llm_engine",           "LLMEngine",         "__init__",                _TOP),
    ("vllm.engine.llm_engine",           "LLMEngine",         "_initialize_kv_caches",   "dist_init"),
    ("vllm.engine.async_llm_engine",     "AsyncLLMEngine",    "__init__",                _TOP_ASYNC),
    ("vllm.worker.worker",               "Worker",            "init_model",              "env_init"),
    ("vllm.worker.worker",               "Worker",            "load_model",              "weights"),
    ("vllm.worker.model_runner",         "ModelRunner",       "load_model",              "weights"),
    ("vllm.worker.model_runner",         "GPUModelRunnerBase","load_model",              "weights"),
    ("vllm.worker.model_runner",         "GPUModelRunner",    "load_model",              "weights"),

    # ── vLLM v1 ─────────────────────────────────────────────────────────────
    ("vllm.v1.engine.llm_engine",        "LLMEngine",         "__init__",                _TOP),
    ("vllm.v1.engine.async_llm",         "AsyncLLM",          "__init__",                _TOP_ASYNC),
    ("vllm.v1.engine.core",              "EngineCore",        "__init__",                _TOP),
    ("vllm.v1.worker.gpu_worker",        "Worker",            "init_device",             "env_init"),
    ("vllm.v1.worker.gpu_worker",        "Worker",            "load_model",              "weights"),
    ("vllm.v1.worker.gpu_worker",        "Worker",            "initialize_cache",        "dist_init"),
    ("vllm.v1.worker.gpu_worker",        "Worker",            "compile_or_warm_up_model","cuda_graph"),
    ("vllm.v1.worker.gpu_model_runner",  "GPUModelRunner",    "load_model",              "weights"),
    ("vllm.v1.worker.gpu_model_runner",  "GPUModelRunner",    "capture_model",           "cuda_graph"),

    # ── shared ───────────────────────────────────────────────────────────────
    ("vllm.transformers_utils.tokenizer_group", "TokenizerGroup", "__init__",            "tokenizer_config"),
]


# ── wrapper 工厂 ─────────────────────────────────────────────────────────────

def _phase_only(phase: str):
    """轻量 wrapper：只计时并记 startup_phases，不建 Transaction。"""
    def wrapper(orig):
        if is_coroutine(orig):
            async def new_fn(self, *args, **kwargs):
                st = time.perf_counter()
                try:
                    return await orig(self, *args, **kwargs)
                finally:
                    _sp_rec(_EVLLM, phase, (time.perf_counter() - st) * 1000.0)
            return new_fn

        def new_fn(self, *args, **kwargs):
            st = time.perf_counter()
            try:
                return orig(self, *args, **kwargs)
            finally:
                _sp_rec(_EVLLM, phase, (time.perf_counter() - st) * 1000.0)
        return new_fn
    return wrapper


def _wrap_engine_init(orig):
    """同步顶层 Transaction，记整个引擎初始化总耗时。"""
    def new_fn(self, *args, **kwargs):
        cls_name = type(self).__name__
        print(f"[llm-monitor] startup: {cls_name}.__init__ begin", flush=True)
        with transaction("vllm.startup", f"{cls_name}.__init__") as tx:
            tx.tags["phase"] = "startup"
            try:
                start = time.perf_counter_ns()
                orig(self, *args, **kwargs)
                elapsed_ms = round((time.perf_counter_ns() - start) / 1e6, 1)
                tx.data["total_ms"] = elapsed_ms
                _sp_rec(_EVLLM, "total", elapsed_ms)
                print(f"[llm-monitor] startup: {cls_name}.__init__ done in {elapsed_ms} ms",
                      flush=True)
            except BaseException as exc:
                event("vllm.startup", "init_failed", status="1",
                      exc=exc.__class__.__name__)
                raise
    return new_fn


def _wrap_async_engine_init(orig):
    """顶层 Transaction，自动适配 async/sync __init__。"""
    if is_coroutine(orig):
        async def new_fn(self, *args, **kwargs):
            cls_name = type(self).__name__
            with transaction("vllm.startup", f"{cls_name}.__init__") as tx:
                tx.tags["phase"] = "startup"
                try:
                    start = time.perf_counter_ns()
                    await orig(self, *args, **kwargs)
                    _ms = round((time.perf_counter_ns() - start) / 1e6, 1)
                    tx.data["total_ms"] = _ms
                    _sp_rec(_EVLLM, "total", _ms)
                except BaseException as exc:
                    event("vllm.startup", "async_init_failed", status="1",
                          exc=exc.__class__.__name__)
                    raise
        return new_fn
    return _wrap_engine_init(orig)


def _make_wrapper(kind: str):
    if kind == _TOP:
        return _wrap_engine_init
    if kind == _TOP_ASYNC:
        return _wrap_async_engine_init
    return _phase_only(kind)


# ── 通用 try_wrap ─────────────────────────────────────────────────────────────

def _try_wrap(module_path: str, class_name: str, method: str, wrapper) -> None:
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name, None)
        if cls is None or not hasattr(cls, method):
            return
        wrap_method(cls, method, wrapper)
        log.info("startup patch: %s.%s.%s", module_path, class_name, method)
        print(f"[llm-monitor] patched startup: {module_path}.{class_name}.{method}",
              flush=True)
    except Exception as e:  # noqa: BLE001
        log.debug("skip startup patch %s.%s.%s: %s", module_path, class_name, method, e)


# ── 自动注册 when_imported ────────────────────────────────────────────────────

def _register_all() -> None:
    """按 module 分组 _SPECS，为每个 module 注册一个 when_imported 回调。"""
    by_module: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for spec in _SPECS:
        by_module[spec[0]].append(spec)

    for mod, specs in by_module.items():
        def _make_patch(specs: list[tuple[str, str, str, str]] = specs):
            def patch_fn() -> None:
                for module, cls, method, kind in specs:
                    _try_wrap(module, cls, method, _make_wrapper(kind))
            return patch_fn
        when_imported(mod)(_make_patch())


_register_all()
