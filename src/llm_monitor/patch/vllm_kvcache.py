"""KV cache 命中率(prefix caching)监控。

vLLM 的 prefix caching:同一 prompt 前缀多个请求可以共享 KV block,
命中的部分不用重算 forward,直接跳过 —— TTFT 大幅下降。

关键指标:
  hit_tokens          本次请求命中的前缀 token 数
  prompt_tokens       本次请求的 prompt 总 token 数
  hit_ratio           hit_tokens / prompt_tokens
  gpu_cache_usage     GPU KV block 已用比例 (显存占用度)

vLLM 版本差异较大,我们挂多个可能的点位,失败静默降级:
  v0: vllm.core.block_manager.BlockSpaceManagerV*
  v1: vllm.v1.core.kv_cache_manager.KVCacheManager

统计方式:
  - 每次分配 KV block 时累加 hit/total
  - Heartbeat 定时输出 rolling hit rate + gpu 显存占用
  - 每个请求的 Transaction data 里也记 hit_tokens/hit_ratio
"""
from __future__ import annotations

import logging
import threading
import time

from ..core.api import metric, transaction  # noqa: F401
from ..core.models import Heartbeat
from ..core.registry import get_stores
from .registry import when_imported
from .util import wrap_method

log = logging.getLogger("llm_monitor.patch.kvcache")


# ---- 全局累计计数(所有请求) --------------------------------------------

class _Counters:
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


_counters = _Counters()

# 最新一次 KV cache 状态快照,供 sampler 每秒 re-emit(保证曲线连续)
_last_kv_state: dict = {}


def _record_hit(hit_tokens: int, prompt_tokens: int) -> None:
    """一次分配的命中数据:更新计数,写 Heartbeat 让看板能实时看到。"""
    if prompt_tokens <= 0:
        return
    _counters.add(hit_tokens, prompt_tokens)
    hit, total, _ = _counters.snapshot()
    ratio = hit / total if total else 0.0
    values = {
        "hit_ratio": round(ratio * 100, 2),
        "hit_tokens_total": float(hit),
        "prompt_tokens_total": float(total),
        "recent_hit_tokens": float(hit_tokens),
        "recent_prompt_tokens": float(prompt_tokens),
    }
    hb = Heartbeat(ts_ns=time.time_ns(), source="vllm.kv_cache", values=values)
    get_stores().heartbeats.append(hb)
    # 更新快照,只保留累计/比例这类稳定字段(recent_* 是瞬时的,不 re-emit)
    _last_kv_state.update({
        "hit_ratio": values["hit_ratio"],
        "hit_tokens_total": values["hit_tokens_total"],
        "prompt_tokens_total": values["prompt_tokens_total"],
    })
    metric("vllm.kv_cache.hit_tokens", float(hit_tokens))
    metric("vllm.kv_cache.prompt_tokens", float(prompt_tokens))


# ---- v1: KVCacheManager.get_computed_blocks ---------------------------
# vLLM v1 里 scheduler 会调 kv_cache_manager 查询"这个 request 有多少 token
# 已经在别的 block 里缓存过了",返回值直接就是命中的 token 数或 block 数

def _wrap_v1_get_computed_blocks(orig):
    def new_fn(self, request, *args, **kwargs):
        result = orig(self, request, *args, **kwargs)
        try:
            # 返回结构可能是 (computed_blocks, num_computed_tokens) 或类似
            hit_tokens = 0
            if isinstance(result, tuple) and len(result) >= 2:
                hit_tokens = int(result[1])
            elif hasattr(result, "num_computed_tokens"):
                hit_tokens = int(result.num_computed_tokens)
            prompt_tokens = int(
                getattr(request, "num_prompt_tokens", None)
                or getattr(request, "num_tokens", 0)
                or 0
            )
            if prompt_tokens > 0:
                _record_hit(hit_tokens, prompt_tokens)
        except Exception:  # noqa: BLE001
            pass
        return result
    return new_fn


# ---- v0: PrefixCachingBlockAllocator.get_num_cached_tokens ------------

def _wrap_v0_get_num_cached_tokens(orig):
    def new_fn(self, seq, *args, **kwargs):
        result = orig(self, seq, *args, **kwargs)
        try:
            hit_tokens = int(result)
            prompt_tokens = int(getattr(seq, "get_num_prompt_tokens", lambda: 0)() or 0)
            if prompt_tokens > 0:
                _record_hit(hit_tokens, prompt_tokens)
        except Exception:  # noqa: BLE001
            pass
        return result
    return new_fn


# ---- 定时读 GPU KV cache 使用率 ---------------------------------------
# vLLM 每次 schedule 后会把 gpu_cache_usage_sys 放进 stats,
# 我们直接从 Scheduler 实例上读 block_manager

def _sample_kv_block_usage(scheduler) -> None:
    try:
        bm = getattr(scheduler, "block_manager", None) or getattr(scheduler, "kv_cache_manager", None)
        if bm is None:
            return
        # v0: BlockSpaceManager
        get_free = getattr(bm, "get_num_free_gpu_blocks", None)
        total_getter = getattr(bm, "num_total_gpu_blocks", None) or getattr(bm, "num_gpu_blocks", None)
        if callable(get_free) and total_getter is not None:
            free = get_free()
            total = total_getter() if callable(total_getter) else total_getter
            if total > 0:
                usage = (total - free) / total * 100
                values = {
                    "gpu_cache_usage_pct": round(usage, 2),
                    "gpu_blocks_used": float(total - free),
                    "gpu_blocks_total": float(total),
                }
                hb = Heartbeat(ts_ns=time.time_ns(), source="vllm.kv_cache", values=values)
                get_stores().heartbeats.append(hb)
                _last_kv_state.update(values)
    except Exception:  # noqa: BLE001
        pass


def _wrap_schedule_for_kv_usage(orig):
    """在 Scheduler.schedule 外层再包一次,用于采集 GPU KV 使用率。
    注意:vllm_engine.py 已经 wrap 过 schedule 记调度耗时,
    这里再 wrap 只是叠一层,靠 _ORIG_ATTR 的机制不会重复代理。
    """
    def new_fn(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        _sample_kv_block_usage(self)
        return result
    return new_fn


# ---- 挂载 ---------------------------------------------------------------

def _try_wrap(module_path: str, class_name: str, method: str, wrapper) -> None:
    import importlib
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name, None)
        if cls is None or not hasattr(cls, method):
            return
        wrap_method(cls, method, wrapper)
        log.info("kvcache patch: %s.%s.%s", module_path, class_name, method)
    except Exception as e:  # noqa: BLE001
        log.debug("skip kvcache patch %s.%s.%s: %s", module_path, class_name, method, e)


@when_imported("vllm.v1.core.kv_cache_manager")
def _patch_v1_kv_manager():
    _try_wrap("vllm.v1.core.kv_cache_manager", "KVCacheManager",
              "get_computed_blocks", _wrap_v1_get_computed_blocks)


@when_imported("vllm.core.block_manager")
def _patch_v0_block_mgr():
    for cls in ("BlockSpaceManagerV1", "BlockSpaceManagerV2", "SelfAttnBlockSpaceManager"):
        _try_wrap("vllm.core.block_manager", cls,
                  "get_num_cached_tokens", _wrap_v0_get_num_cached_tokens)


@when_imported("vllm.core.block.prefix_caching_block")
def _patch_v0_prefix_alloc():
    _try_wrap("vllm.core.block.prefix_caching_block", "PrefixCachingBlockAllocator",
              "get_num_cached_tokens", _wrap_v0_get_num_cached_tokens)


@when_imported("vllm.core.scheduler")
def _patch_v0_scheduler_kv_usage():
    _try_wrap("vllm.core.scheduler", "Scheduler",
              "schedule", _wrap_schedule_for_kv_usage)


@when_imported("vllm.v1.core.sched.scheduler")
def _patch_v1_scheduler_kv_usage():
    _try_wrap("vllm.v1.core.sched.scheduler", "Scheduler",
              "schedule", _wrap_schedule_for_kv_usage)
