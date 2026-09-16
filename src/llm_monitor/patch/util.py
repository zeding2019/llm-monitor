"""封装 monkey-patch 的公共套路:安全替换类方法,可回滚。"""
from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

_ORIG_ATTR = "__llm_monitor_orig__"


def wrap_method(cls: type, name: str, wrapper: Callable[[Callable], Callable]) -> None:
    """把 cls.name 替换为 wrapper(orig)。幂等(重复替换时先复原)。"""
    if not hasattr(cls, name):
        raise AttributeError(f"{cls!r} has no attribute {name!r}")
    current = getattr(cls, name)
    orig = getattr(current, _ORIG_ATTR, current)
    new_fn = wrapper(orig)
    functools.wraps(orig)(new_fn)
    setattr(new_fn, _ORIG_ATTR, orig)
    setattr(cls, name, new_fn)


def unwrap_method(cls: type, name: str) -> None:
    current = getattr(cls, name, None)
    if current is None:
        return
    orig = getattr(current, _ORIG_ATTR, None)
    if orig is not None:
        setattr(cls, name, orig)


def is_coroutine(fn: Any) -> bool:
    return inspect.iscoroutinefunction(fn)


def is_async_gen(fn: Any) -> bool:
    return inspect.isasyncgenfunction(fn)
