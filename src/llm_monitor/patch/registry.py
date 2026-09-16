"""Patch 注册表:延迟到目标模块被导入后再打补丁。

用途:vLLM 的模块路径在不同版本有差异(v0/v1、engine 位置变化),因此我们采用
"post-import 回调"模式:注册待打补丁的模块名,在其首次进入 sys.modules 后触发。

实现方式:包装 `builtins.__import__`,单向、幂等、失败降级。
"""
from __future__ import annotations

import builtins
import logging
import sys
import threading
from collections.abc import Callable

log = logging.getLogger("llm_monitor.patch")

_lock = threading.Lock()
_hooks: dict[str, list[Callable[[], None]]] = {}
_applied: set[int] = set()  # id(func) 用作去重键
_in_progress: set[int] = set()  # 正在执行的 hook,防止重入递归
_installed_import_hook = False
_orig_import: Callable | None = None
# 导入钩子重入守卫:hook 执行期间的嵌套 import 不再触发 hook 处理。
# 这是根治 "maximum recursion depth exceeded" 的关键 —— patch 函数内部会
# import 目标模块/torch 等,若不守卫,同一 hook 会在标记 applied 前被反复调用。
_hook_local = threading.local()


def when_imported(module_name: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
    """装饰器:注册当 module_name 导入后要执行的补丁函数。

    补丁函数无参数,内部自行完成 monkey-patch;必须幂等。
    """
    def deco(fn: Callable[[], None]) -> Callable[[], None]:
        with _lock:
            _hooks.setdefault(module_name, []).append(fn)
        # 若已经导入,立即执行一次
        if module_name in sys.modules:
            _run_hooks_for(module_name)
        _ensure_import_hook()
        return fn
    return deco


def _run_hooks_for(module_name: str) -> None:
    for fn in list(_hooks.get(module_name, [])):
        key = id(fn)
        # 在锁内判定并占位:已应用或正在执行(可能是本线程重入,也可能是
        # 另一线程 —— fallback scanner)都跳过,避免重复/递归执行。
        with _lock:
            if key in _applied or key in _in_progress:
                continue
            _in_progress.add(key)
        try:
            fn()
            with _lock:
                _applied.add(key)
            log.info("patched: %s -> %s", module_name, fn.__qualname__)
        except Exception as e:  # noqa: BLE001
            # 真失败:不标记 applied,下次 import 会重试(但重入已被守卫挡住)
            log.warning("patch %s failed: %s", fn.__qualname__, e)
        finally:
            with _lock:
                _in_progress.discard(key)


def _ensure_import_hook() -> None:
    global _installed_import_hook, _orig_import
    if _installed_import_hook:
        return
    _orig_import = builtins.__import__

    def _hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = _orig_import(name, globals, locals, fromlist, level)
        # 重入守卫:若已在处理 hook(嵌套 import),直接返回,不再遍历 hook。
        # 否则 patch 函数内部的 import 会递归触发本函数 → 栈溢出。
        if getattr(_hook_local, "active", False):
            return module
        _hook_local.active = True
        try:
            # 命中任何已注册且已加载的模块名就跑一次,天然幂等
            for mod_name in list(_hooks.keys()):
                if mod_name in sys.modules:
                    _run_hooks_for(mod_name)
        finally:
            _hook_local.active = False
        return module

    builtins.__import__ = _hooked_import
    _installed_import_hook = True


def apply_all_now() -> None:
    """install() 时主动触发一次,处理已加载但尚未打补丁的模块。"""
    for mod_name in list(_hooks.keys()):
        if mod_name in sys.modules:
            _run_hooks_for(mod_name)
    _start_fallback_scanner()


def _start_fallback_scanner() -> None:
    """后台扫描 sys.modules 兜底:处理 importlib.import_module 等绕过 __import__ 的情况。
    只在启动初期扫,60 秒后退出;开销极小。
    """
    global _scanner_started
    if _scanner_started:
        return
    _scanner_started = True

    def _scan():
        import time
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            for mod_name in list(_hooks.keys()):
                if mod_name in sys.modules:
                    _run_hooks_for(mod_name)
            time.sleep(0.2)

    t = threading.Thread(target=_scan, name="llm-monitor-patch-scanner", daemon=True)
    t.start()


_scanner_started = False
