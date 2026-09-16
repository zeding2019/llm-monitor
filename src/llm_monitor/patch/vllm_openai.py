"""vLLM OpenAI server 补丁:HTTP 层入口 → 顶层 Transaction。

覆盖:
- /v1/chat/completions
- /v1/completions
- /v1/embeddings

策略:vLLM 用 FastAPI 定义路由,handler 是 async def。我们把 handler
包一层,建 `http.request` 顶层 tx,tag 上 request_id、model、client_ip 等。
后续 vllm.generate 会因为 contextvars 自动挂为子节点。

vLLM 版本差异较大,因此走两种路径:
1) 直接补丁 `vllm.entrypoints.openai.api_server` 里的 handler 函数
2) 提供 FastAPI Middleware 兜底(用户可手动挂载)
"""
from __future__ import annotations

import logging
import time

from ..core.api import event, transaction
from .registry import when_imported
from .util import is_async_gen, is_coroutine

log = logging.getLogger("llm_monitor.patch.openai")


# ---- HTTP handler wrapper ----------------------------------------------

def _wrap_http_handler(op_name: str):
    """返回一个 wrapper(orig),用于包 async def handler。"""

    def wrapper(orig):
        if is_async_gen(orig):
            async def gen_new(*args, **kwargs):
                start = time.perf_counter_ns()
                async with _http_tx(op_name, args, kwargs, start) as tx:
                    first = True
                    async for chunk in orig(*args, **kwargs):
                        if first:
                            tx.data["ttfb_ns"] = time.perf_counter_ns() - start
                            first = False
                        yield chunk
            return gen_new

        if is_coroutine(orig):
            async def coro_new(*args, **kwargs):
                start = time.perf_counter_ns()
                async with _http_tx(op_name, args, kwargs, start) as tx:  # noqa: F841
                    return await orig(*args, **kwargs)
            return coro_new

        def sync_new(*args, **kwargs):
            with transaction("http.request", op_name) as tx:
                _fill_request_tags(tx, args, kwargs)
                return orig(*args, **kwargs)
        return sync_new
    return wrapper


class _http_tx:  # noqa: N801
    """async context manager 包 transaction(),自动填 tags。"""
    def __init__(self, op_name, args, kwargs, start):
        self.op_name = op_name
        self.args = args
        self.kwargs = kwargs
        self.start = start
        self._cm = None
        self.tx = None

    async def __aenter__(self):
        self._cm = transaction("http.request", self.op_name)
        self.tx = self._cm.__enter__()
        _fill_request_tags(self.tx, self.args, self.kwargs)
        return self.tx

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is not None:
            event("http.request", "error", status="1", op=self.op_name,
                  exc=exc_type.__name__)
        return self._cm.__exit__(exc_type, exc, tb)


def _fill_request_tags(tx, args, kwargs) -> None:
    """尽力从 handler 参数中提取 request_id / model / client_ip。"""
    try:
        # vLLM handler 常见签名:(request: ChatCompletionRequest, raw_request: Request)
        for a in list(args) + list(kwargs.values()):
            if a is None:
                continue
            # Pydantic body: 有 model 字段
            model = getattr(a, "model", None)
            if model and "model" not in tx.tags:
                tx.tags["model"] = str(model)
            # FastAPI Request: 有 client
            client = getattr(a, "client", None)
            if client is not None:
                host = getattr(client, "host", None)
                if host and "client_ip" not in tx.tags:
                    tx.tags["client_ip"] = str(host)
            # OpenAI 有些实现把 request_id 塞在 headers
            headers = getattr(a, "headers", None)
            if headers is not None:
                try:
                    rid = headers.get("x-request-id") if hasattr(headers, "get") else None
                    if rid:
                        tx.tags["request_id"] = str(rid)
                        tx.name = str(rid)
                except Exception:  # noqa: BLE001
                    pass
    except Exception as e:  # noqa: BLE001
        log.debug("fill http tags failed: %s", e)


# ---- 目标 handler 名(vLLM 各版本尽量覆盖)-----------------------------

_HANDLERS = [
    ("create_chat_completion", "POST /v1/chat/completions"),
    ("create_completion", "POST /v1/completions"),
    ("create_embedding", "POST /v1/embeddings"),
]


def _patch_module_handlers(module_name: str) -> None:
    import importlib
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:  # noqa: BLE001
        log.debug("import %s failed: %s", module_name, e)
        return
    for fn_name, op in _HANDLERS:
        fn = getattr(mod, fn_name, None)
        if fn is None or not callable(fn):
            continue
        # 用 setattr 直接替换 module-level 函数
        try:
            wrapper = _wrap_http_handler(op)
            new_fn = wrapper(fn)
            setattr(mod, fn_name, new_fn)
            log.info("patched http handler %s.%s", module_name, fn_name)
        except Exception as e:  # noqa: BLE001
            log.debug("patch %s.%s failed: %s", module_name, fn_name, e)


@when_imported("vllm.entrypoints.openai.api_server")
def _patch_openai_api_server():
    import importlib
    try:
        mod = importlib.import_module("vllm.entrypoints.openai.api_server")
        # 找 FastAPI app 对象,直接挂 ASGI middleware — 无论 vLLM 版本如何都生效
        app = getattr(mod, "app", None)
        if app is not None and callable(getattr(app, "add_middleware", None)):
            app.add_middleware(LlmMonitorMiddleware)
            log.info("LlmMonitorMiddleware added to vllm FastAPI app")
        else:
            log.warning("vllm api_server app not found, fallback to handler patch")
            _patch_module_handlers("vllm.entrypoints.openai.api_server")
    except Exception as e:  # noqa: BLE001
        log.debug("patch openai api_server failed: %s", e)


# ---- 兜底:FastAPI Middleware,用户可显式挂载 ------------------------

class LlmMonitorMiddleware:
    """兜底 ASGI middleware。用法:

        from llm_monitor.patch.vllm_openai import LlmMonitorMiddleware
        app.add_middleware(LlmMonitorMiddleware)
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method = scope.get("method", "?")
        path = scope.get("path", "/")
        op = f"{method} {path}"
        with transaction("http.request", op) as tx:
            # 从 headers 拿 request-id / client
            for k, v in scope.get("headers", []):
                if k == b"x-request-id":
                    tx.tags["request_id"] = v.decode()
                    tx.name = v.decode()
            client = scope.get("client")
            if client:
                tx.tags["client_ip"] = client[0]

            status_holder = {"code": 200}

            async def _send(msg):
                if msg["type"] == "http.response.start":
                    status_holder["code"] = msg["status"]
                    if msg["status"] >= 500:
                        tx.status = str(msg["status"])
                await send(msg)

            try:
                await self.app(scope, receive, _send)
            except BaseException:
                event("http.request", "error", status="1", op=op)
                raise
            finally:
                tx.data["http_status"] = status_holder["code"]
