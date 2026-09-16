"""HTTP 层 patch 的单测(不依赖真 vLLM)。"""
import asyncio

import pytest

from llm_monitor.core.api import transaction
from llm_monitor.core.registry import get_stores, reset_stores_for_test
from llm_monitor.patch.vllm_openai import (
    LlmMonitorMiddleware,
    _wrap_http_handler,
)


def setup_function(_):
    reset_stores_for_test()


class FakeReq:
    def __init__(self, model=None, request_id=None, client_host=None):
        self.model = model
        self.headers = {"x-request-id": request_id} if request_id else {}
        self.client = type("C", (), {"host": client_host})() if client_host else None


def test_wrap_async_handler_fills_tags_and_nests_children():
    async def handler(body, raw_request):
        # 模拟内部调用 vllm.generate,建子事务
        with transaction("vllm.generate", "req-inner"):
            await asyncio.sleep(0)
        return {"ok": True}

    wrapped = _wrap_http_handler("POST /v1/chat/completions")(handler)

    async def drive():
        return await wrapped(FakeReq(model="gpt-4", request_id="rid-abc", client_host="1.2.3.4"), None)

    result = asyncio.run(drive())
    assert result == {"ok": True}
    trees = get_stores().transactions.snapshot()
    assert len(trees) == 1
    root = trees[0]
    assert root.type == "http.request"
    assert root.name == "rid-abc"  # request_id 覆盖了 op_name
    assert root.tags["model"] == "gpt-4"
    assert root.tags["client_ip"] == "1.2.3.4"
    assert root.tags["request_id"] == "rid-abc"
    # 子事务挂上
    assert [c.type for c in root.children] == ["vllm.generate"]


def test_wrap_async_gen_records_ttfb():
    async def stream_handler(body, raw_request):
        for i in range(3):
            await asyncio.sleep(0)
            yield {"chunk": i}

    wrapped = _wrap_http_handler("POST /v1/completions")(stream_handler)

    async def drive():
        out = []
        async for c in wrapped(FakeReq(model="m"), None):
            out.append(c)
        return out

    out = asyncio.run(drive())
    assert len(out) == 3
    root = get_stores().transactions.snapshot()[0]
    assert "ttfb_ns" in root.data
    assert root.data["ttfb_ns"] > 0


def test_wrap_records_exception_and_event():
    async def handler(body, raw_request):
        raise RuntimeError("boom")

    wrapped = _wrap_http_handler("POST /v1/embeddings")(handler)

    with pytest.raises(RuntimeError):
        asyncio.run(wrapped(FakeReq(), None))

    trees = get_stores().transactions.snapshot()
    assert trees[-1].status == "RuntimeError"
    events = get_stores().events.snapshot()
    assert any(e.name == "error" and e.type == "http.request" for e in events)


@pytest.mark.asyncio
async def test_middleware_wraps_asgi_call():
    events_recorded = []

    async def app(scope, receive, send):
        assert scope["type"] == "http"
        # 内部逻辑:创建一个子事务
        with transaction("vllm.generate", "asgi-req"):
            pass
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})
        events_recorded.append("done")

    middleware = LlmMonitorMiddleware(app)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"x-request-id", b"req-xyz")],
        "client": ("10.0.0.1", 12345),
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = []

    async def send(msg):
        sent.append(msg)

    await middleware(scope, receive, send)

    assert events_recorded == ["done"]
    assert sent[0]["status"] == 200
    trees = get_stores().transactions.snapshot()
    root = trees[0]
    assert root.type == "http.request"
    assert root.name == "req-xyz"
    assert root.tags["client_ip"] == "10.0.0.1"
    assert root.data["http_status"] == 200
    assert [c.type for c in root.children] == ["vllm.generate"]


@pytest.mark.asyncio
async def test_middleware_records_5xx_as_error():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 503, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = LlmMonitorMiddleware(app)
    scope = {"type": "http", "method": "GET", "path": "/x", "headers": [], "client": None}
    await middleware(scope, lambda: None, lambda m: asyncio.sleep(0))

    root = get_stores().transactions.snapshot()[-1]
    assert root.status == "503"
    assert root.data["http_status"] == 503
