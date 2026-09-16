"""用假 module 覆盖 patch/registry 的核心行为。"""
import sys
import types

import pytest

from llm_monitor.core.api import transaction
from llm_monitor.core.registry import get_stores, reset_stores_for_test
from llm_monitor.patch.registry import _hooks, when_imported
from llm_monitor.patch.util import unwrap_method, wrap_method


def setup_function(_):
    reset_stores_for_test()


def test_wrap_and_unwrap_method():
    class Foo:
        def bar(self, x):
            return x + 1

    def wrapper(orig):
        def inner(self, x):
            return orig(self, x) * 10
        return inner

    wrap_method(Foo, "bar", wrapper)
    assert Foo().bar(2) == 30

    # 幂等:再包一次,底层 orig 不会变
    wrap_method(Foo, "bar", wrapper)
    assert Foo().bar(2) == 30

    unwrap_method(Foo, "bar")
    assert Foo().bar(2) == 3


def test_when_imported_triggers_on_existing_module():
    mod = types.ModuleType("fake_vllm_alpha")
    sys.modules["fake_vllm_alpha"] = mod
    called = []

    @when_imported("fake_vllm_alpha")
    def _hook():
        called.append(1)

    assert called == [1]  # 已在 sys.modules → 立刻跑

    # 清理
    del sys.modules["fake_vllm_alpha"]
    _hooks.pop("fake_vllm_alpha", None)


def test_when_imported_triggers_on_future_import():
    called = []

    @when_imported("fake_vllm_beta")
    def _hook():
        called.append("later")

    assert called == []
    mod = types.ModuleType("fake_vllm_beta")
    sys.modules["fake_vllm_beta"] = mod
    import fake_vllm_beta  # noqa: F401 走 import 钩子
    assert called == ["later"]

    del sys.modules["fake_vllm_beta"]
    _hooks.pop("fake_vllm_beta", None)


def test_wrapped_generate_produces_tree():
    """构造一个假的 async generator engine,用 _wrap_generate 包装,验证消息树写入。"""
    import asyncio

    from llm_monitor.patch.vllm_engine import _wrap_generate

    class FakeEngine:
        async def generate(self, prompt, sampling_params=None, request_id="rid"):
            for _ in range(3):
                await asyncio.sleep(0)
                yield {"token": "x"}

    wrap_method(FakeEngine, "generate", _wrap_generate)

    async def drive():
        collected = []
        async for out in FakeEngine().generate("hi", None, request_id="req-1"):
            # 模拟请求内部再嵌一层 sub-transaction
            with transaction("app.postprocess", "pp"):
                collected.append(out)
        return collected

    assert asyncio.run(drive()) == [{"token": "x"}] * 3
    stored = get_stores().transactions.snapshot()
    assert len(stored) == 1
    root = stored[0]
    assert root.type == "vllm.generate"
    assert root.name == "req-1"
    # 3 次 pp
    assert [c.type for c in root.children] == ["app.postprocess"] * 3
    assert "ttft_ns" in root.data
    assert root.data["output_events"] == 3


@pytest.mark.asyncio
async def test_wrapped_generate_records_error_event():
    from llm_monitor.patch.vllm_engine import _wrap_generate

    class FakeEngine:
        async def generate(self, request_id="rid"):
            yield 1
            raise RuntimeError("boom")

    wrap_method(FakeEngine, "generate", _wrap_generate)

    with pytest.raises(RuntimeError):
        async for _ in FakeEngine().generate(request_id="req-x"):
            pass

    events = get_stores().events.snapshot()
    assert any(e.name == "error" for e in events)
    stored = get_stores().transactions.snapshot()
    assert stored[-1].status == "RuntimeError"
