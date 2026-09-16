"""对外记录 API:transaction() / event() / heartbeat() / metric()。

设计要求:热路径极简。所有序列化/聚合都在后台线程做。
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

from .context import (
    TreeContext,
    get_context,
    new_tx_id,
    reset_context,
    set_context,
)
from .models import Event, Heartbeat, Metric, RequestTrace, SchedStep, Transaction
from .registry import get_stores


@contextmanager
def transaction(
    type: str,
    name: str,
    tags: dict[str, str] | None = None,
    data: dict[str, Any] | None = None,
):
    """开启一个 Transaction。可嵌套。

    用法:
        with transaction("vllm.generate", request_id) as tx:
            tx.data["prompt_tokens"] = 128
            ...
    """
    start_ns = time.perf_counter_ns()
    wall_ns = time.time_ns()
    tx = Transaction(
        type=type,
        name=name,
        start_ns=wall_ns,
        tags=tags or {},
        data=data or {},
        tx_id=new_tx_id(),
    )

    ctx = get_context()
    token = None
    if ctx is None:
        ctx = TreeContext(root=tx)
        token = set_context(ctx)
    else:
        ctx.push(tx)

    try:
        yield tx
    except BaseException as exc:  # noqa: BLE001
        tx.status = exc.__class__.__name__
        raise
    finally:
        tx.duration_ns = time.perf_counter_ns() - start_ns
        if token is not None:
            # 顶层:整棵树落缓冲,清 context
            get_stores().transactions.append(ctx.root)
            reset_context(token)
        else:
            ctx.pop()


def event(type: str, name: str, status: str = "0", **data: Any) -> None:
    ev = Event(type=type, name=name, ts_ns=time.time_ns(), status=status, data=data)
    get_stores().events.append(ev)


def heartbeat(source: str, values: dict[str, float]) -> None:
    hb = Heartbeat(ts_ns=time.time_ns(), source=source, values=values)
    get_stores().heartbeats.append(hb)


def metric(
    name: str,
    value: float = 1.0,
    count: int = 1,
    tags: dict[str, str] | None = None,
) -> None:
    m = Metric(
        name=name,
        ts_ns=time.time_ns(),
        count=count,
        sum=value,
        tags=tags or {},
    )
    get_stores().metrics.append(m)


def emit_request_trace(rt: RequestTrace) -> None:
    """把一条完成的请求生命周期投入排空队列,由 DbWriter 落库。"""
    get_stores().request_traces.append(rt)


def emit_sched_step(step: SchedStep) -> None:
    """把一次 scheduler batch 执行投入排空队列,由 DbWriter 落库。"""
    get_stores().scheduler_steps.append(step)
