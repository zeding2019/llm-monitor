"""四种指标模型的数据类。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Transaction:
    """一段可嵌套的耗时执行,构成消息树。"""
    type: str
    name: str
    start_ns: int
    duration_ns: int = 0
    status: str = "0"  # "0" == OK
    tags: dict[str, str] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    children: list[Transaction] = field(default_factory=list)
    tx_id: str = ""
    parent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "start_ns": self.start_ns,
            "duration_ns": self.duration_ns,
            "status": self.status,
            "tags": self.tags,
            "data": self.data,
            "tx_id": self.tx_id,
            "parent_id": self.parent_id,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class Event:
    """一次性事件,只计数。"""
    type: str
    name: str
    ts_ns: int
    status: str = "0"
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Heartbeat:
    """定期采样值。"""
    ts_ns: int
    source: str  # "host" / "gpu:0" / ...
    values: dict[str, float] = field(default_factory=dict)


@dataclass
class Metric:
    """业务指标。多次上报会在聚合层合并。"""
    name: str
    ts_ns: int
    count: int = 1
    sum: float = 0.0
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class RequestTrace:
    """单请求生命周期(按 SGLang rid 聚合)。在 Scheduler 进程里累计,完成时落库。

    时间戳都是 wall clock ns。ttft/tpot/e2e 在完成时算好,前端直接展示。
    """
    rid: str
    arrival_ts: int
    first_token_ts: int | None = None
    finish_ts: int | None = None
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    decode_steps: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    e2e_ms: float = 0.0
    status: str = "0"
    pid: int = 0
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class SchedStep:
    """Scheduler 一次 batch 执行(逐条落库,供「引擎事务」明细表)。"""
    ts_ns: int
    pid: int
    mode: str  # prefill / extend / decode / ...
    batch_reqs: int = 0
    batch_tokens: int = 0
    dur_ms: float = 0.0
