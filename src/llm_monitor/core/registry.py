"""进程内单例存储:热数据环形缓冲。"""
from __future__ import annotations

import threading

from ..config import Config
from .models import Event, Heartbeat, Metric, RequestTrace, SchedStep, Transaction
from .ringbuf import DrainQueue, RingBuffer, ShardedRingBuffer

_lock = threading.Lock()
_stores: dict[str, Stores] = {}


class Stores:
    def __init__(self, cfg: Config) -> None:
        self.transactions: RingBuffer[Transaction] = RingBuffer(cfg.transaction_buf)
        self.events: RingBuffer[Event] = RingBuffer(cfg.event_buf)
        # heartbeat 按 source 分桶:每个 source 独立保留 heartbeat_buf 条,
        # 高频 source 不会挤掉低频 source(如 GPU)的历史。
        self.heartbeats: ShardedRingBuffer[Heartbeat] = ShardedRingBuffer(cfg.heartbeat_buf)
        self.metrics: RingBuffer[Metric] = RingBuffer(cfg.transaction_buf)
        # 完成的请求生命周期,待 DbWriter 排空落库。每进程独立。
        self.request_traces: DrainQueue[RequestTrace] = DrainQueue()
        # scheduler 逐 batch 执行记录(引擎事务明细),同样由 DbWriter 落库。
        self.scheduler_steps: DrainQueue[SchedStep] = DrainQueue(maxlen=20000)


def get_stores() -> Stores:
    """懒初始化,若尚未 install 也可用(用默认 Config)。"""
    with _lock:
        if "default" not in _stores:
            _stores["default"] = Stores(Config())
        return _stores["default"]


def reset_stores_for_test() -> None:
    with _lock:
        _stores.clear()
