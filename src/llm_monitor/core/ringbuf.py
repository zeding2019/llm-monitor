"""线程安全环形缓冲。满则丢最旧的。"""
from __future__ import annotations

import threading
from collections import deque
from typing import Generic, Protocol, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._buf: deque[T] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def append(self, item: T) -> None:
        with self._lock:
            self._buf.append(item)

    def snapshot(self) -> list[T]:
        with self._lock:
            return list(self._buf)

    def tail(self, n: int) -> list[T]:
        with self._lock:
            if n >= len(self._buf):
                return list(self._buf)
            return list(self._buf)[-n:]

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


class _HasSource(Protocol):
    source: str
    ts_ns: int


S = TypeVar("S", bound=_HasSource)


class ShardedRingBuffer(Generic[S]):
    """按 source 分桶的环形缓冲:每个 source 独立容量,互不挤占。

    为什么需要它:heartbeat 多个 source 共用一个 buffer 时,高频 source
    (如 sglang.queue 每 batch 一条)会把低频 source(如 GPU 每 5 秒一条)
    的历史挤掉,导致时间窗口再大也只剩最近几分钟。分桶后每个 source 各自
    保留 per_source 条,时间跨度只由自己的采样频率决定。

    对外接口与 RingBuffer 一致(append/snapshot/tail),可无缝替换。
    """

    def __init__(self, per_source: int) -> None:
        if per_source <= 0:
            raise ValueError("per_source must be > 0")
        self._per_source = per_source
        self._buckets: dict[str, deque[S]] = {}
        self._lock = threading.Lock()

    def append(self, item: S) -> None:
        src = getattr(item, "source", "") or ""
        with self._lock:
            buf = self._buckets.get(src)
            if buf is None:
                buf = deque(maxlen=self._per_source)
                self._buckets[src] = buf
            buf.append(item)

    def snapshot(self) -> list[S]:
        """合并所有桶,按 ts_ns 升序返回(与单 buffer 时间序一致)。"""
        with self._lock:
            merged: list[S] = []
            for buf in self._buckets.values():
                merged.extend(buf)
        merged.sort(key=lambda h: h.ts_ns)
        return merged

    def tail(self, n: int) -> list[S]:
        """按时间序取最后 n 条(跨所有 source)。"""
        merged = self.snapshot()
        if n >= len(merged):
            return merged
        return merged[-n:]

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()

    def __len__(self) -> int:
        with self._lock:
            return sum(len(b) for b in self._buckets.values())


class DrainQueue(Generic[T]):
    """线程安全的一次性排空队列:生产者 append,消费者 drain(取走并清空)。

    用于 request trace:Scheduler 进程里的采集代码 append 完成的请求,
    后台 writer 线程周期 drain 出来批量落库。带 maxlen 上界,writer 万一
    没起来也不会无限涨(丢最旧的)。
    """

    def __init__(self, maxlen: int = 10000) -> None:
        self._buf: deque[T] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, item: T) -> None:
        with self._lock:
            self._buf.append(item)

    def drain(self) -> list[T]:
        with self._lock:
            items = list(self._buf)
            self._buf.clear()
            return items

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
