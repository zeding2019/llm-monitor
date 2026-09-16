"""ShardedRingBuffer:按 source 分桶,高频 source 不挤掉低频 source。"""
from dataclasses import dataclass

import pytest

from llm_monitor.core.ringbuf import ShardedRingBuffer


@dataclass
class _HB:
    source: str
    ts_ns: int


def test_high_freq_source_does_not_evict_low_freq():
    buf = ShardedRingBuffer(per_source=3)
    for i in range(100):
        buf.append(_HB("busy", i))
    buf.append(_HB("gpu", 1000))
    buf.append(_HB("gpu", 1001))

    snap = buf.snapshot()
    busy = [h.ts_ns for h in snap if h.source == "busy"]
    gpu = [h.ts_ns for h in snap if h.source == "gpu"]
    # busy 只保留自己最后 3 条
    assert busy == [97, 98, 99]
    # gpu 完整保留,不受 busy 洪泛影响
    assert gpu == [1000, 1001]


def test_snapshot_is_globally_time_ordered():
    buf = ShardedRingBuffer(per_source=10)
    buf.append(_HB("a", 3))
    buf.append(_HB("b", 1))
    buf.append(_HB("a", 5))
    buf.append(_HB("b", 2))
    ts = [h.ts_ns for h in buf.snapshot()]
    assert ts == [1, 2, 3, 5]


def test_tail_crosses_sources_in_time_order():
    buf = ShardedRingBuffer(per_source=10)
    buf.append(_HB("a", 10))
    buf.append(_HB("b", 20))
    buf.append(_HB("a", 30))
    assert [h.ts_ns for h in buf.tail(2)] == [20, 30]


def test_len_sums_all_buckets():
    buf = ShardedRingBuffer(per_source=2)
    buf.append(_HB("a", 1))
    buf.append(_HB("a", 2))
    buf.append(_HB("a", 3))  # 挤掉 ts=1
    buf.append(_HB("b", 4))
    assert len(buf) == 3  # a:2 + b:1


def test_empty_source_string_is_its_own_bucket():
    buf = ShardedRingBuffer(per_source=5)
    buf.append(_HB("", 1))
    buf.append(_HB("x", 2))
    assert len(buf) == 2


def test_zero_capacity_rejected():
    with pytest.raises(ValueError):
        ShardedRingBuffer(per_source=0)
