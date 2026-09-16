import time

from llm_monitor.core.aggregator import aggregate, minute_of
from llm_monitor.core.models import Event, Metric, Transaction


def _make_tx(type_, name, start_ns, dur_ns, children=None, status="0"):
    return Transaction(
        type=type_, name=name, start_ns=start_ns,
        duration_ns=dur_ns, status=status, children=children or [],
        tx_id="t" + name,
    )


def test_aggregate_transactions_and_percentiles():
    now = time.time_ns()
    old_min_ns = ((now // 1_000_000_000) // 60 - 2) * 60 * 1_000_000_000
    txs = [
        _make_tx("vllm.generate", "req", old_min_ns + i * 1_000, i * 1_000_000)  # duration i ms
        for i in range(1, 101)
    ]
    rows = aggregate(txs, [], [], now_ns=now)
    tx_rows = [r for r in rows if r.metric_type == "transaction"]
    assert len(tx_rows) == 1
    r = tx_rows[0]
    assert r.count == 100
    assert r.max == 100.0
    assert 40 <= r.p50 <= 60
    assert 90 <= r.p95 <= 100
    assert 90 <= r.p99 <= 100


def test_aggregate_skips_current_minute():
    now = time.time_ns()
    txs = [_make_tx("t", "n", now, 1_000_000)]
    rows = aggregate(txs, [], [], now_ns=now)
    assert rows == []


def test_aggregate_events_and_metrics():
    now = time.time_ns()
    old_ns = ((now // 1_000_000_000) // 60 - 1) * 60 * 1_000_000_000
    events = [Event(type="app", name="err", ts_ns=old_ns, status="1") for _ in range(3)]
    metrics = [Metric(name="tokens", ts_ns=old_ns, count=1, sum=5.0) for _ in range(4)]
    rows = aggregate([], metrics, events, now_ns=now)
    kinds = {r.metric_type for r in rows}
    assert kinds == {"metric", "event"}
    m = next(r for r in rows if r.metric_type == "metric")
    assert m.count == 4
    assert m.sum == 20.0


def test_minute_of_rounds_to_60s():
    ns = 123 * 1_000_000_000 + 500  # 123.0 sec
    assert minute_of(ns) == 120
