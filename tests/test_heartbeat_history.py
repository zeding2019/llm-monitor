"""heartbeat 时序:内存热数据 + SQLite 历史合并,长窗口不受内存 buffer 限制。"""
import os
import tempfile
import time

from llm_monitor.core.models import Heartbeat
from llm_monitor.store.sqlite import SqliteStore


def _store():
    d = tempfile.mkdtemp()
    return SqliteStore(os.path.join(d, "t.db"))


def test_query_heartbeats_full_resolution_within_budget():
    store = _store()
    now = time.time_ns()
    hbs = [
        Heartbeat(ts_ns=now - (720 - i) * 5_000_000_000,
                  source="gpu:hygon:0", values={"t": float(i)})
        for i in range(720)
    ]
    store.insert_heartbeats(hbs)
    rows = store.query_heartbeats(now - 3600 * 1_000_000_000, source="gpu:hygon:0")
    assert len(rows) == 720
    assert [r[1] for r in rows] == sorted(r[1] for r in rows)
    store.close()


def test_query_heartbeats_downsamples_over_budget_but_keeps_endpoints():
    store = _store()
    now = time.time_ns()
    hbs = [
        Heartbeat(ts_ns=now - (5000 - i) * 700_000_000,
                  source="sglang.queue", values={"w": float(i)})
        for i in range(5000)
    ]
    store.insert_heartbeats(hbs)
    rows = store.query_heartbeats(now - 3600 * 1_000_000_000, source=None, max_points=1000)
    assert len(rows) <= 1002  # budget + 末点
    # 时间升序,末点是最新
    assert [r[1] for r in rows] == sorted(r[1] for r in rows)
    assert rows[-1][1] == hbs[-1].ts_ns


def test_query_heartbeats_per_source_isolation_in_downsample():
    store = _store()
    now = time.time_ns()
    gpu = [Heartbeat(ts_ns=now - (100 - i) * 1_000_000_000,
                     source="gpu:hygon:0", values={"t": float(i)}) for i in range(100)]
    busy = [Heartbeat(ts_ns=now - (9000 - i) * 300_000_000,
                      source="sglang.queue", values={"w": float(i)}) for i in range(9000)]
    store.insert_heartbeats(gpu + busy)
    rows = store.query_heartbeats(now - 3600 * 1_000_000_000, source=None, max_points=1000)
    srcs = {r[0] for r in rows}
    # 两个 source 都在;低频 GPU 不被高频 source 挤没
    assert "gpu:hygon:0" in srcs
    assert "sglang.queue" in srcs
    gpu_rows = [r for r in rows if r[0] == "gpu:hygon:0"]
    assert len(gpu_rows) >= 90  # GPU 全量(在预算内)基本都保留
    store.close()


def test_query_heartbeats_respects_cutoff():
    store = _store()
    now = time.time_ns()
    hbs = [
        Heartbeat(ts_ns=now - 7200 * 1_000_000_000, source="host", values={"cpu": 1.0}),  # 2h前
        Heartbeat(ts_ns=now - 60 * 1_000_000_000, source="host", values={"cpu": 2.0}),    # 1min前
    ]
    store.insert_heartbeats(hbs)
    rows = store.query_heartbeats(now - 3600 * 1_000_000_000, source="host")
    assert len(rows) == 1  # 只有 1min 前那条落在 1 小时窗口内
    store.close()
