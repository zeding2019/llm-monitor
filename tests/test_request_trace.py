"""请求生命周期追踪:采集(按 rid)、跨进程 upsert 不覆盖、端点组装瀑布。"""
import os
import tempfile
import time

from llm_monitor.core.models import RequestTrace, SchedStep
from llm_monitor.core.registry import get_stores, reset_stores_for_test
from llm_monitor.patch import sglang as S
from llm_monitor.store.sqlite import SqliteStore


def _store():
    d = tempfile.mkdtemp()
    return SqliteStore(os.path.join(d, "t.db"))


class _FM:
    def __init__(self, name):
        self.name = name


class _Batch:
    def __init__(self, mode, reqs):
        self.forward_mode = _FM(mode)
        self.reqs = reqs


class _Req:
    def __init__(self, rid, prompt=8):
        self.rid = rid
        self.origin_input_ids = list(range(prompt))
        self.output_ids = []
        self._fin = False

    def finished(self):
        return self._fin


# ---------- store 层 ----------

def test_upsert_and_query_roundtrip():
    store = _store()
    rt = RequestTrace(rid="r1", arrival_ts=1000, first_token_ts=1200, finish_ts=2000,
                      ttft_ms=0.2, tpot_ms=0.1, e2e_ms=1.0, output_tokens=5, pid=111)
    store.upsert_request_traces([rt])
    rows = store.query_request_traces()
    assert len(rows) == 1
    assert rows[0][0] == "r1"
    one = store.get_request_trace("r1")
    assert one[0] == "r1" and one[8] == 5  # output_tokens 列
    store.close()


def test_multi_pid_same_db_no_clobber():
    """两个 Scheduler 进程(不同 pid)各写各的 rid,互不覆盖。"""
    store = _store()
    store.upsert_request_traces([RequestTrace(rid="a", arrival_ts=1, pid=100)])
    store.upsert_request_traces([RequestTrace(rid="b", arrival_ts=2, pid=200)])
    rids = {r[0] for r in store.query_request_traces()}
    assert rids == {"a", "b"}
    store.close()


def test_page_query_time_and_rid_filter_and_order():
    store = _store()
    # 时间轴:ts=1000/2000/3000,rid 前缀混用
    store.upsert_request_traces([
        RequestTrace(rid="req-aaaa", arrival_ts=1000),
        RequestTrace(rid="req-bbbb", arrival_ts=2000),
        RequestTrace(rid="abc-cccc", arrival_ts=3000),
    ])
    # 全量倒序
    total, rows = store.query_request_traces_page(limit=10)
    assert total == 3
    assert [r[0] for r in rows] == ["abc-cccc", "req-bbbb", "req-aaaa"]
    # 时间窗:>=1500 → 2 条
    total, rows = store.query_request_traces_page(since_ns=1500, limit=10)
    assert total == 2
    assert [r[0] for r in rows] == ["abc-cccc", "req-bbbb"]
    # rid 模糊:req → 2 条;abc → 1 条
    total, rows = store.query_request_traces_page(rid="req", limit=10)
    assert total == 2 and rows[0][0] == "req-bbbb"
    total, rows = store.query_request_traces_page(rid="abc", limit=10)
    assert total == 1 and rows[0][0] == "abc-cccc"
    # 分页:offset=1 limit=1 → 第 2 条
    total, rows = store.query_request_traces_page(offset=1, limit=1)
    assert total == 3 and rows[0][0] == "req-bbbb"
    store.close()


def test_scheduler_steps_insert_and_page():
    store = _store()
    steps = [
        SchedStep(ts_ns=1000, pid=1, mode="prefill", batch_reqs=2, batch_tokens=64, dur_ms=12.0),
        SchedStep(ts_ns=2000, pid=1, mode="decode", batch_reqs=4, batch_tokens=16, dur_ms=8.0),
        SchedStep(ts_ns=3000, pid=2, mode="decode", batch_reqs=3, batch_tokens=12, dur_ms=9.0),
    ]
    store.insert_scheduler_steps(steps)
    total, rows = store.query_scheduler_steps_page(limit=10)
    assert total == 3
    assert rows[0][0] == 3000  # 倒序
    total, rows = store.query_scheduler_steps_page(pid=1, limit=10)
    assert total == 2
    total, rows = store.query_scheduler_steps_page(since_ns=2000, limit=10)
    assert total == 2
    store.close()


def test_report_window_query():
    store = _store()
    s = 1_000_000_000
    store.upsert_request_traces([
        RequestTrace(rid="a", arrival_ts=1 * s, finish_ts=2 * s, ttft_ms=1.0, e2e_ms=1.0),
        RequestTrace(rid="b", arrival_ts=5 * s, finish_ts=6 * s, ttft_ms=2.0, e2e_ms=2.0),
        RequestTrace(rid="c", arrival_ts=20 * s, finish_ts=21 * s, ttft_ms=3.0, e2e_ms=3.0),
    ])
    # any(默认):与 [0,10s] 有交集 → a、b
    rows = store.query_request_trace_report(0, 10 * s)
    assert len(rows) == 2
    # start:开始时间落在窗口内 → a、b
    rows = store.query_request_trace_report(0, 10 * s, by="start")
    assert {r[0] for r in rows} == {"a", "b"}
    # finish:结束时间落在 [1.5s,6.5s] → a(2s)、b(6s)
    rows = store.query_request_trace_report(1.5 * s, 6.5 * s, by="finish")
    assert {r[0] for r in rows} == {"a", "b"}
    # both:开始≥3s 且结束≤7s → 只有 b(5s→6s);a 结束 2s 早于 3s
    rows = store.query_request_trace_report(3 * s, 7 * s, by="both")
    assert {r[0] for r in rows} == {"b"}
    store.close()


def test_same_rid_upsert_updates_not_duplicates():
    store = _store()
    store.upsert_request_traces([RequestTrace(rid="x", arrival_ts=1, output_tokens=1)])
    store.upsert_request_traces([RequestTrace(rid="x", arrival_ts=1, output_tokens=9)])
    rows = store.query_request_traces()
    assert len(rows) == 1
    assert rows[0][8] == 9  # 最新值
    store.close()


# ---------- 采集层 ----------

def test_lifecycle_prefill_decode_finish():
    reset_stores_for_test()
    r = _Req("req-1", prompt=8)
    S._track_requests("extend", _Batch("EXTEND", [r]), wall_ms=10.0)
    for i in range(3):
        r.output_ids.append(i)
        S._track_requests("decode", _Batch("DECODE", [r]), wall_ms=5.0)
        time.sleep(0.002)
    r._fin = True
    r.output_ids.append(99)
    S._track_requests("decode", _Batch("DECODE", [r]), wall_ms=5.0)

    traces = get_stores().request_traces.drain()
    assert len(traces) == 1
    t = traces[0]
    assert t.rid == "req-1"
    assert t.decode_steps == 4
    assert t.prompt_tokens == 8
    assert t.output_tokens == 4
    assert t.ttft_ms > 0
    assert t.e2e_ms > 0
    assert abs(t.tpot_ms - (t.decode_ms / t.decode_steps)) < 1e-6


def test_finished_via_finished_reason_attribute():
    """没有 finished() 方法,只有 finished_reason 属性的版本也能检出完成。"""
    reset_stores_for_test()

    class Req2:
        def __init__(self):
            self.rid = "req-2"
            self.input_ids = [1, 2, 3]
            self.output_ids = [9]
            self.finished_reason = None

    r = Req2()
    S._track_requests("prefill", _Batch("PREFILL", [r]), wall_ms=8.0)
    r.finished_reason = "stop"
    S._track_requests("decode", _Batch("DECODE", [r]), wall_ms=4.0)
    traces = get_stores().request_traces.drain()
    assert len(traces) == 1 and traces[0].rid == "req-2"
    assert traces[0].status == "stop"


def test_rid_missing_is_skipped():
    reset_stores_for_test()

    class NoRid:
        rid = None
    S._track_requests("decode", _Batch("DECODE", [NoRid()]), wall_ms=4.0)
    assert len(get_stores().request_traces.drain()) == 0
