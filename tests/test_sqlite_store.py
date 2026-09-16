import json
import tempfile
from pathlib import Path

from llm_monitor.core.aggregator import MinuteRow
from llm_monitor.core.models import Event, Heartbeat, Transaction
from llm_monitor.store.sqlite import SqliteStore


def _tmp():
    return Path(tempfile.mkdtemp()) / "t.db"


def test_schema_created_and_pragma_wal():
    p = _tmp()
    s = SqliteStore(str(p))
    mode = s.query("PRAGMA journal_mode")[0][0]
    assert mode.lower() == "wal"
    s.close()


def test_insert_and_query_minute_rows():
    s = SqliteStore(str(_tmp()))
    rows = [
        MinuteRow("transaction", "vllm.generate|req", 60, count=3, sum=6.0,
                  max=3.0, p50=2.0, p95=3.0, p99=3.0),
        MinuteRow("metric", "tokens", 60, count=10, sum=100.0, max=20.0,
                  p50=10.0, p95=18.0, p99=20.0),
    ]
    s.insert_minute_rows(rows)
    got = s.query("SELECT metric_type, name, count FROM minute_agg ORDER BY name")
    assert got == [("metric", "tokens", 10), ("transaction", "vllm.generate|req", 3)]
    s.close()


def test_insert_sampled_transactions_and_events_heartbeats():
    s = SqliteStore(str(_tmp()))
    t = Transaction(type="a", name="b", start_ns=1_000, duration_ns=2_000_000_000,
                    status="0", tx_id="abcd")
    s.insert_sampled_transactions([(t, "slow")])
    s.insert_events([Event(type="ex", name="oom", ts_ns=5_000, status="1", data={"gpu": 0})])
    s.insert_heartbeats([Heartbeat(ts_ns=9_000, source="host", values={"cpu_pct": 1.5})])

    tx_rows = s.query("SELECT id, sampled_reason FROM transactions")
    assert tx_rows == [("abcd", "slow")]

    ev_rows = s.query("SELECT name, data_json FROM events")
    assert ev_rows[0][0] == "oom"
    assert json.loads(ev_rows[0][1]) == {"gpu": 0}

    hb_rows = s.query("SELECT source, values_json FROM heartbeat")
    assert hb_rows[0][0] == "host"
    s.close()
