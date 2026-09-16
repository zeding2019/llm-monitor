"""SQLite 存储层。

写入模型:每进程一个 DbWriter 线程独占**写连接**,采集侧只把数据丢进内存
缓冲/队列(非阻塞),由 DbWriter 批量、周期性落盘 —— 这是 SQLite 下正确的
"异步写入"形态(SQLite 文件级单写,再多写连接也只会互相 BUSY 争锁)。

读取模型:一组**只读连接池**,web 的查询端点并发借用,和写线程互不阻塞
(WAL 允许多读 + 单写)。跨进程写竞争用 busy_timeout 让其排队重试而非报错。
"""
from __future__ import annotations

import contextlib
import json
import logging
import queue
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..core.aggregator import MinuteRow, aggregate
from ..core.registry import get_stores

log = logging.getLogger("llm_monitor.store.sqlite")

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_BUSY_TIMEOUT_MS = 5000


class SqliteStore:
    """一个写连接(单写线程独占) + 一个只读连接池(并发读)。"""

    def __init__(self, path: str, read_pool_size: int = 4) -> None:
        self.path = path
        # 写连接:仅 DbWriter 线程使用。check_same_thread=False 是因为
        # 建连接的线程和实际写的线程可能不同(bootstrap 建、writer 用)。
        self._write_conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._write_lock = threading.Lock()
        # busy_timeout 必须在任何建表/写之前设好:多个 SGLang 进程同时启动时
        # 都在跑 CREATE TABLE IF NOT EXISTS,没有 timeout 会立刻 SQLITE_BUSY。
        self._tune(self._write_conn)
        self._init_schema()

        # 只读连接池(惰性创建):必须在 WAL 打开后建,读端才能与写并发。
        # 惰性是为了子进程(只写 trace 不读)不白开一堆空闲连接。
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue()
        self._pool_lock = threading.Lock()
        self._pool_size = max(1, read_pool_size)
        self._pool_created = 0

    def _init_schema(self) -> None:
        with self._write_lock, self._write_conn:
            self._write_conn.executescript(_SCHEMA_PATH.read_text())

    def _tune(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        # 跨进程/跨线程写竞争:等锁而非立刻 SQLITE_BUSY 报错
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")

    def _new_read_conn(self) -> sqlite3.Connection:
        rc = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._tune(rc)
        rc.execute("PRAGMA query_only=ON")
        return rc

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        conn = None
        try:
            conn = self._pool.get_nowait()
        except queue.Empty:
            with self._pool_lock:
                if self._pool_created < self._pool_size:
                    self._pool_created += 1
                    conn = self._new_read_conn()
            if conn is None:
                conn = self._pool.get()  # 池满且都在用,等一条归还
        try:
            yield conn
        finally:
            self._pool.put(conn)

    # ---------------- 写入(仅 DbWriter 线程调用)----------------

    def insert_minute_rows(self, rows: list[MinuteRow]) -> None:
        if not rows:
            return
        payload = [
            (r.metric_type, r.name, r.minute_ts, r.count, r.sum, r.max, r.p50, r.p95, r.p99, r.tags_json)
            for r in rows
        ]
        with self._write_lock, self._write_conn:
            self._write_conn.executemany(
                "INSERT OR REPLACE INTO minute_agg "
                "(metric_type, name, minute_ts, count, sum, max, p50, p95, p99, tags_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                payload,
            )

    def insert_sampled_transactions(self, txs) -> None:
        if not txs:
            return
        payload = []
        for t, reason in txs:
            payload.append((
                t.tx_id, t.type, t.name, t.start_ns, t.duration_ns, t.status,
                json.dumps(t.to_dict()), reason,
            ))
        with self._write_lock, self._write_conn:
            self._write_conn.executemany(
                "INSERT OR REPLACE INTO transactions "
                "(id, type, name, start_ts, duration_ns, status, tree_json, sampled_reason) "
                "VALUES (?,?,?,?,?,?,?,?)",
                payload,
            )

    def insert_events(self, events) -> None:
        if not events:
            return
        payload = [(e.type, e.name, e.ts_ns, e.status, json.dumps(e.data)) for e in events]
        with self._write_lock, self._write_conn:
            self._write_conn.executemany(
                "INSERT INTO events (type, name, ts, status, data_json) VALUES (?,?,?,?,?)",
                payload,
            )

    def insert_heartbeats(self, hbs) -> None:
        if not hbs:
            return
        payload = [(h.source, h.ts_ns, json.dumps(h.values)) for h in hbs]
        with self._write_lock, self._write_conn:
            self._write_conn.executemany(
                "INSERT INTO heartbeat (source, ts, values_json) VALUES (?,?,?)",
                payload,
            )

    def upsert_request_traces(self, traces) -> None:
        """按 rid upsert 请求生命周期。多进程写同一库:rid 是 PK,各写各的,不覆盖。"""
        if not traces:
            return
        payload = [
            (
                t.rid, t.arrival_ts, t.first_token_ts, t.finish_ts,
                t.prefill_ms, t.decode_ms, t.decode_steps,
                t.prompt_tokens, t.output_tokens,
                t.ttft_ms, t.tpot_ms, t.e2e_ms, t.status, t.pid,
                json.dumps(t.tags),
            )
            for t in traces
        ]
        with self._write_lock, self._write_conn:
            self._write_conn.executemany(
                "INSERT OR REPLACE INTO request_trace "
                "(rid, arrival_ts, first_token_ts, finish_ts, prefill_ms, decode_ms, "
                " decode_steps, prompt_tokens, output_tokens, ttft_ms, tpot_ms, e2e_ms, "
                " status, pid, tags_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                payload,
            )

    def cleanup(self, retention_days: int) -> None:
        cutoff_ns = time.time_ns() - retention_days * 86400 * 1_000_000_000
        cutoff_min = cutoff_ns // 1_000_000_000
        with self._write_lock, self._write_conn:
            self._write_conn.execute("DELETE FROM transactions WHERE start_ts < ?", (cutoff_ns,))
            self._write_conn.execute("DELETE FROM events WHERE ts < ?", (cutoff_ns,))
            self._write_conn.execute("DELETE FROM heartbeat WHERE ts < ?", (cutoff_ns,))
            self._write_conn.execute("DELETE FROM minute_agg WHERE minute_ts < ?", (cutoff_min,))
            self._write_conn.execute("DELETE FROM request_trace WHERE arrival_ts < ?", (cutoff_ns,))
            self._write_conn.execute("DELETE FROM scheduler_step WHERE ts < ?", (cutoff_ns,))

    # ---------------- 读取(web 端点,走只读连接池)----------------

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._read() as conn:
            cur = conn.execute(sql, params)
            return cur.fetchall()

    def query_request_traces(self, since_ns: int = 0, limit: int = 100) -> list[tuple]:
        """最近的请求生命周期,按到达时间倒序。"""
        sql = (
            "SELECT rid, arrival_ts, first_token_ts, finish_ts, prefill_ms, decode_ms, "
            "decode_steps, prompt_tokens, output_tokens, ttft_ms, tpot_ms, e2e_ms, "
            "status, pid, tags_json FROM request_trace"
        )
        params: list = []
        if since_ns > 0:
            sql += " WHERE arrival_ts >= ?"
            params.append(since_ns)
        sql += " ORDER BY arrival_ts DESC LIMIT ?"
        params.append(limit)
        return self.query(sql, tuple(params))

    def get_request_trace(self, rid: str) -> tuple | None:
        rows = self.query(
            "SELECT rid, arrival_ts, first_token_ts, finish_ts, prefill_ms, decode_ms, "
            "decode_steps, prompt_tokens, output_tokens, ttft_ms, tpot_ms, e2e_ms, "
            "status, pid, tags_json FROM request_trace WHERE rid = ?",
            (rid,),
        )
        return rows[0] if rows else None

    def query_request_traces_page(
        self,
        since_ns: int = 0,
        rid: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[int, list[tuple]]:
        """分页查请求生命周期:(total, rows)。可按时间窗(since_ns)与 rid 过滤,
        按到达时间倒序。rid 支持模糊匹配(前缀/子串)。"""
        where = []
        params: list = []
        if since_ns > 0:
            where.append("arrival_ts >= ?")
            params.append(since_ns)
        if rid:
            where.append("rid LIKE ?")
            params.append(f"%{rid}%")
        cond = (" WHERE " + " AND ".join(where)) if where else ""
        total = self.query(
            f"SELECT COUNT(*) FROM request_trace{cond}", tuple(params)
        )[0][0]
        cols = (
            "rid, arrival_ts, first_token_ts, finish_ts, prefill_ms, decode_ms, "
            "decode_steps, prompt_tokens, output_tokens, ttft_ms, tpot_ms, e2e_ms, "
            "status, pid, tags_json"
        )
        rows = self.query(
            f"SELECT {cols} FROM request_trace{cond} "
            "ORDER BY arrival_ts DESC LIMIT ? OFFSET ?",
            tuple(params + [limit, offset]),
        )
        return total, rows

    def insert_scheduler_steps(self, steps) -> None:
        """追加 scheduler 逐 batch 执行记录(仅写,多进程安全)。"""
        if not steps:
            return
        payload = [(s.ts_ns, s.pid, s.mode, s.batch_reqs, s.batch_tokens, s.dur_ms)
                   for s in steps]
        with self._write_lock, self._write_conn:
            self._write_conn.executemany(
                "INSERT INTO scheduler_step (ts, pid, mode, batch_reqs, batch_tokens, dur_ms) "
                "VALUES (?,?,?,?,?,?)",
                payload,
            )

    def query_scheduler_steps_page(
        self,
        since_ns: int = 0,
        pid: int | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[int, list[tuple]]:
        where = []
        params: list = []
        if since_ns > 0:
            where.append("ts >= ?")
            params.append(since_ns)
        if pid:
            where.append("pid = ?")
            params.append(pid)
        cond = (" WHERE " + " AND ".join(where)) if where else ""
        total = self.query(
            f"SELECT COUNT(*) FROM scheduler_step{cond}", tuple(params)
        )[0][0]
        rows = self.query(
            f"SELECT ts, pid, mode, batch_reqs, batch_tokens, dur_ms "
            f"FROM scheduler_step{cond} ORDER BY ts DESC LIMIT ? OFFSET ?",
            tuple(params + [limit, offset]),
        )
        return total, rows

    def query_request_trace_report(
        self, start_ns: int, end_ns: int, by: str = "any"
    ) -> list[tuple]:
        """取时间窗内的请求原始行,供报告聚合。by 过滤口径:
          - any   窗口与请求生命周期有交集(默认,向后兼容)
          - start 按 rid 开始(arrival)落在窗口内
          - finish 按 rid 结束(finish)落在窗口内
          - both  开始与结束都落在窗口内
        """
        cols = ("rid, arrival_ts, first_token_ts, finish_ts, ttft_ms, tpot_ms, "
                "e2e_ms, status, prompt_tokens, output_tokens, decode_steps, pid")
        base = "SELECT " + cols + " FROM request_trace WHERE "
        if by == "start":
            where, params = "arrival_ts >= ? AND arrival_ts <= ?", [start_ns, end_ns]
        elif by == "finish":
            where, params = "finish_ts >= ? AND finish_ts <= ?", [start_ns, end_ns]
        elif by == "both":
            where, params = "arrival_ts >= ? AND finish_ts <= ?", [start_ns, end_ns]
        else:  # any
            where, params = "finish_ts >= ? AND arrival_ts <= ?", [start_ns, end_ns]
        return self.query(base + where, tuple(params))

    def distinct_heartbeat_sources(self) -> list[str]:
        """DB 里出现过的所有 heartbeat source —— 跨进程来源(如子进程产生的
        sglang.kv_cache/queue/prefill/decode)即使不在当前进程内存也能被发现。"""
        rows = self.query("SELECT DISTINCT source FROM heartbeat")
        return [r[0] for r in rows]

    def query_heartbeats(
        self,
        cutoff_ns: int,
        source: str | None = None,
        max_points: int = 20000,
    ) -> list[tuple[str, int, str]]:
        """取 [cutoff_ns, now] 窗口内的原始 heartbeat,供时序图读历史。

        返回 (source, ts_ns, values_json) 列表,按 ts 升序。
        超过 max_points 时按等距抽样降采样(每个 source 内保序),避免长窗口爆量。
        """
        sql = "SELECT source, ts, values_json FROM heartbeat WHERE ts >= ?"
        params: list = [cutoff_ns]
        if source:
            sql += " AND source = ?"
            params.append(source)
        sql += " ORDER BY ts ASC"
        rows = self.query(sql, tuple(params))
        if len(rows) <= max_points:
            return rows
        # 降采样:按 source 分组后等距抽样,保证每条曲线形状不失真
        by_src: dict[str, list[tuple[str, int, str]]] = {}
        for r in rows:
            by_src.setdefault(r[0], []).append(r)
        budget_per_src = max(2, max_points // max(len(by_src), 1))
        out: list[tuple[str, int, str]] = []
        for group in by_src.values():
            if len(group) <= budget_per_src:
                out.extend(group)
                continue
            step = len(group) / budget_per_src
            picked = [group[min(int(i * step), len(group) - 1)] for i in range(budget_per_src)]
            if picked[-1][1] != group[-1][1]:
                picked.append(group[-1])  # 末点保留,曲线右端不丢最新
            out.extend(picked)
        out.sort(key=lambda r: r[1])
        return out

    def close(self) -> None:
        with self._write_lock:
            with contextlib.suppress(Exception):
                self._write_conn.close()
            while not self._pool.empty():
                with contextlib.suppress(Exception):
                    self._pool.get_nowait().close()


class DbWriter(threading.Thread):
    """统一异步写入线程:每进程一个,独占写连接。

    - 快路径(每 trace_interval 秒):排空 request_traces 队列 → upsert。所有进程都跑。
    - 慢路径(每 agg_interval 秒,仅 aggregate=True 的单例进程):tx/metric/event 聚合成
      minute_agg、采样慢/错事务、heartbeat/event 增量落盘;每小时 cleanup 一次。

    为什么慢路径只在单例:minute_agg 用 INSERT OR REPLACE,多进程同时写会互相覆盖;
    而 request_trace 按 rid upsert,多进程天然安全,故快路径在所有进程都跑。
    """

    def __init__(
        self,
        store: SqliteStore,
        aggregate: bool = True,
        trace_interval_s: float = 2.0,
        agg_interval_s: float = 60.0,
        retention_days: int = 7,
        slow_threshold_ns: int = 1_000_000_000,
    ) -> None:
        super().__init__(name="llm-monitor-db-writer", daemon=True)
        self.store = store
        self.aggregate = aggregate
        self.trace_interval = trace_interval_s
        self.agg_interval = agg_interval_s
        self.retention_days = retention_days
        self.slow_threshold_ns = slow_threshold_ns
        self._stop = threading.Event()
        self._last_agg = 0.0
        self._last_cleanup = 0.0
        self._last_hb_ns = 0
        self._last_ev_id = 0

    def run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.trace_interval)
            try:
                self._flush_traces()
            except Exception as e:  # noqa: BLE001
                log.warning("trace flush failed: %s", e)
            if not self.aggregate:
                continue
            now = time.monotonic()
            if now - self._last_agg >= self.agg_interval:
                self._last_agg = now
                try:
                    self._flush_aggregates()
                except Exception as e:  # noqa: BLE001
                    log.warning("aggregate flush failed: %s", e)
            if now - self._last_cleanup >= 3600:
                self._last_cleanup = now
                try:
                    self.store.cleanup(self.retention_days)
                except Exception as e:  # noqa: BLE001
                    log.warning("cleanup failed: %s", e)

    def _flush_traces(self) -> None:
        stores = get_stores()
        traces = stores.request_traces.drain()
        if traces:
            self.store.upsert_request_traces(traces)
        steps = stores.scheduler_steps.drain()
        if steps:
            self.store.insert_scheduler_steps(steps)
        # heartbeat 也由每进程落库:KV/queue/prefill/decode 等心跳产生在 Scheduler
        # 子进程,若只在单例进程刷则概览永远看不到(进程隔离)。2s 一刷 + 按 ts 去重。
        hbs = stores.heartbeats.snapshot()
        new_hbs = [h for h in hbs if h.ts_ns > self._last_hb_ns]
        if new_hbs:
            self.store.insert_heartbeats(new_hbs)
            self._last_hb_ns = new_hbs[-1].ts_ns

    def _flush_aggregates(self) -> None:
        stores = get_stores()
        txs = stores.transactions.snapshot()
        metrics = stores.metrics.snapshot()
        events = stores.events.snapshot()

        self.store.insert_minute_rows(aggregate(txs, metrics, events))

        # 采样保留:慢事务、异常事务
        sampled = []
        for t in txs:
            reason = None
            if t.status != "0":
                reason = "error"
            elif t.duration_ns >= self.slow_threshold_ns:
                reason = "slow"
            if reason:
                sampled.append((t, reason))
        self.store.insert_sampled_transactions(sampled)

        # event 增量落盘(用 ts 去重)。heartbeat 已在 _flush_traces 统一刷。
        new_evs = [e for e in events if e.ts_ns > self._last_ev_id]
        if new_evs:
            self.store.insert_events(new_evs)
            self._last_ev_id = new_evs[-1].ts_ns

    def stop(self) -> None:
        self._stop.set()
        with contextlib.suppress(Exception):
            self._flush_traces()
        if self.aggregate:
            with contextlib.suppress(Exception):
                self._flush_aggregates()
