"""Web 服务:后台线程运行 uvicorn,提供 REST + 静态页 + Prometheus。"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse

from ..core.aggregator import aggregate
from ..core.registry import get_stores
from . import prometheus

log = logging.getLogger("llm_monitor.web")

_STATIC_DIR = Path(__file__).with_name("static")


def _build_app(sqlite_store=None) -> FastAPI:
    app = FastAPI(title="llm-monitor", version="0.1.0", docs_url="/docs")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/heartbeat/tail")
    def hb_tail(
        n: int = 120,
        source: str | None = None,
        since_sec: int = 0,
    ) -> dict:
        """按数量或时间窗口取心跳。
        since_sec > 0 时以时间窗口为准,忽略 n;否则按 n 取尾部。

        时间窗口模式下会合并内存热数据 + SQLite 历史:内存 buffer 只留最近若干条,
        长窗口(超过 buffer 覆盖范围)的历史从 SQLite 补齐,dedup by (source, ts)。
        """
        import time as _t
        if since_sec > 0:
            cutoff = _t.time_ns() - since_sec * 1_000_000_000
            all_items = get_stores().heartbeats.snapshot()
            mem = [h for h in all_items if h.ts_ns >= cutoff]
            if source:
                mem = [h for h in mem if h.source == source]
            merged = _merge_hb_with_sqlite(mem, cutoff, source)
            return {"items": merged}
        else:
            items = get_stores().heartbeats.tail(n * (5 if source else 1))
            if source:
                items = [h for h in items if h.source == source][-n:]
        return {"items": [{"ts_ns": h.ts_ns, "source": h.source, "values": h.values} for h in items]}

    def _merge_hb_with_sqlite(mem, cutoff_ns, source):
        """内存 heartbeat + SQLite 历史合并,按 (source, ts) 去重,ts 升序。"""
        seen: set[tuple[str, int]] = set()
        out: list[dict] = []
        if sqlite_store is not None:
            try:
                for src, ts, vjson in sqlite_store.query_heartbeats(cutoff_ns, source):
                    key = (src, ts)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({"ts_ns": ts, "source": src, "values": json.loads(vjson)})
            except Exception as e:  # noqa: BLE001
                log.warning("query_heartbeats failed, memory-only: %s", e)
        # 内存数据覆盖/补最新(60s flush 间隔内的点只在内存里)
        for h in mem:
            key = (h.source, h.ts_ns)
            if key in seen:
                continue
            seen.add(key)
            out.append({"ts_ns": h.ts_ns, "source": h.source, "values": h.values})
        out.sort(key=lambda d: d["ts_ns"])
        return out

    @app.get("/api/heartbeat/sources")
    def hb_sources() -> dict:
        srcs = {h.source for h in get_stores().heartbeats.snapshot()}
        # 并集 DB 里出现过的 source:子进程(如 scheduler)产生的心跳也已被 DbWriter
        # 落库,概览据此能看到 KV/queue/prefill/decode 等跨进程指标。
        if sqlite_store is not None:
            try:
                srcs.update(sqlite_store.distinct_heartbeat_sources())
            except Exception as e:  # noqa: BLE001
                log.warning("distinct sources failed: %s", e)
        return {"sources": sorted(srcs)}

    @app.get("/api/heartbeat/latest")
    def hb_latest(source: str) -> dict:
        """某 source 的最新一条心跳(先内存,后 SQLite),用于展示当前值。"""
        srcs = get_stores().heartbeats.snapshot()
        for h in reversed(srcs):
            if h.source == source:
                return {"ts_ns": h.ts_ns, "source": h.source, "values": h.values}
        if sqlite_store is not None:
            rows = sqlite_store.query(
                "SELECT source, ts, values_json FROM heartbeat "
                "WHERE source = ? ORDER BY ts DESC LIMIT 1", (source,))
            if rows:
                return {"ts_ns": rows[0][1], "source": rows[0][0],
                        "values": json.loads(rows[0][2])}
            # KV 类 source 的 pool 采样行不含 hit_ratio;命中率行可能稍旧。
            # 回退:在最近 N 条里优先挑带 hit_ratio 的那条,保证页面显示命中率。
            if "kv_cache" in source:
                recent = sqlite_store.query(
                    "SELECT ts, values_json FROM heartbeat "
                    "WHERE source = ? ORDER BY ts DESC LIMIT 300", (source,))
                for ts, vj in recent:
                    if '"hit_ratio"' in vj:
                        v = json.loads(vj)
                        if v.get("hit_ratio"):
                            return {"ts_ns": ts, "source": source, "values": v}
        return {"ts_ns": 0, "source": source, "values": {}}

    @app.get("/api/transactions/tail")
    def tx_tail(n: int = 50, since_sec: int = 0) -> dict:
        if since_sec > 0:
            import time as _t
            cutoff = _t.time_ns() - since_sec * 1_000_000_000
            all_items = get_stores().transactions.snapshot()
            items = [t for t in all_items if t.start_ns >= cutoff]
        else:
            items = get_stores().transactions.tail(n)
        return {"items": [_tx_summary(t) for t in items]}

    @app.get("/api/transactions/{tx_id}")
    def tx_detail(tx_id: str) -> dict:
        for t in reversed(get_stores().transactions.snapshot()):
            if t.tx_id == tx_id:
                return t.to_dict()
        # 尝试 SQLite
        if sqlite_store is not None:
            rows = sqlite_store.query(
                "SELECT tree_json FROM transactions WHERE id=?", (tx_id,)
            )
            if rows:
                return json.loads(rows[0][0])
        raise HTTPException(404, f"tx {tx_id} not found")

    @app.get("/api/events/tail")
    def ev_tail(n: int = 100) -> dict:
        items = get_stores().events.tail(n)
        return {
            "items": [
                {"type": e.type, "name": e.name, "ts_ns": e.ts_ns, "status": e.status, "data": e.data}
                for e in items
            ]
        }

    @app.get("/api/metrics/tail")
    def me_tail(n: int = 100) -> dict:
        items = get_stores().metrics.tail(n)
        return {
            "items": [
                {"name": m.name, "ts_ns": m.ts_ns, "count": m.count, "sum": m.sum, "tags": m.tags}
                for m in items
            ]
        }

    _TRACE_COLS = (
        "rid", "arrival_ts", "first_token_ts", "finish_ts", "prefill_ms", "decode_ms",
        "decode_steps", "prompt_tokens", "output_tokens", "ttft_ms", "tpot_ms", "e2e_ms",
        "status", "pid", "tags_json",
    )

    def _trace_row_to_dict(row) -> dict:
        d = dict(zip(_TRACE_COLS, row, strict=False))
        try:
            d["tags"] = json.loads(d.pop("tags_json") or "{}")
        except Exception:  # noqa: BLE001
            d["tags"] = {}
        return d

    @app.get("/api/traces")
    def traces(
        since_sec: int = 0,
        rid: str | None = None,
        offset: int = 0,
        limit: int = Query(20, ge=1, le=500),
    ) -> dict:
        """请求生命周期列表:分页 + 时间窗(since_sec)+ rid 模糊过滤。
        跨进程数据来自共享 SQLite 的 request_trace 表。"""
        if sqlite_store is None:
            return {"items": [], "total": 0, "offset": 0, "limit": limit}
        import time as _t
        since_ns = _t.time_ns() - since_sec * 1_000_000_000 if since_sec > 0 else 0
        total, rows = sqlite_store.query_request_traces_page(
            since_ns, rid=rid or None, offset=max(offset, 0), limit=limit
        )
        return {
            "items": [_trace_row_to_dict(r) for r in rows],
            "total": total,
            "offset": max(offset, 0),
            "limit": limit,
        }

    @app.get("/api/trace/{rid}")
    def trace_detail(rid: str) -> dict:
        """单请求生命周期 → 瀑布节点树(到达→prefill/TTFT→decode→完成),复用前端甘特。"""
        if sqlite_store is None:
            raise HTTPException(404, "trace store disabled")
        row = sqlite_store.get_request_trace(rid)
        if row is None:
            raise HTTPException(404, f"trace {rid} not found")
        d = _trace_row_to_dict(row)
        arrival = d["arrival_ts"]
        first_tok = d["first_token_ts"]
        finish = d["finish_ts"] or arrival
        e2e_ns = max(finish - arrival, 1)
        children = []
        # prefill / 排队 阶段:到达 → 首 token(即 TTFT 窗口)
        pf_end = first_tok or finish
        children.append({
            "type": "prefill", "name": "prefill + 排队 (TTFT)",
            "start_ns": arrival, "duration_ns": max(pf_end - arrival, 0),
            "status": "0", "tags": {}, "data": {"prefill_ms": d["prefill_ms"], "ttft_ms": d["ttft_ms"]},
            "tx_id": rid + ":prefill", "children": [],
        })
        # decode 阶段:首 token → 完成
        if first_tok and finish > first_tok:
            children.append({
                "type": "decode", "name": f"decode ×{d['decode_steps']}",
                "start_ns": first_tok, "duration_ns": finish - first_tok,
                "status": "0", "tags": {},
                "data": {"decode_ms": d["decode_ms"], "tpot_ms": d["tpot_ms"],
                         "output_tokens": d["output_tokens"]},
                "tx_id": rid + ":decode", "children": [],
            })
        return {
            "type": "request", "name": rid,
            "start_ns": arrival, "duration_ns": e2e_ns,
            "status": d["status"],
            "tags": {**d["tags"], "pid": str(d["pid"])},
            "data": {
                "ttft_ms": d["ttft_ms"], "tpot_ms": d["tpot_ms"], "e2e_ms": d["e2e_ms"],
                "prompt_tokens": d["prompt_tokens"], "output_tokens": d["output_tokens"],
                "decode_steps": d["decode_steps"],
            },
            "tx_id": rid,
            "children": children,
        }

    @app.get("/api/scheduler/steps")
    def scheduler_steps(
        since_sec: int = 0,
        pid: int | None = None,
        offset: int = 0,
        limit: int = Query(200, ge=1, le=2000),
    ) -> dict:
        """引擎事务:逐 batch 执行明细(分页 + 时间窗 + 按 pid 过滤)。"""
        if sqlite_store is None:
            return {"items": [], "total": 0}
        import time as _t
        since_ns = _t.time_ns() - since_sec * 1_000_000_000 if since_sec > 0 else 0
        total, rows = sqlite_store.query_scheduler_steps_page(
            since_ns, pid=pid, offset=max(offset, 0), limit=limit
        )
        return {
            "items": [
                {"ts_ns": r[0], "pid": r[1], "mode": r[2],
                 "batch_reqs": r[3], "batch_tokens": r[4], "dur_ms": r[5]}
                for r in rows
            ],
            "total": total,
        }

    @app.get("/api/report")
    def report(
        start_sec: int | None = None,
        end_sec: int | None = None,
        by: str = "any",
    ) -> dict:
        """时间段报告:按 [start_sec,end_sec](unix 秒,留空=默认最近1小时)
        聚合 request_trace,输出汇总统计供前端渲染。

        by 过滤口径:any=与窗口有交集 / start=按 rid 开始时间 /
        finish=按 rid 结束时间 / both=开始与结束都在窗口内。"""
        if sqlite_store is None:
            raise HTTPException(404, "trace store disabled")
        if by not in ("any", "start", "finish", "both"):
            by = "any"
        import time as _t
        now = _t.time()
        start = start_sec if start_sec else now - 3600
        end = end_sec if end_sec else now
        start_ns, end_ns = int(start * 1e9), int(end * 1e9)
        rows = sqlite_store.query_request_trace_report(start_ns, end_ns, by=by)
        from ..core.aggregator import _percentile
        ttft, tpot, e2e, ptok, otok = [], [], [], [], []
        ok = fail = 0
        by_pid: dict = {}
        span_s = max((end - start), 1)
        for r in rows:
            # rid, arrival, first_tok, finish, ttft, tpot, e2e, status, ptok, otok, steps, pid
            st = r[7] or "0"
            if st == "0":
                ok += 1
            else:
                fail += 1
            if r[4]:
                ttft.append(r[4])
            if r[5]:
                tpot.append(r[5])
            if r[6]:
                e2e.append(r[6])
            ptok.append(r[8] or 0)
            otok.append(r[9] or 0)
            by_pid[r[11]] = by_pid.get(r[11], 0) + 1
        out_tok = sum(otok)
        dur = end - start

        def stat(vals):
            s = sorted(vals)
            return {
                "count": len(s),
                "avg": round(sum(s) / len(s), 3) if s else 0.0,
                "p50": round(_percentile(s, 0.5), 3) if s else 0.0,
                "p95": round(_percentile(s, 0.95), 3) if s else 0.0,
                "p99": round(_percentile(s, 0.99), 3) if s else 0.0,
            }

        return {
            "start_sec": int(start), "end_sec": int(end),
            "span_sec": round(dur, 1),
            "total": len(rows), "ok": ok, "fail": fail,
            "qps": round(len(rows) / span_s, 3),
            "output_tokens": out_tok,
            "output_tok_per_sec": round(out_tok / span_s, 2),
            "ttft_ms": stat(ttft), "tpot_ms": stat(tpot), "e2e_ms": stat(e2e),
            "avg_prompt_tokens": round(sum(ptok) / len(ptok), 1) if ptok else 0,
            "avg_output_tokens": round(out_tok / len(rows), 1) if rows else 0,
            "by_pid": [{"pid": k, "count": v} for k, v in sorted(by_pid.items())],
        }

    @app.get("/api/history/transactions")
    def hist_tx(
        limit: int = Query(50, le=500),
        type: str | None = None,
        status: str | None = None,
    ) -> dict:
        if sqlite_store is None:
            return {"items": []}
        sql = "SELECT id, type, name, start_ts, duration_ns, status, sampled_reason FROM transactions"
        conds, params = [], []
        if type:
            conds.append("type=?")
            params.append(type)
        if status:
            conds.append("status=?")
            params.append(status)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY start_ts DESC LIMIT ?"
        params.append(limit)
        rows = sqlite_store.query(sql, tuple(params))
        return {"items": [
            {"id": r[0], "type": r[1], "name": r[2], "start_ts": r[3],
             "duration_ns": r[4], "status": r[5], "sampled_reason": r[6]}
            for r in rows
        ]}

    @app.get("/api/minute_agg")
    def minute(
        metric_type: str | None = None,
        since: int = 0,
        limit: int = Query(200, le=2000),
    ) -> dict:
        if sqlite_store is None:
            return {"items": []}
        sql = "SELECT metric_type, name, minute_ts, count, sum, p50, p95, p99 FROM minute_agg WHERE minute_ts>=?"
        params: list = [since]
        if metric_type:
            sql += " AND metric_type=?"
            params.append(metric_type)
        sql += " ORDER BY minute_ts DESC LIMIT ?"
        params.append(limit)
        rows = sqlite_store.query(sql, tuple(params))
        return {"items": [
            {"metric_type": r[0], "name": r[1], "minute_ts": r[2],
             "count": r[3], "sum": r[4], "p50": r[5], "p95": r[6], "p99": r[7]}
            for r in rows
        ]}

    @app.get("/metrics/prom", response_class=PlainTextResponse)
    def prom() -> str:
        stores = get_stores()
        rows = aggregate(
            stores.transactions.snapshot(),
            stores.metrics.snapshot(),
            stores.events.snapshot(),
        )
        return prometheus.render(stores.heartbeats.snapshot(), rows)

    @app.get("/api/latency/dist")
    def latency_dist(
        window_sec: int = Query(300, ge=10, le=86400),
        types: str = "http.request,vllm.generate,sglang.batch",
    ) -> dict:
        """请求耗时分布统计。
        返回 count / min / max / avg / p50/p95/p99 + 直方图桶。
        同时如果 tx.data.ttft_ns 存在(vllm.generate 流式),也返回 TTFT 分布。
        """
        import time as _t
        now = _t.time_ns()
        cutoff = now - window_sec * 1_000_000_000
        wanted = {t.strip() for t in types.split(",") if t.strip()}

        durs_ms: list[float] = []
        ttft_ms: list[float] = []
        for t in get_stores().transactions.snapshot():
            if t.type not in wanted:
                continue
            if t.start_ns < cutoff:
                continue
            durs_ms.append(t.duration_ns / 1e6)
            ttft = t.data.get("ttft_ns") if isinstance(t.data, dict) else None
            if isinstance(ttft, int | float) and ttft > 0:
                ttft_ms.append(float(ttft) / 1e6)

        return {
            "window_sec": window_sec,
            "latency": _dist_summary(durs_ms),
            "ttft": _dist_summary(ttft_ms),
        }

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    return app


def _tx_summary(t) -> dict:
    return {
        "tx_id": t.tx_id,
        "type": t.type,
        "name": t.name,
        "start_ns": t.start_ns,
        "duration_ns": t.duration_ns,
        "status": t.status,
        "child_count": _count_children(t),
    }


# 直方图桶(毫秒),覆盖 1ms ~ 60s
_LATENCY_BUCKETS_MS = [
    5, 10, 25, 50, 100, 200, 500,
    1000, 2000, 5000, 10000, 30000, 60000,
]


def _dist_summary(values_ms: list[float]) -> dict:
    """返回 count/min/max/avg/p50/p95/p99 + 直方图桶。"""
    n = len(values_ms)
    if n == 0:
        return {
            "count": 0, "min_ms": 0.0, "max_ms": 0.0, "avg_ms": 0.0,
            "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0,
            "buckets": [{"le_ms": b, "count": 0} for b in _LATENCY_BUCKETS_MS] +
                       [{"le_ms": None, "count": 0}],
        }
        sorted_vals = sorted(values_ms)  # unreachable
    sorted_vals = sorted(values_ms)

    def pct(p: float) -> float:
        idx = min(n - 1, int(round(p * (n - 1))))
        return sorted_vals[idx]

    # 直方图:每个桶记落在 (prev, le] 的数量;最后 +inf
    buckets = []
    prev = 0.0
    for le in _LATENCY_BUCKETS_MS:
        c = sum(1 for v in sorted_vals if prev < v <= le)
        buckets.append({"le_ms": le, "count": c})
        prev = float(le)
    inf_count = sum(1 for v in sorted_vals if v > _LATENCY_BUCKETS_MS[-1])
    buckets.append({"le_ms": None, "count": inf_count})

    return {
        "count": n,
        "min_ms": round(sorted_vals[0], 3),
        "max_ms": round(sorted_vals[-1], 3),
        "avg_ms": round(sum(sorted_vals) / n, 3),
        "p50_ms": round(pct(0.50), 3),
        "p95_ms": round(pct(0.95), 3),
        "p99_ms": round(pct(0.99), 3),
        "buckets": buckets,
    }


def _count_children(t) -> int:
    n = len(t.children)
    for c in t.children:
        n += _count_children(c)
    return n


class WebServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 9109, sqlite_store=None) -> None:
        self.host = host
        self.port = port
        self.sqlite_store = sqlite_store
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        app = _build_app(self.sqlite_store)
        config = uvicorn.Config(
            app, host=self.host, port=self.port,
            log_level="warning", access_log=False, lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self._thread = threading.Thread(
            target=self._server.run, name="llm-monitor-web", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=3.0)


