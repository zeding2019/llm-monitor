"""分钟聚合器:把最近一分钟的 Transaction / Metric / Event 汇成一行。

线程模型:由 DbWriter 每分钟触发一次聚合 flush,从 stores 取快照 → 聚合。
计算:count / sum / max / p50 / p95 / p99。用简单排序法,分钟级样本量不大够用。
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class MinuteRow:
    metric_type: str
    name: str
    minute_ts: int
    count: int
    sum: float
    max: float
    p50: float
    p95: float
    p99: float
    tags_json: str = "{}"


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    idx = min(n - 1, int(q * n))
    return sorted_values[idx]


def minute_of(ts_ns: int) -> int:
    return (ts_ns // 1_000_000_000) // 60 * 60


def aggregate(
    transactions,
    metrics,
    events,
    now_ns: int | None = None,
) -> list[MinuteRow]:
    """把上一分钟的样本汇成 MinuteRow 列表。当前分钟未结束,不会被汇。"""
    now_ns = now_ns or time.time_ns()
    cur_min = minute_of(now_ns)

    # Transaction:按 (type|name, minute) 聚合 duration_ns(转 ms)
    tx_buckets: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for t in transactions:
        m = minute_of(t.start_ns)
        if m >= cur_min:
            continue
        # 消息树只汇根事务,子事务另计
        _collect_tx(t, m, tx_buckets)

    metric_buckets: dict[tuple[str, str, int, str], list[tuple[int, float]]] = defaultdict(list)
    for mm in metrics:
        m = minute_of(mm.ts_ns)
        if m >= cur_min:
            continue
        tags_key = json.dumps(mm.tags, sort_keys=True) if mm.tags else "{}"
        metric_buckets[("metric", mm.name, m, tags_key)].append((mm.count, mm.sum))

    ev_buckets: dict[tuple[str, str, int], int] = defaultdict(int)
    for e in events:
        m = minute_of(e.ts_ns)
        if m >= cur_min:
            continue
        ev_buckets[("event", f"{e.type}|{e.name}|{e.status}", m)] += 1

    rows: list[MinuteRow] = []

    for (type_, name, minute), durs in tx_buckets.items():
        durs.sort()
        rows.append(MinuteRow(
            metric_type="transaction",
            name=f"{type_}|{name}",
            minute_ts=minute,
            count=len(durs),
            sum=sum(durs),
            max=durs[-1],
            p50=_percentile(durs, 0.5),
            p95=_percentile(durs, 0.95),
            p99=_percentile(durs, 0.99),
        ))

    for (mtype, name, minute, tags_key), samples in metric_buckets.items():
        counts = [c for c, _ in samples]
        sums = [s for _, s in samples]
        vals = sorted(sums)
        rows.append(MinuteRow(
            metric_type=mtype,
            name=name,
            minute_ts=minute,
            count=sum(counts),
            sum=sum(sums),
            max=max(sums),
            p50=_percentile(vals, 0.5),
            p95=_percentile(vals, 0.95),
            p99=_percentile(vals, 0.99),
            tags_json=tags_key,
        ))

    for (mtype, name, minute), n in ev_buckets.items():
        rows.append(MinuteRow(
            metric_type=mtype, name=name, minute_ts=minute,
            count=n, sum=float(n), max=0.0, p50=0.0, p95=0.0, p99=0.0,
        ))

    return rows


def _collect_tx(t, minute, buckets):
    """汇 t 本身,并递归汇 children。所有耗时单位换成毫秒。"""
    buckets[(t.type, t.name, minute)].append(t.duration_ns / 1e6)
    for c in t.children:
        # 子事务用其自身 minute
        cm = minute_of(c.start_ns)
        _collect_tx(c, cm, buckets)
