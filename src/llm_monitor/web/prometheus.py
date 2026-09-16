"""Prometheus text 格式导出。

只暴露最近一次 heartbeat + 最近 5 分钟聚合的 count/p95/p99。
Prometheus 用 pull,我们省去客户端库,自己拼字符串,零依赖。
"""
from __future__ import annotations

import time


def render(hbs_snapshot, minute_rows) -> str:
    lines: list[str] = []

    # Heartbeat 展开成 gauge
    latest_by_source: dict = {}
    for h in hbs_snapshot:
        latest_by_source[h.source] = h  # 后到覆盖旧
    for source, h in latest_by_source.items():
        safe_src = source.replace(":", "_").replace("-", "_")
        for k, v in h.values.items():
            metric = f"llm_monitor_{safe_src}_{k}"
            lines.append(f"# TYPE {metric} gauge")
            lines.append(f"{metric} {v}")

    # 分钟聚合展开
    for r in minute_rows:
        metric_base = _safe(f"llm_monitor_{r.metric_type}_{r.name}")
        labels = f'{{minute="{r.minute_ts}"}}'
        lines.append(f"{metric_base}_count{labels} {r.count}")
        lines.append(f"{metric_base}_sum{labels} {r.sum}")
        if r.metric_type == "transaction":
            lines.append(f"{metric_base}_p95{labels} {r.p95}")
            lines.append(f"{metric_base}_p99{labels} {r.p99}")

    lines.append(f"# scraped_at {int(time.time())}")
    return "\n".join(lines) + "\n"


def _safe(name: str) -> str:
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() or ch == "_" else "_")
    return "".join(out)
