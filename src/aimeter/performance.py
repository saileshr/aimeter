"""Performance analytics over recorded LLMEvents.

Pure, stdlib-only aggregation. Given a list of events, computes latency
percentiles, throughput, and error counts — globally and broken down by
model, provider, project, and tag key.

Used by MemoryExporter.summary() and report.print_report().
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from statistics import fmean
from typing import Any

from aimeter.types import LLMEvent


def compute_performance(events: list[LLMEvent]) -> dict[str, Any]:
    """Aggregate performance stats over a list of events.

    Excludes outcome events entirely. Events with `error is not None` feed
    only the `errors` block; latency and throughput aggregates are computed
    from successful events only.
    """
    eligible = [e for e in events if e.event_type != "outcome"]
    ok = [e for e in eligible if e.error is None]
    err_count = len(eligible) - len(ok)
    err_rate = (err_count / len(eligible)) if eligible else 0.0

    by_model: dict[str, list[LLMEvent]] = defaultdict(list)
    by_provider: dict[str, list[LLMEvent]] = defaultdict(list)
    by_project: dict[str, list[LLMEvent]] = defaultdict(list)
    for e in ok:
        label = f"{e.provider}/{e.model}" if e.provider else e.model
        by_model[label].append(e)
        if e.provider:
            by_provider[e.provider].append(e)
        by_project[e.project or "default"].append(e)

    all_tag_keys = {k for e in ok for k in e.tags}
    by_tag: dict[str, dict[str, Any]] = {}
    for key in all_tag_keys:
        buckets: dict[str, list[LLMEvent]] = defaultdict(list)
        for e in ok:
            if key in e.tags:
                buckets[e.tags[key]].append(e)
        by_tag[key] = {v: _stats_for(b) for v, b in buckets.items()}

    return {
        "global": _stats_for(ok),
        "errors": {"count": err_count, "rate": err_rate},
        "by_model": {k: _stats_for(v) for k, v in by_model.items()},
        "by_provider": {k: _stats_for(v) for k, v in by_provider.items()},
        "by_project": {k: _stats_for(v) for k, v in by_project.items()},
        "by_tag": by_tag,
    }


def _stats_for(bucket: list[LLMEvent]) -> dict[str, Any] | None:
    latencies = [e.latency_ms for e in bucket if e.latency_ms > 0]
    if not latencies:
        return None
    latencies_sorted = sorted(latencies)

    window_seconds = _timestamp_span_seconds(bucket)
    count = len(bucket)
    total_output_tokens = sum(e.tokens.output_tokens for e in bucket)
    if window_seconds > 0:
        rps: float | None = count / window_seconds
        tps: float | None = total_output_tokens / window_seconds
    else:
        rps = None
        tps = None

    per_call = [v for v in (e.output_tokens_per_sec for e in bucket) if v is not None]
    per_call_summary: dict[str, float] | None
    if per_call:
        per_call_summary = {
            "min": min(per_call),
            "max": max(per_call),
            "mean": fmean(per_call),
        }
    else:
        per_call_summary = None

    return {
        "count": count,
        "latency_ms": {
            "min": min(latencies_sorted),
            "max": max(latencies_sorted),
            "mean": fmean(latencies_sorted),
            "p50": _pct(latencies_sorted, 50),
            "p95": _pct(latencies_sorted, 95),
            "p99": _pct(latencies_sorted, 99),
        },
        "throughput": {
            "window_seconds": window_seconds,
            "requests_per_sec": rps,
            "output_tokens_per_sec": tps,
        },
        "output_tokens_per_sec_per_call": per_call_summary,
    }


def _pct(xs_sorted: list[float], p: int) -> float:
    """Nearest-rank percentile. Assumes xs_sorted is non-empty and sorted."""
    n = len(xs_sorted)
    if n == 1:
        return xs_sorted[0]
    k = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
    return xs_sorted[k]


def _timestamp_span_seconds(bucket: list[LLMEvent]) -> float:
    if len(bucket) <= 1:
        return 0.0
    ts = [datetime.fromisoformat(e.timestamp) for e in bucket]
    return (max(ts) - min(ts)).total_seconds()
