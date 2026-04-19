"""Tests for aimeter.performance — aggregate performance analytics."""

from datetime import datetime, timedelta, timezone

from aimeter.performance import compute_performance
from aimeter.types import LLMEvent, TokenUsage


def _event(
    *,
    latency_ms: float = 100.0,
    output_tokens: int = 10,
    input_tokens: int = 0,
    provider: str = "openai",
    model: str = "gpt-4o",
    project: str = "default",
    tags: dict[str, str] | None = None,
    error: str | None = None,
    event_type: str = "llm.call",
    timestamp: str | None = None,
) -> LLMEvent:
    return LLMEvent(
        provider=provider,
        model=model,
        project=project,
        event_type=event_type,
        tokens=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        latency_ms=latency_ms,
        tags=tags or {},
        error=error,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
    )


class TestComputePerformance:
    def test_empty_events(self):
        p = compute_performance([])
        assert p["global"] is None
        assert p["errors"] == {"count": 0, "rate": 0.0}
        assert p["by_model"] == {}
        assert p["by_provider"] == {}
        assert p["by_project"] == {}
        assert p["by_tag"] == {}

    def test_outcome_events_ignored(self):
        events = [
            _event(latency_ms=100.0),
            _event(event_type="outcome", latency_ms=9999.0),
        ]
        p = compute_performance(events)
        assert p["global"]["count"] == 1

    def test_error_events_excluded_from_latency(self):
        events = [
            _event(latency_ms=100.0),
            _event(latency_ms=200.0),
            _event(latency_ms=9999.0, error="boom"),
        ]
        p = compute_performance(events)
        assert p["global"]["count"] == 2
        assert p["errors"]["count"] == 1
        assert abs(p["errors"]["rate"] - (1 / 3)) < 1e-9
        # Error latency must not pollute percentiles
        assert p["global"]["latency_ms"]["max"] == 200.0

    def test_single_event_percentiles(self):
        p = compute_performance([_event(latency_ms=250.0)])
        lat = p["global"]["latency_ms"]
        assert lat["p50"] == lat["p95"] == lat["p99"] == 250.0
        assert p["global"]["throughput"]["requests_per_sec"] is None
        assert p["global"]["throughput"]["output_tokens_per_sec"] is None
        assert p["global"]["throughput"]["window_seconds"] == 0.0

    def test_five_events_percentiles(self):
        events = [_event(latency_ms=v) for v in [10.0, 20.0, 30.0, 40.0, 50.0]]
        p = compute_performance(events)
        lat = p["global"]["latency_ms"]
        assert lat["min"] == 10.0
        assert lat["max"] == 50.0
        assert lat["mean"] == 30.0
        assert lat["p50"] == 30.0
        assert lat["p95"] == 50.0
        assert lat["p99"] == 50.0

    def test_throughput_window(self):
        t0 = datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=2)
        events = [
            _event(output_tokens=40, timestamp=t0.isoformat()),
            _event(output_tokens=60, timestamp=t1.isoformat()),
        ]
        p = compute_performance(events)
        tp = p["global"]["throughput"]
        assert tp["window_seconds"] == 2.0
        assert tp["requests_per_sec"] == 1.0
        assert tp["output_tokens_per_sec"] == 50.0

    def test_by_model_breakdown(self):
        events = [
            _event(provider="openai", model="gpt-4o"),
            _event(provider="openai", model="gpt-4o-mini"),
            _event(provider="anthropic", model="claude-sonnet-4-6"),
            _event(provider="anthropic", model="claude-haiku-4-5"),
        ]
        p = compute_performance(events)
        assert set(p["by_model"].keys()) == {
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5",
        }

    def test_by_provider_breakdown(self):
        events = [
            _event(provider="openai"),
            _event(provider="openai"),
            _event(provider="anthropic"),
        ]
        p = compute_performance(events)
        assert set(p["by_provider"].keys()) == {"openai", "anthropic"}
        assert p["by_provider"]["openai"]["count"] == 2
        assert p["by_provider"]["anthropic"]["count"] == 1

    def test_by_project_breakdown(self):
        events = [
            _event(project="a"),
            _event(project="b"),
            _event(project="default"),
        ]
        p = compute_performance(events)
        assert set(p["by_project"].keys()) == {"a", "b", "default"}

    def test_by_tag_breakdown_multi_key(self):
        events = [
            _event(tags={"env": "prod", "feature": "x"}),
            _event(tags={"env": "dev"}),
        ]
        p = compute_performance(events)
        assert set(p["by_tag"].keys()) == {"env", "feature"}
        assert set(p["by_tag"]["env"].keys()) == {"prod", "dev"}
        assert set(p["by_tag"]["feature"].keys()) == {"x"}

    def test_tag_key_missing_on_some_events(self):
        events = [
            _event(tags={"env": "prod"}),
            _event(tags={}),
        ]
        p = compute_performance(events)
        assert set(p["by_tag"]["env"].keys()) == {"prod"}
        assert p["by_tag"]["env"]["prod"]["count"] == 1

    def test_per_call_tokens_per_sec_summary(self):
        events = [
            _event(latency_ms=1000.0, output_tokens=100),  # 100 tok/s
            _event(latency_ms=500.0, output_tokens=100),  # 200 tok/s
            _event(latency_ms=2000.0, output_tokens=100),  # 50 tok/s
        ]
        p = compute_performance(events)
        s = p["global"]["output_tokens_per_sec_per_call"]
        assert s is not None
        assert s["min"] == 50.0
        assert s["max"] == 200.0
        assert abs(s["mean"] - (100.0 + 200.0 + 50.0) / 3) < 1e-9

    def test_per_call_tokens_per_sec_all_none(self):
        # All events have zero latency → property returns None → summary is None.
        # But _stats_for needs at least one latency_ms > 0 to return non-None.
        # Use positive latency but zero output tokens so per-call is None but bucket is non-empty.
        events = [
            _event(latency_ms=100.0, output_tokens=0),
            _event(latency_ms=200.0, output_tokens=0),
        ]
        p = compute_performance(events)
        assert p["global"]["output_tokens_per_sec_per_call"] is None
