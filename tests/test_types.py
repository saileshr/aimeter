"""Tests for aimeter.types — core data model."""

import json

import pytest

from aimeter.types import CostBreakdown, LLMEvent, Outcome, TokenUsage


class TestTokenUsage:
    def test_defaults_to_zero(self):
        t = TokenUsage()
        assert t.input_tokens == 0
        assert t.output_tokens == 0
        assert t.cached_tokens == 0
        assert t.total_tokens == 0

    def test_total_is_sum(self):
        t = TokenUsage(input_tokens=100, output_tokens=50, cached_tokens=25)
        assert t.total_tokens == 175

    def test_frozen(self):
        t = TokenUsage(input_tokens=10)
        with pytest.raises(AttributeError):
            t.input_tokens = 20  # type: ignore[misc]


class TestCostBreakdown:
    def test_defaults_to_zero(self):
        c = CostBreakdown()
        assert c.total_cost_usd == 0.0

    def test_total_is_sum(self):
        c = CostBreakdown(input_cost_usd=0.01, output_cost_usd=0.05, cached_cost_usd=0.002)
        assert abs(c.total_cost_usd - 0.062) < 1e-9

    def test_frozen(self):
        c = CostBreakdown(input_cost_usd=0.01)
        with pytest.raises(AttributeError):
            c.input_cost_usd = 0.02  # type: ignore[misc]


class TestLLMEvent:
    def test_defaults(self):
        e = LLMEvent()
        assert e.event_id  # non-empty UUID
        assert e.project == "default"
        assert e.event_type == "llm.call"
        assert e.tokens.total_tokens == 0
        assert e.cost.total_cost_usd == 0.0
        assert e.error is None
        assert e.tool_calls == []
        assert e.tags == {}

    def test_unique_event_ids(self):
        e1 = LLMEvent()
        e2 = LLMEvent()
        assert e1.event_id != e2.event_id

    def test_to_dict_roundtrip(self):
        e = LLMEvent(
            provider="openai",
            model="gpt-4o",
            tokens=TokenUsage(input_tokens=100, output_tokens=50),
            cost=CostBreakdown(input_cost_usd=0.00025, output_cost_usd=0.0005),
            latency_ms=123.4,
            tool_calls=["search"],
            tags={"team": "cx"},
        )
        d = e.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["provider"] == "openai"
        assert parsed["model"] == "gpt-4o"
        assert parsed["tokens"]["input_tokens"] == 100
        assert parsed["tokens"]["total_tokens"] == 150
        assert parsed["cost"]["total_cost_usd"] == 0.00075
        assert parsed["latency_ms"] == 123.4
        assert parsed["tool_calls"] == ["search"]
        assert parsed["tags"] == {"team": "cx"}

    def test_to_dict_has_all_fields(self):
        e = LLMEvent()
        d = e.to_dict()
        expected_keys = {
            "event_id", "run_id", "project", "provider", "model",
            "event_type", "tokens", "cost", "latency_ms", "tool_calls",
            "error", "tags", "timestamp", "metadata",
        }
        assert set(d.keys()) == expected_keys


class TestOutcome:
    def test_defaults(self):
        o = Outcome()
        assert o.outcome_id
        assert o.value_usd is None
        assert o.outcome == ""

    def test_with_value(self):
        o = Outcome(
            run_id="run-123",
            outcome="ticket_resolved",
            value_usd=12.50,
            metadata={"ticket_id": "T-1234"},
        )
        assert o.run_id == "run-123"
        assert o.outcome == "ticket_resolved"
        assert o.value_usd == 12.50
        assert o.metadata["ticket_id"] == "T-1234"

    def test_frozen(self):
        o = Outcome(outcome="test")
        with pytest.raises(AttributeError):
            o.outcome = "changed"  # type: ignore[misc]
