"""Tests for the memory exporter."""

from agentmeter.exporters.memory import MemoryExporter
from agentmeter.types import CostBreakdown, LLMEvent, TokenUsage


class TestMemoryExporter:
    def test_stores_events(self):
        exporter = MemoryExporter()
        events = [LLMEvent(model="gpt-4o"), LLMEvent(model="gpt-4o-mini")]
        exporter.export(events)
        assert len(exporter.events) == 2

    def test_total_cost(self):
        exporter = MemoryExporter()
        exporter.export([
            LLMEvent(cost=CostBreakdown(input_cost_usd=0.01, output_cost_usd=0.02)),
            LLMEvent(cost=CostBreakdown(input_cost_usd=0.005, output_cost_usd=0.01)),
        ])
        assert abs(exporter.total_cost - 0.045) < 1e-9

    def test_total_tokens(self):
        exporter = MemoryExporter()
        exporter.export([
            LLMEvent(tokens=TokenUsage(input_tokens=100, output_tokens=50)),
            LLMEvent(tokens=TokenUsage(input_tokens=200, output_tokens=100)),
        ])
        assert exporter.total_tokens == 450

    def test_events_by_run(self):
        exporter = MemoryExporter()
        exporter.export([
            LLMEvent(run_id="run-1", model="a"),
            LLMEvent(run_id="run-2", model="b"),
            LLMEvent(run_id="run-1", model="c"),
        ])
        run1 = exporter.events_by_run("run-1")
        assert len(run1) == 2
        assert all(e.run_id == "run-1" for e in run1)

    def test_summary(self):
        exporter = MemoryExporter()
        exporter.export([
            LLMEvent(
                provider="openai",
                model="gpt-4o",
                tokens=TokenUsage(input_tokens=100),
                cost=CostBreakdown(input_cost_usd=0.01),
            ),
        ])
        s = exporter.summary()
        assert s["event_count"] == 1
        assert s["total_cost_usd"] == 0.01
        assert s["total_tokens"] == 100
        assert "gpt-4o" in s["models_used"]
        assert "openai" in s["providers_used"]

    def test_clear(self):
        exporter = MemoryExporter()
        exporter.export([LLMEvent()])
        assert len(exporter.events) == 1
        exporter.clear()
        assert len(exporter.events) == 0

    def test_empty_summary(self):
        exporter = MemoryExporter()
        s = exporter.summary()
        assert s["event_count"] == 0
        assert s["total_cost_usd"] == 0.0
