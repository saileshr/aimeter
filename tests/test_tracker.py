"""Tests for agentmeter.tracker — core engine."""

from agentmeter.config import AgentMeterConfig
from agentmeter.exporters.memory import MemoryExporter
from agentmeter.tracker import Tracker, configure, get_tracker, reset
from agentmeter.types import CostBreakdown, LLMEvent, TokenUsage


class TestTracker:
    def test_record_enriches_cost(self):
        mem = MemoryExporter()
        config = AgentMeterConfig(exporters=[mem])
        tracker = Tracker(config)

        event = LLMEvent(
            provider="openai",
            model="gpt-4o",
            tokens=TokenUsage(input_tokens=1000, output_tokens=500),
        )
        result = tracker.record(event)

        assert result.cost.total_cost_usd > 0
        assert abs(result.cost.input_cost_usd - 0.0025) < 1e-9
        assert abs(result.cost.output_cost_usd - 0.005) < 1e-9
        assert len(mem.events) == 1

    def test_record_preserves_existing_cost(self):
        mem = MemoryExporter()
        config = AgentMeterConfig(exporters=[mem])
        tracker = Tracker(config)

        custom_cost = CostBreakdown(input_cost_usd=0.99)
        event = LLMEvent(
            provider="openai",
            model="gpt-4o",
            tokens=TokenUsage(input_tokens=1000),
            cost=custom_cost,
        )
        result = tracker.record(event)
        assert result.cost.input_cost_usd == 0.99

    def test_record_applies_default_project(self):
        mem = MemoryExporter()
        config = AgentMeterConfig(project="my-project", exporters=[mem])
        tracker = Tracker(config)

        event = LLMEvent(provider="openai", model="gpt-4o")
        result = tracker.record(event)
        assert result.project == "my-project"

    def test_record_applies_default_tags(self):
        mem = MemoryExporter()
        config = AgentMeterConfig(tags={"team": "cx"}, exporters=[mem])
        tracker = Tracker(config)

        event = LLMEvent()
        result = tracker.record(event)
        assert result.tags == {"team": "cx"}

    def test_record_does_not_overwrite_event_tags(self):
        mem = MemoryExporter()
        config = AgentMeterConfig(tags={"team": "cx"}, exporters=[mem])
        tracker = Tracker(config)

        event = LLMEvent(tags={"env": "prod"})
        result = tracker.record(event)
        assert result.tags == {"env": "prod"}

    def test_disabled_tracker(self):
        mem = MemoryExporter()
        config = AgentMeterConfig(enabled=False, exporters=[mem])
        tracker = Tracker(config)

        event = LLMEvent(provider="openai", model="gpt-4o",
                         tokens=TokenUsage(input_tokens=1000))
        result = tracker.record(event)
        assert result.cost.total_cost_usd == 0.0
        assert len(mem.events) == 0

    def test_exporter_error_does_not_raise(self):
        class BrokenExporter:
            def export(self, events):
                raise RuntimeError("boom")
            def shutdown(self):
                pass

        config = AgentMeterConfig(exporters=[BrokenExporter()])
        tracker = Tracker(config)
        # Should not raise
        tracker.record(LLMEvent())

    def test_multiple_exporters(self):
        mem1 = MemoryExporter()
        mem2 = MemoryExporter()
        config = AgentMeterConfig(exporters=[mem1, mem2])
        tracker = Tracker(config)

        tracker.record(LLMEvent())
        assert len(mem1.events) == 1
        assert len(mem2.events) == 1


class TestGlobalTracker:
    def setup_method(self):
        reset()

    def teardown_method(self):
        reset()

    def test_get_tracker_creates_singleton(self):
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2

    def test_configure_replaces_tracker(self):
        t1 = get_tracker()
        t2 = configure(project="new")
        assert t1 is not t2
        assert t2.config.project == "new"

    def test_configure_with_exporters(self):
        mem = MemoryExporter()
        tracker = configure(exporters=[mem])
        tracker.record(LLMEvent(provider="openai", model="gpt-4o",
                                tokens=TokenUsage(input_tokens=100)))
        assert len(mem.events) == 1

    def test_reset_clears_singleton(self):
        get_tracker()
        reset()
        # Next call creates a new tracker
        t = get_tracker()
        assert t is not None
