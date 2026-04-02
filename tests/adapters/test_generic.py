"""Tests for the generic adapter."""

from agentmeter.adapters.generic import track_llm_call
from agentmeter.config import AgentMeterConfig
from agentmeter.exporters.memory import MemoryExporter
from agentmeter.tracker import configure, reset


class TestTrackLLMCall:
    def setup_method(self):
        reset()
        self.mem = MemoryExporter()
        configure(exporters=[self.mem])

    def teardown_method(self):
        reset()

    def test_basic_tracking(self):
        with track_llm_call(provider="openai", model="gpt-4o") as call:
            call.input_tokens = 100
            call.output_tokens = 50

        assert len(self.mem.events) == 1
        event = self.mem.events[0]
        assert event.provider == "openai"
        assert event.model == "gpt-4o"
        assert event.tokens.input_tokens == 100
        assert event.tokens.output_tokens == 50
        assert event.cost.total_cost_usd > 0
        assert event.latency_ms > 0

    def test_captures_error(self):
        try:
            with track_llm_call(provider="openai", model="gpt-4o") as call:
                call.input_tokens = 50
                raise ValueError("test error")
        except ValueError:
            pass

        assert len(self.mem.events) == 1
        event = self.mem.events[0]
        assert event.error == "test error"

    def test_project_and_tags(self):
        with track_llm_call(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            project="my-project",
            tags={"env": "prod"},
        ) as call:
            call.input_tokens = 200

        event = self.mem.events[0]
        assert event.project == "my-project"
        assert event.tags == {"env": "prod"}

    def test_run_id(self):
        with track_llm_call(
            provider="openai", model="gpt-4o", run_id="run-123"
        ) as call:
            call.input_tokens = 10

        event = self.mem.events[0]
        assert event.run_id == "run-123"

    def test_tool_calls(self):
        with track_llm_call(provider="openai", model="gpt-4o") as call:
            call.input_tokens = 100
            call.tool_calls = ["search", "calculate"]

        event = self.mem.events[0]
        assert event.tool_calls == ["search", "calculate"]

    def test_metadata(self):
        with track_llm_call(provider="openai", model="gpt-4o") as call:
            call.metadata["request_id"] = "abc"

        event = self.mem.events[0]
        assert event.metadata["request_id"] == "abc"

    def test_cached_tokens(self):
        with track_llm_call(provider="openai", model="gpt-4o") as call:
            call.input_tokens = 100
            call.cached_tokens = 50

        event = self.mem.events[0]
        assert event.tokens.cached_tokens == 50
