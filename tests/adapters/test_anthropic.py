"""Tests for the Anthropic adapter using mock objects."""

from dataclasses import dataclass, field
from typing import Any

import pytest

from aimeter.adapters.anthropic import (
    _extract_tool_calls,
    _extract_usage,
    track_anthropic,
)
from aimeter.exporters.memory import MemoryExporter
from aimeter.tracker import configure, reset

# --- Mock Anthropic objects ---


@dataclass
class MockAnthropicUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class MockToolUseBlock:
    type: str = "tool_use"
    name: str = ""


@dataclass
class MockTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class MockAnthropicResponse:
    model: str = "claude-sonnet-4-20250514"
    usage: MockAnthropicUsage = field(default_factory=MockAnthropicUsage)
    content: list = field(default_factory=list)


class MockMessages:
    def __init__(self, response: MockAnthropicResponse | None = None):
        self._response = response or MockAnthropicResponse()

    def create(self, **kwargs: Any) -> MockAnthropicResponse:
        return self._response


class MockAnthropicClient:
    def __init__(self, messages: MockMessages | None = None):
        self.messages = messages or MockMessages()
        self.api_key = "test-key"


# --- Tests ---


class TestExtractUsage:
    def test_basic_usage(self):
        resp = MockAnthropicResponse(
            usage=MockAnthropicUsage(input_tokens=200, output_tokens=100)
        )
        usage = _extract_usage(resp)
        assert usage.input_tokens == 200
        assert usage.output_tokens == 100
        assert usage.cached_tokens == 0

    def test_cached_tokens(self):
        resp = MockAnthropicResponse(
            usage=MockAnthropicUsage(
                input_tokens=200, output_tokens=100, cache_read_input_tokens=50
            )
        )
        usage = _extract_usage(resp)
        assert usage.cached_tokens == 50

    def test_no_usage(self):
        resp = MockAnthropicResponse()
        resp.usage = None  # type: ignore
        usage = _extract_usage(resp)
        assert usage.total_tokens == 0


class TestExtractToolCalls:
    def test_with_tool_use(self):
        resp = MockAnthropicResponse(content=[
            MockTextBlock(text="Let me search..."),
            MockToolUseBlock(name="search"),
            MockToolUseBlock(name="calculate"),
        ])
        names = _extract_tool_calls(resp)
        assert names == ["search", "calculate"]

    def test_no_tool_use(self):
        resp = MockAnthropicResponse(content=[MockTextBlock(text="Hello")])
        assert _extract_tool_calls(resp) == []

    def test_empty_content(self):
        resp = MockAnthropicResponse(content=[])
        assert _extract_tool_calls(resp) == []


class TestTrackAnthropic:
    def setup_method(self):
        reset()
        self.mem = MemoryExporter()
        configure(exporters=[self.mem])

    def teardown_method(self):
        reset()

    def test_basic_tracking(self):
        response = MockAnthropicResponse(
            model="claude-sonnet-4-20250514",
            usage=MockAnthropicUsage(input_tokens=200, output_tokens=100),
        )
        mock_client = MockAnthropicClient(messages=MockMessages(response))
        client = track_anthropic(mock_client, project="test")

        result = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024, messages=[]
        )

        assert result.model == "claude-sonnet-4-20250514"
        assert len(self.mem.events) == 1
        event = self.mem.events[0]
        assert event.provider == "anthropic"
        assert event.tokens.input_tokens == 200
        assert event.tokens.output_tokens == 100
        assert event.cost.total_cost_usd > 0
        assert event.latency_ms > 0

    def test_with_tool_use(self):
        response = MockAnthropicResponse(
            usage=MockAnthropicUsage(input_tokens=300, output_tokens=150),
            content=[MockToolUseBlock(name="web_search")],
        )
        mock_client = MockAnthropicClient(messages=MockMessages(response))
        client = track_anthropic(mock_client)

        client.messages.create(model="claude-sonnet-4-20250514", max_tokens=1024, messages=[])

        event = self.mem.events[0]
        assert event.tool_calls == ["web_search"]

    def test_error_tracking(self):
        class FailingMessages:
            def create(self, **kwargs):
                raise RuntimeError("API error")

        mock_client = MockAnthropicClient(messages=FailingMessages())
        client = track_anthropic(mock_client)

        with pytest.raises(RuntimeError):
            client.messages.create(model="claude-sonnet-4-20250514", max_tokens=1024, messages=[])

        assert len(self.mem.events) == 1
        assert self.mem.events[0].error == "API error"

    def test_passthrough_attributes(self):
        mock_client = MockAnthropicClient()
        client = track_anthropic(mock_client)
        assert client.api_key == "test-key"

    def test_project_and_tags(self):
        response = MockAnthropicResponse(
            usage=MockAnthropicUsage(input_tokens=10, output_tokens=5),
        )
        mock_client = MockAnthropicClient(messages=MockMessages(response))
        client = track_anthropic(mock_client, project="research", tags={"env": "staging"})

        client.messages.create(model="claude-sonnet-4-20250514", max_tokens=1024, messages=[])

        event = self.mem.events[0]
        assert event.project == "research"
        assert event.tags == {"env": "staging"}
