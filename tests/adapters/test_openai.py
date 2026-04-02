"""Tests for the OpenAI adapter using mock objects.

No real OpenAI SDK needed — we mock the client and response shapes.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from aimeter.adapters.openai import (
    _extract_tool_calls,
    _extract_usage,
    track_openai,
)
from aimeter.exporters.memory import MemoryExporter
from aimeter.tracker import configure, reset

# --- Mock OpenAI objects (match real SDK's attribute shapes) ---


@dataclass
class MockUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_details: Any = None


@dataclass
class MockPromptTokensDetails:
    cached_tokens: int = 0


@dataclass
class MockFunctionCall:
    name: str = ""


@dataclass
class MockToolCall:
    function: MockFunctionCall = field(default_factory=MockFunctionCall)


@dataclass
class MockMessage:
    tool_calls: list[MockToolCall] | None = None


@dataclass
class MockChoice:
    message: MockMessage = field(default_factory=MockMessage)


@dataclass
class MockResponse:
    model: str = "gpt-4o"
    usage: MockUsage = field(default_factory=MockUsage)
    choices: list[MockChoice] = field(default_factory=list)


class MockCompletions:
    def __init__(self, response: MockResponse | None = None):
        self._response = response or MockResponse()
        self.last_kwargs: dict = {}

    def create(self, **kwargs: Any) -> MockResponse:
        self.last_kwargs = kwargs
        return self._response


class MockChat:
    def __init__(self, completions: MockCompletions | None = None):
        self.completions = completions or MockCompletions()


class MockOpenAIClient:
    def __init__(self, chat: MockChat | None = None):
        self.chat = chat or MockChat()
        self.api_key = "test-key"


# --- Tests ---


class TestExtractUsage:
    def test_basic_usage(self):
        resp = MockResponse(usage=MockUsage(prompt_tokens=100, completion_tokens=50))
        usage = _extract_usage(resp)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cached_tokens == 0

    def test_cached_tokens(self):
        details = MockPromptTokensDetails(cached_tokens=30)
        resp = MockResponse(usage=MockUsage(
            prompt_tokens=100, completion_tokens=50,
            prompt_tokens_details=details,
        ))
        usage = _extract_usage(resp)
        assert usage.cached_tokens == 30

    def test_no_usage(self):
        resp = MockResponse()
        resp.usage = None  # type: ignore
        usage = _extract_usage(resp)
        assert usage.total_tokens == 0


class TestExtractToolCalls:
    def test_with_tool_calls(self):
        resp = MockResponse(choices=[
            MockChoice(message=MockMessage(tool_calls=[
                MockToolCall(function=MockFunctionCall(name="search")),
                MockToolCall(function=MockFunctionCall(name="calculate")),
            ]))
        ])
        names = _extract_tool_calls(resp)
        assert names == ["search", "calculate"]

    def test_no_tool_calls(self):
        resp = MockResponse(choices=[MockChoice()])
        assert _extract_tool_calls(resp) == []

    def test_no_choices(self):
        resp = MockResponse(choices=[])
        assert _extract_tool_calls(resp) == []


class TestTrackOpenAI:
    def setup_method(self):
        reset()
        self.mem = MemoryExporter()
        configure(exporters=[self.mem])

    def teardown_method(self):
        reset()

    def test_basic_tracking(self):
        response = MockResponse(
            model="gpt-4o",
            usage=MockUsage(prompt_tokens=100, completion_tokens=50),
        )
        mock_client = MockOpenAIClient(
            chat=MockChat(completions=MockCompletions(response))
        )
        client = track_openai(mock_client, project="test")

        result = client.chat.completions.create(model="gpt-4o", messages=[])

        assert result.model == "gpt-4o"
        assert len(self.mem.events) == 1
        event = self.mem.events[0]
        assert event.provider == "openai"
        assert event.model == "gpt-4o"
        assert event.tokens.input_tokens == 100
        assert event.tokens.output_tokens == 50
        assert event.cost.total_cost_usd > 0
        assert event.latency_ms > 0

    def test_with_tool_calls(self):
        response = MockResponse(
            model="gpt-4o",
            usage=MockUsage(prompt_tokens=200, completion_tokens=100),
            choices=[MockChoice(message=MockMessage(tool_calls=[
                MockToolCall(function=MockFunctionCall(name="get_weather")),
            ]))],
        )
        mock_client = MockOpenAIClient(
            chat=MockChat(completions=MockCompletions(response))
        )
        client = track_openai(mock_client)

        client.chat.completions.create(model="gpt-4o", messages=[])

        event = self.mem.events[0]
        assert event.tool_calls == ["get_weather"]

    def test_error_tracking(self):
        class FailingCompletions:
            def create(self, **kwargs):
                raise RuntimeError("API error")

        mock_client = MockOpenAIClient(
            chat=MockChat(completions=FailingCompletions())
        )
        client = track_openai(mock_client)

        with pytest.raises(RuntimeError):
            client.chat.completions.create(model="gpt-4o", messages=[])

        assert len(self.mem.events) == 1
        event = self.mem.events[0]
        assert event.error == "API error"
        assert event.model == "gpt-4o"

    def test_passthrough_attributes(self):
        mock_client = MockOpenAIClient()
        client = track_openai(mock_client)
        assert client.api_key == "test-key"

    def test_project_and_tags(self):
        response = MockResponse(
            usage=MockUsage(prompt_tokens=10, completion_tokens=5),
        )
        mock_client = MockOpenAIClient(
            chat=MockChat(completions=MockCompletions(response))
        )
        client = track_openai(mock_client, project="sales", tags={"team": "cx"})

        client.chat.completions.create(model="gpt-4o", messages=[])

        event = self.mem.events[0]
        assert event.project == "sales"
        assert event.tags == {"team": "cx"}

    def test_run_id(self):
        response = MockResponse(
            usage=MockUsage(prompt_tokens=10, completion_tokens=5),
        )
        mock_client = MockOpenAIClient(
            chat=MockChat(completions=MockCompletions(response))
        )
        client = track_openai(mock_client, run_id="run-456")

        client.chat.completions.create(model="gpt-4o", messages=[])

        event = self.mem.events[0]
        assert event.run_id == "run-456"

    def test_cached_tokens(self):
        details = MockPromptTokensDetails(cached_tokens=40)
        response = MockResponse(
            usage=MockUsage(
                prompt_tokens=100, completion_tokens=50,
                prompt_tokens_details=details,
            ),
        )
        mock_client = MockOpenAIClient(
            chat=MockChat(completions=MockCompletions(response))
        )
        client = track_openai(mock_client)

        client.chat.completions.create(model="gpt-4o", messages=[])

        event = self.mem.events[0]
        assert event.tokens.cached_tokens == 40
