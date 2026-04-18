"""Tests for the Google Gemini adapter using mock objects."""

from dataclasses import dataclass, field
from typing import Any

import pytest

from aimeter.adapters.gemini import (
    _extract_tool_calls,
    _extract_usage,
    track_gemini,
)
from aimeter.exporters.memory import MemoryExporter
from aimeter.tracker import configure, reset

# --- Mock google-genai objects (match real SDK's attribute shapes) ---


@dataclass
class MockUsageMetadata:
    prompt_token_count: int = 0
    candidates_token_count: int = 0
    cached_content_token_count: int = 0
    total_token_count: int = 0


@dataclass
class MockFunctionCall:
    name: str = ""
    args: dict = field(default_factory=dict)


@dataclass
class MockPart:
    text: str | None = None
    function_call: MockFunctionCall | None = None


@dataclass
class MockContent:
    parts: list[MockPart] = field(default_factory=list)
    role: str = "model"


@dataclass
class MockCandidate:
    content: MockContent = field(default_factory=MockContent)


@dataclass
class MockGeminiResponse:
    model_version: str = "gemini-2.5-flash"
    usage_metadata: MockUsageMetadata | None = field(default_factory=MockUsageMetadata)
    candidates: list[MockCandidate] = field(default_factory=list)


class MockModels:
    def __init__(self, response: MockGeminiResponse | None = None):
        self._response = response or MockGeminiResponse()
        self.last_kwargs: dict = {}

    def generate_content(self, **kwargs: Any) -> MockGeminiResponse:
        self.last_kwargs = kwargs
        return self._response


class MockGeminiClient:
    def __init__(self, models: MockModels | None = None):
        self.models = models or MockModels()
        self.api_key = "test-key"


# --- Tests ---


class TestExtractUsage:
    def test_basic_usage(self):
        resp = MockGeminiResponse(
            usage_metadata=MockUsageMetadata(
                prompt_token_count=200, candidates_token_count=100
            )
        )
        usage = _extract_usage(resp)
        assert usage.input_tokens == 200
        assert usage.output_tokens == 100
        assert usage.cached_tokens == 0

    def test_cached_tokens(self):
        resp = MockGeminiResponse(
            usage_metadata=MockUsageMetadata(
                prompt_token_count=200,
                candidates_token_count=100,
                cached_content_token_count=50,
            )
        )
        usage = _extract_usage(resp)
        assert usage.cached_tokens == 50

    def test_no_usage(self):
        resp = MockGeminiResponse(usage_metadata=None)
        usage = _extract_usage(resp)
        assert usage.total_tokens == 0


class TestExtractToolCalls:
    def test_with_function_calls(self):
        resp = MockGeminiResponse(candidates=[
            MockCandidate(content=MockContent(parts=[
                MockPart(text="Let me check..."),
                MockPart(function_call=MockFunctionCall(name="search")),
                MockPart(function_call=MockFunctionCall(name="calculate")),
            ]))
        ])
        names = _extract_tool_calls(resp)
        assert names == ["search", "calculate"]

    def test_no_function_calls(self):
        resp = MockGeminiResponse(candidates=[
            MockCandidate(content=MockContent(parts=[MockPart(text="Hello")]))
        ])
        assert _extract_tool_calls(resp) == []

    def test_no_candidates(self):
        resp = MockGeminiResponse(candidates=[])
        assert _extract_tool_calls(resp) == []

    def test_multiple_candidates(self):
        resp = MockGeminiResponse(candidates=[
            MockCandidate(content=MockContent(parts=[
                MockPart(function_call=MockFunctionCall(name="search")),
            ])),
            MockCandidate(content=MockContent(parts=[
                MockPart(function_call=MockFunctionCall(name="translate")),
            ])),
        ])
        assert _extract_tool_calls(resp) == ["search", "translate"]


class TestTrackGemini:
    def setup_method(self):
        reset()
        self.mem = MemoryExporter()
        configure(exporters=[self.mem])

    def teardown_method(self):
        reset()

    def test_basic_tracking(self):
        response = MockGeminiResponse(
            model_version="gemini-2.5-flash",
            usage_metadata=MockUsageMetadata(
                prompt_token_count=200, candidates_token_count=100
            ),
        )
        mock_client = MockGeminiClient(models=MockModels(response))
        client = track_gemini(mock_client, project="test")

        result = client.models.generate_content(
            model="gemini-2.5-flash", contents="Hello"
        )

        assert result.model_version == "gemini-2.5-flash"
        assert len(self.mem.events) == 1
        event = self.mem.events[0]
        assert event.provider == "google"
        assert event.model == "gemini-2.5-flash"
        assert event.tokens.input_tokens == 200
        assert event.tokens.output_tokens == 100
        assert event.cost.total_cost_usd > 0
        assert event.latency_ms > 0

    def test_with_function_calls(self):
        response = MockGeminiResponse(
            usage_metadata=MockUsageMetadata(
                prompt_token_count=300, candidates_token_count=150
            ),
            candidates=[MockCandidate(content=MockContent(parts=[
                MockPart(function_call=MockFunctionCall(name="web_search")),
            ]))],
        )
        mock_client = MockGeminiClient(models=MockModels(response))
        client = track_gemini(mock_client)

        client.models.generate_content(model="gemini-2.5-flash", contents="Hi")

        event = self.mem.events[0]
        assert event.tool_calls == ["web_search"]

    def test_error_tracking(self):
        class FailingModels:
            def generate_content(self, **kwargs):
                raise RuntimeError("API error")

        mock_client = MockGeminiClient(models=FailingModels())
        client = track_gemini(mock_client)

        with pytest.raises(RuntimeError):
            client.models.generate_content(model="gemini-2.5-flash", contents="Hi")

        assert len(self.mem.events) == 1
        event = self.mem.events[0]
        assert event.error == "API error"
        assert event.model == "gemini-2.5-flash"

    def test_passthrough_attributes(self):
        mock_client = MockGeminiClient()
        client = track_gemini(mock_client)
        assert client.api_key == "test-key"

    def test_project_and_tags(self):
        response = MockGeminiResponse(
            usage_metadata=MockUsageMetadata(
                prompt_token_count=10, candidates_token_count=5
            ),
        )
        mock_client = MockGeminiClient(models=MockModels(response))
        client = track_gemini(mock_client, project="research", tags={"env": "staging"})

        client.models.generate_content(model="gemini-2.5-flash", contents="Hi")

        event = self.mem.events[0]
        assert event.project == "research"
        assert event.tags == {"env": "staging"}

    def test_run_id(self):
        response = MockGeminiResponse(
            usage_metadata=MockUsageMetadata(
                prompt_token_count=10, candidates_token_count=5
            ),
        )
        mock_client = MockGeminiClient(models=MockModels(response))
        client = track_gemini(mock_client, run_id="run-789")

        client.models.generate_content(model="gemini-2.5-flash", contents="Hi")

        event = self.mem.events[0]
        assert event.run_id == "run-789"

    def test_model_fallback_to_kwargs(self):
        """If response lacks model_version, fall back to the request model kwarg."""
        response = MockGeminiResponse(
            usage_metadata=MockUsageMetadata(
                prompt_token_count=10, candidates_token_count=5
            ),
        )
        response.model_version = None  # type: ignore
        mock_client = MockGeminiClient(models=MockModels(response))
        client = track_gemini(mock_client)

        client.models.generate_content(model="gemini-2.5-pro", contents="Hi")

        event = self.mem.events[0]
        assert event.model == "gemini-2.5-pro"
