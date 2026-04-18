"""Google Gemini adapter — thin wrapper for cost tracking.

Wraps a google-genai client to automatically track token usage and costs
for all generate_content calls.

Usage:
    from aimeter import track_gemini
    from google import genai

    client = track_gemini(genai.Client(api_key="..."), project="my-agent")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Hello",
    )
    # Tokens, cost, and latency are automatically tracked.
"""

from __future__ import annotations

import time
from typing import Any

from aimeter.tracker import get_tracker
from aimeter.types import LLMEvent, TokenUsage


def _extract_usage(response: Any) -> TokenUsage:
    """Extract token usage from a google-genai response object."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return TokenUsage()

    return TokenUsage(
        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        cached_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
    )


def _extract_tool_calls(response: Any) -> list[str]:
    """Extract function call names from a google-genai response (names only, for privacy)."""
    try:
        candidates = getattr(response, "candidates", None) or []
        names: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                function_call = getattr(part, "function_call", None)
                if function_call is None:
                    continue
                name = getattr(function_call, "name", None)
                if name:
                    names.append(name)
        return names
    except (TypeError, AttributeError):
        return []


def _extract_model(response: Any, kwargs: dict[str, Any]) -> str:
    """Get model name from response (preferred) or request kwargs."""
    return (
        getattr(response, "model_version", None)
        or getattr(response, "model", None)
        or kwargs.get("model", "")
    )


class _TrackedModels:
    """Proxy for client.models that tracks generate_content() calls."""

    def __init__(self, models: Any, project: str, tags: dict[str, str],
                 run_id: str) -> None:
        self._models = models
        self._project = project
        self._tags = tags
        self._run_id = run_id

    def generate_content(self, **kwargs: Any) -> Any:
        """Call models.generate_content and record the event."""
        start = time.perf_counter_ns()
        error = None

        try:
            response = self._models.generate_content(**kwargs)
        except Exception as exc:
            error = str(exc)
            latency_ms = (time.perf_counter_ns() - start) / 1_000_000
            event = LLMEvent(
                run_id=self._run_id,
                project=self._project,
                provider="google",
                model=kwargs.get("model", ""),
                event_type="llm.call",
                tokens=TokenUsage(),
                latency_ms=latency_ms,
                error=error,
                tags=dict(self._tags),
            )
            get_tracker().record(event)
            raise

        latency_ms = (time.perf_counter_ns() - start) / 1_000_000
        event = LLMEvent(
            run_id=self._run_id,
            project=self._project,
            provider="google",
            model=_extract_model(response, kwargs),
            event_type="llm.call",
            tokens=_extract_usage(response),
            latency_ms=latency_ms,
            tool_calls=_extract_tool_calls(response),
            tags=dict(self._tags),
        )
        get_tracker().record(event)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._models, name)


class _TrackedGemini:
    """Proxy around a google-genai client that tracks all LLM calls."""

    def __init__(self, client: Any, project: str, tags: dict[str, str],
                 run_id: str) -> None:
        self._client = client
        self.models = _TrackedModels(client.models, project, tags, run_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def track_gemini(
    client: Any,
    *,
    project: str = "default",
    tags: dict[str, str] | None = None,
    run_id: str = "",
) -> Any:
    """Wrap a google-genai client to automatically track costs.

    Args:
        client: A google.genai.Client() instance.
        project: Project name for grouping events.
        tags: Optional tags for filtering/grouping.
        run_id: Optional run ID for grouping multiple calls.

    Returns:
        A wrapped client that behaves identically but tracks all LLM calls.

    Example:
        from google import genai
        from aimeter import track_gemini

        client = track_gemini(genai.Client(api_key="..."), project="my-agent")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hello",
        )
        # Cost and tokens are automatically tracked.
    """
    return _TrackedGemini(client, project, tags or {}, run_id)
