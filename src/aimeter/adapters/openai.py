"""OpenAI adapter — thin wrapper for cost tracking.

Wraps an OpenAI client to automatically track token usage and costs
for all chat completion calls.

Usage:
    from aimeter import track_openai
    import openai

    client = track_openai(openai.OpenAI(), project="my-agent")
    response = client.chat.completions.create(model="gpt-4o", messages=[...])
    # Tokens, cost, and latency are automatically tracked.
"""

from __future__ import annotations

import time
from typing import Any

from aimeter.tracker import get_tracker
from aimeter.types import LLMEvent, TokenUsage


def _extract_usage(response: Any) -> TokenUsage:
    """Extract token usage from an OpenAI response object."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()

    return TokenUsage(
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cached_tokens=getattr(
            getattr(usage, "prompt_tokens_details", None),
            "cached_tokens",
            0,
        ) or 0,
    )


def _extract_tool_calls(response: Any) -> list[str]:
    """Extract tool call function names from an OpenAI response (names only, for privacy)."""
    try:
        choices = getattr(response, "choices", [])
        if not choices:
            return []
        message = getattr(choices[0], "message", None)
        if message is None:
            return []
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return []
        return [
            getattr(getattr(tc, "function", None), "name", "unknown")
            for tc in tool_calls
        ]
    except (IndexError, AttributeError):
        return []


def _extract_model(response: Any, kwargs: dict[str, Any]) -> str:
    """Get model name from response (preferred) or request kwargs."""
    return getattr(response, "model", None) or kwargs.get("model", "")


class _TrackedCompletions:
    """Proxy for client.chat.completions that tracks create() calls."""

    def __init__(self, completions: Any, project: str, tags: dict[str, str],
                 run_id: str) -> None:
        self._completions = completions
        self._project = project
        self._tags = tags
        self._run_id = run_id

    def create(self, **kwargs: Any) -> Any:
        """Call chat.completions.create and record the event."""
        start = time.perf_counter_ns()
        error = None

        try:
            response = self._completions.create(**kwargs)
        except Exception as exc:
            error = str(exc)
            latency_ms = (time.perf_counter_ns() - start) / 1_000_000
            event = LLMEvent(
                run_id=self._run_id,
                project=self._project,
                provider="openai",
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
            provider="openai",
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
        return getattr(self._completions, name)


class _TrackedChat:
    """Proxy for client.chat that exposes tracked completions."""

    def __init__(self, chat: Any, project: str, tags: dict[str, str],
                 run_id: str) -> None:
        self._chat = chat
        self.completions = _TrackedCompletions(
            chat.completions, project, tags, run_id
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _TrackedOpenAI:
    """Proxy around an OpenAI client that tracks all LLM calls."""

    def __init__(self, client: Any, project: str, tags: dict[str, str],
                 run_id: str) -> None:
        self._client = client
        self.chat = _TrackedChat(client.chat, project, tags, run_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def track_openai(
    client: Any,
    *,
    project: str = "default",
    tags: dict[str, str] | None = None,
    run_id: str = "",
) -> Any:
    """Wrap an OpenAI client to automatically track costs.

    Args:
        client: An openai.OpenAI() instance.
        project: Project name for grouping events.
        tags: Optional tags for filtering/grouping.
        run_id: Optional run ID for grouping multiple calls.

    Returns:
        A wrapped client that behaves identically but tracks all LLM calls.

    Example:
        import openai
        from aimeter import track_openai

        client = track_openai(openai.OpenAI(), project="my-agent")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        # Cost and tokens are automatically tracked.
    """
    return _TrackedOpenAI(client, project, tags or {}, run_id)
