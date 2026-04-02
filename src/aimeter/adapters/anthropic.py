"""Anthropic adapter — thin wrapper for cost tracking.

Wraps an Anthropic client to automatically track token usage and costs
for all message creation calls.

Usage:
    from aimeter import track_anthropic
    import anthropic

    client = track_anthropic(anthropic.Anthropic(), project="my-agent")
    response = client.messages.create(model="claude-sonnet-4-20250514", ...)
    # Tokens, cost, and latency are automatically tracked.
"""

from __future__ import annotations

import time
from typing import Any

from aimeter.tracker import get_tracker
from aimeter.types import LLMEvent, TokenUsage


def _extract_usage(response: Any) -> TokenUsage:
    """Extract token usage from an Anthropic response object."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()

    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cached_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
    )


def _extract_tool_calls(response: Any) -> list[str]:
    """Extract tool use names from an Anthropic response (names only, for privacy)."""
    try:
        content = getattr(response, "content", [])
        return [
            getattr(block, "name", "unknown")
            for block in content
            if getattr(block, "type", None) == "tool_use"
        ]
    except (TypeError, AttributeError):
        return []


def _extract_model(response: Any, kwargs: dict[str, Any]) -> str:
    """Get model name from response (preferred) or request kwargs."""
    return getattr(response, "model", None) or kwargs.get("model", "")


class _TrackedMessages:
    """Proxy for client.messages that tracks create() calls."""

    def __init__(self, messages: Any, project: str, tags: dict[str, str],
                 run_id: str) -> None:
        self._messages = messages
        self._project = project
        self._tags = tags
        self._run_id = run_id

    def create(self, **kwargs: Any) -> Any:
        """Call messages.create and record the event."""
        start = time.perf_counter_ns()
        error = None

        try:
            response = self._messages.create(**kwargs)
        except Exception as exc:
            error = str(exc)
            latency_ms = (time.perf_counter_ns() - start) / 1_000_000
            event = LLMEvent(
                run_id=self._run_id,
                project=self._project,
                provider="anthropic",
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
            provider="anthropic",
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
        return getattr(self._messages, name)


class _TrackedAnthropic:
    """Proxy around an Anthropic client that tracks all LLM calls."""

    def __init__(self, client: Any, project: str, tags: dict[str, str],
                 run_id: str) -> None:
        self._client = client
        self.messages = _TrackedMessages(client.messages, project, tags, run_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def track_anthropic(
    client: Any,
    *,
    project: str = "default",
    tags: dict[str, str] | None = None,
    run_id: str = "",
) -> Any:
    """Wrap an Anthropic client to automatically track costs.

    Args:
        client: An anthropic.Anthropic() instance.
        project: Project name for grouping events.
        tags: Optional tags for filtering/grouping.
        run_id: Optional run ID for grouping multiple calls.

    Returns:
        A wrapped client that behaves identically but tracks all LLM calls.

    Example:
        import anthropic
        from aimeter import track_anthropic

        client = track_anthropic(anthropic.Anthropic(), project="my-agent")
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )
        # Cost and tokens are automatically tracked.
    """
    return _TrackedAnthropic(client, project, tags or {}, run_id)
