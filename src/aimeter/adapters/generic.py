"""Generic adapter — manual instrumentation for any LLM call.

Use this when your framework doesn't have a dedicated adapter.

Usage:
    from aimeter.adapters.generic import track_llm_call

    with track_llm_call(provider="openai", model="gpt-4o") as call:
        response = my_custom_llm_call(...)
        call.input_tokens = response.usage.input_tokens
        call.output_tokens = response.usage.output_tokens
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from aimeter.tracker import get_tracker
from aimeter.types import LLMEvent, TokenUsage


@dataclass
class LLMCallCapture:
    """Mutable capture object yielded by track_llm_call.

    Set token counts and other metadata inside the context manager.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    tool_calls: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@contextmanager
def track_llm_call(
    *,
    provider: str,
    model: str,
    project: str = "default",
    run_id: str = "",
    tags: dict[str, str] | None = None,
) -> Iterator[LLMCallCapture]:
    """Context manager for tracking any LLM call.

    Measures latency automatically. Set token counts on the yielded
    capture object to enable cost calculation.

    Args:
        provider: LLM provider name (e.g., "openai", "anthropic").
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4-20250514").
        project: Project name for grouping events.
        run_id: Optional run ID for grouping multiple calls.
        tags: Optional tags for filtering/grouping.

    Yields:
        LLMCallCapture: Set .input_tokens, .output_tokens, etc. on this object.
    """
    capture = LLMCallCapture()
    start = time.perf_counter_ns()

    try:
        yield capture
    except Exception as exc:
        capture.error = str(exc)
        raise
    finally:
        latency_ms = (time.perf_counter_ns() - start) / 1_000_000

        event = LLMEvent(
            run_id=run_id,
            project=project,
            provider=provider,
            model=model,
            event_type="llm.call",
            tokens=TokenUsage(
                input_tokens=capture.input_tokens,
                output_tokens=capture.output_tokens,
                cached_tokens=capture.cached_tokens,
            ),
            latency_ms=latency_ms,
            tool_calls=capture.tool_calls,
            error=capture.error,
            tags=tags or {},
            metadata=dict(capture.metadata),
        )

        tracker = get_tracker()
        tracker.record(event)
