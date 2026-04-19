"""Core data model for AIMeter.

All types are plain dataclasses — no external dependencies.
Privacy-by-default: no message content fields exist.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cached_tokens


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Cost in USD for a single LLM call."""

    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    cached_cost_usd: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd + self.cached_cost_usd


@dataclass(slots=True)
class LLMEvent:
    """A single tracked LLM call.

    This is the fundamental unit of data in AIMeter.
    Captures cost metadata only — never conversation content.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    project: str = "default"
    provider: str = ""
    model: str = ""
    event_type: str = "llm.call"
    tokens: TokenUsage = field(default_factory=TokenUsage)
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    latency_ms: float = 0.0
    tool_calls: list[str] = field(default_factory=list)
    error: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON export."""
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "project": self.project,
            "provider": self.provider,
            "model": self.model,
            "event_type": self.event_type,
            "tokens": {
                "input_tokens": self.tokens.input_tokens,
                "output_tokens": self.tokens.output_tokens,
                "cached_tokens": self.tokens.cached_tokens,
                "total_tokens": self.tokens.total_tokens,
            },
            "cost": {
                "input_cost_usd": self.cost.input_cost_usd,
                "output_cost_usd": self.cost.output_cost_usd,
                "cached_cost_usd": self.cost.cached_cost_usd,
                "total_cost_usd": self.cost.total_cost_usd,
            },
            "latency_ms": self.latency_ms,
            "tool_calls": self.tool_calls,
            "error": self.error,
            "tags": self.tags,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @property
    def output_tokens_per_sec(self) -> float | None:
        """Output-token throughput for this call; None if undefined."""
        if self.latency_ms <= 0:
            return None
        if self.tokens.output_tokens <= 0:
            return None
        return self.tokens.output_tokens / (self.latency_ms / 1000.0)


@dataclass(frozen=True, slots=True)
class Outcome:
    """A business outcome linked to an agent run.

    Connects agent costs to business value — the key differentiator.
    """

    outcome_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    outcome: str = ""
    value_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
