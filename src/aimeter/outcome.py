"""Outcome attribution — connect agent costs to business value.

This is the key differentiator: not just "how much did this cost?"
but "what was the business value per dollar spent?"

Usage:
    from aimeter import record_outcome

    record_outcome(
        run_id="run-123",
        outcome="ticket_resolved",
        value_usd=12.50,
        metadata={"ticket_id": "T-1234"},
    )
"""

from __future__ import annotations

from typing import Any

from aimeter.tracker import get_tracker
from aimeter.types import LLMEvent, Outcome


def record_outcome(
    *,
    run_id: str,
    outcome: str,
    value_usd: float | None = None,
    project: str = "default",
    tags: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Outcome:
    """Record a business outcome linked to an agent run.

    Outcomes are stored as special LLMEvents with event_type="outcome"
    so they flow through the same exporter pipeline.

    Args:
        run_id: The run ID to associate this outcome with.
        outcome: Name of the outcome (e.g., "ticket_resolved", "lead_qualified").
        value_usd: Estimated business value in USD (optional).
        project: Project name for grouping.
        tags: Optional tags.
        metadata: Arbitrary metadata (e.g., ticket_id, customer_id).

    Returns:
        The created Outcome object.
    """
    outcome_obj = Outcome(
        run_id=run_id,
        outcome=outcome,
        value_usd=value_usd,
        metadata=metadata or {},
    )

    # Record as a special event so it flows through exporters
    event = LLMEvent(
        event_id=outcome_obj.outcome_id,
        run_id=run_id,
        project=project,
        event_type="outcome",
        tags=tags or {},
        metadata={
            "outcome": outcome,
            "value_usd": value_usd,
            **(metadata or {}),
        },
    )

    get_tracker().record(event)
    return outcome_obj
