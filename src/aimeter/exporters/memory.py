"""Memory exporter — stores events in-memory for testing and local dev."""

from __future__ import annotations

from aimeter.types import LLMEvent


class MemoryExporter:
    """Stores events in a list. Useful for testing and local inspection."""

    def __init__(self) -> None:
        self.events: list[LLMEvent] = []

    def export(self, events: list[LLMEvent]) -> None:
        self.events.extend(events)

    def shutdown(self) -> None:
        pass

    def clear(self) -> None:
        """Clear all stored events."""
        self.events.clear()

    @property
    def total_cost(self) -> float:
        """Sum of all event costs."""
        return sum(e.cost.total_cost_usd for e in self.events)

    @property
    def total_tokens(self) -> int:
        """Sum of all event tokens."""
        return sum(e.tokens.total_tokens for e in self.events)

    def events_by_run(self, run_id: str) -> list[LLMEvent]:
        """Filter events by run_id."""
        return [e for e in self.events if e.run_id == run_id]

    def summary(self) -> dict[str, object]:
        """Quick summary of all tracked events."""
        return {
            "event_count": len(self.events),
            "total_cost_usd": self.total_cost,
            "total_tokens": self.total_tokens,
            "models_used": list({e.model for e in self.events if e.model}),
            "providers_used": list({e.provider for e in self.events if e.provider}),
        }
