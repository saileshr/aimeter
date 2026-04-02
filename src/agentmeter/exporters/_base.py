"""Exporter protocol — the interface all exporters implement."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentmeter.types import LLMEvent


@runtime_checkable
class Exporter(Protocol):
    """Protocol for event exporters.

    Exporters receive batches of LLMEvents and write them somewhere
    (stderr, file, memory, HTTP, etc.).
    """

    def export(self, events: list[LLMEvent]) -> None:
        """Export a batch of events."""
        ...

    def shutdown(self) -> None:
        """Clean up resources. Called on process exit."""
        ...
