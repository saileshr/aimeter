"""Core tracker — the central coordination point for AgentMeter.

Records LLM events, enriches them with cost data, and sends them to exporters.
Synchronous and simple — no background threads, no batching.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agentmeter.config import AgentMeterConfig
from agentmeter.cost import CostRegistry

if TYPE_CHECKING:
    from agentmeter.types import LLMEvent

logger = logging.getLogger("agentmeter")

# Module-level singleton
_tracker: Tracker | None = None


class Tracker:
    """Records LLM events, enriches with cost, exports immediately."""

    def __init__(self, config: AgentMeterConfig | None = None) -> None:
        self.config = config or AgentMeterConfig()
        self.cost_registry = CostRegistry()
        self._exporters = self.config.exporters

    def record(self, event: LLMEvent) -> LLMEvent:
        """Record an LLM event.

        Enriches the event with cost data if not already set,
        then exports to all configured exporters.
        Returns the enriched event.
        """
        if not self.config.enabled:
            return event

        # Apply default project and tags if not set on the event
        if event.project == "default" and self.config.project != "default":
            event.project = self.config.project
        if self.config.tags and not event.tags:
            event.tags = dict(self.config.tags)

        # Enrich with cost if not already calculated
        if event.cost.total_cost_usd == 0.0 and event.tokens.total_tokens > 0:
            event.cost = self.cost_registry.calculate(
                event.provider, event.model, event.tokens
            )

        # Export
        for exporter in self._exporters:
            try:
                exporter.export([event])
            except Exception:
                logger.exception("agentmeter: exporter %s failed", type(exporter).__name__)

        return event

    def shutdown(self) -> None:
        """Shut down all exporters."""
        for exporter in self._exporters:
            try:
                exporter.shutdown()
            except Exception:
                logger.exception(
                    "agentmeter: exporter %s shutdown failed", type(exporter).__name__
                )


def get_tracker(**kwargs: object) -> Tracker:
    """Get or create the global tracker singleton.

    Keyword arguments are passed to AgentMeterConfig if creating a new tracker.
    """
    global _tracker
    if _tracker is None:
        config = AgentMeterConfig(**kwargs)  # type: ignore[arg-type]
        _tracker = Tracker(config)
    return _tracker


def configure(**kwargs: object) -> Tracker:
    """Configure (or reconfigure) the global tracker.

    Always creates a new tracker, replacing any existing one.
    """
    global _tracker
    if _tracker is not None:
        _tracker.shutdown()
    config = AgentMeterConfig(**kwargs)  # type: ignore[arg-type]
    _tracker = Tracker(config)
    return _tracker


def reset() -> None:
    """Reset the global tracker. Primarily for testing."""
    global _tracker
    if _tracker is not None:
        _tracker.shutdown()
    _tracker = None
