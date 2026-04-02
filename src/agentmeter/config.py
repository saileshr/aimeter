"""Configuration for AgentMeter.

Resolution order: explicit kwargs > environment variables > defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentmeter.exporters._base import Exporter


@dataclass
class AgentMeterConfig:
    """Configuration for the AgentMeter tracker.

    Attributes:
        project: Default project name for events.
        tags: Default tags applied to all events.
        exporters: List of exporters to send events to. Defaults to ConsoleExporter.
        enabled: Kill switch — set to False to disable all tracking.
        debug: Enable debug logging for troubleshooting.
    """

    project: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    exporters: list[Exporter] = field(default_factory=list)
    enabled: bool = True
    debug: bool = False

    def __post_init__(self) -> None:
        # Environment variable overrides
        if not self.project:
            self.project = os.environ.get("AGENTMETER_PROJECT", "default")
        if os.environ.get("AGENTMETER_ENABLED", "").lower() == "false":
            self.enabled = False
        if os.environ.get("AGENTMETER_DEBUG", "").lower() == "true":
            self.debug = True

        # Default exporter if none specified
        if not self.exporters:
            export_type = os.environ.get("AGENTMETER_EXPORT", "console").lower()
            if export_type == "memory":
                from agentmeter.exporters.memory import MemoryExporter

                self.exporters = [MemoryExporter()]
            else:
                from agentmeter.exporters.console import ConsoleExporter

                self.exporters = [ConsoleExporter()]
