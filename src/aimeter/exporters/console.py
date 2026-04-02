"""Console exporter — writes JSON-lines to stderr.

Uses stderr to avoid mixing with agent stdout output.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aimeter.types import LLMEvent


class ConsoleExporter:
    """Writes events as JSON-lines to stderr."""

    def export(self, events: list[LLMEvent]) -> None:
        for event in events:
            line = json.dumps(event.to_dict(), separators=(",", ":"))
            sys.stderr.write(line + "\n")

    def shutdown(self) -> None:
        pass
