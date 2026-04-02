"""Tests for the console exporter."""

import json

from agentmeter.exporters.console import ConsoleExporter
from agentmeter.types import CostBreakdown, LLMEvent, TokenUsage


class TestConsoleExporter:
    def test_writes_to_stderr(self, capsys):
        exporter = ConsoleExporter()
        event = LLMEvent(
            provider="openai",
            model="gpt-4o",
            tokens=TokenUsage(input_tokens=100, output_tokens=50),
            cost=CostBreakdown(input_cost_usd=0.00025, output_cost_usd=0.0005),
        )
        exporter.export([event])
        captured = capsys.readouterr()
        assert captured.out == ""  # nothing to stdout
        assert captured.err  # something to stderr
        parsed = json.loads(captured.err.strip())
        assert parsed["provider"] == "openai"
        assert parsed["model"] == "gpt-4o"
        assert parsed["tokens"]["input_tokens"] == 100

    def test_multiple_events(self, capsys):
        exporter = ConsoleExporter()
        events = [LLMEvent(model=f"model-{i}") for i in range(3)]
        exporter.export(events)
        captured = capsys.readouterr()
        lines = captured.err.strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines):
            parsed = json.loads(line)
            assert parsed["model"] == f"model-{i}"

    def test_empty_batch(self, capsys):
        exporter = ConsoleExporter()
        exporter.export([])
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_shutdown_is_noop(self):
        exporter = ConsoleExporter()
        exporter.shutdown()  # should not raise
