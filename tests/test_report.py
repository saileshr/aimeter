"""Tests for aimeter.report — terminal report rendering."""

import io

from aimeter.report import print_report
from aimeter.types import LLMEvent, TokenUsage


class TestPrintReportPerformance:
    def test_report_renders_performance_section(self):
        events = [
            LLMEvent(
                provider="openai",
                model="gpt-4o",
                tokens=TokenUsage(input_tokens=100, output_tokens=50),
                latency_ms=200.0,
            ),
            LLMEvent(
                provider="anthropic",
                model="claude-sonnet-4-6",
                tokens=TokenUsage(input_tokens=120, output_tokens=60),
                latency_ms=350.0,
            ),
            LLMEvent(
                provider="openai",
                model="gpt-4o",
                tokens=TokenUsage(input_tokens=80, output_tokens=40),
                latency_ms=250.0,
            ),
        ]
        buf = io.StringIO()
        print_report(events, file=buf)
        text = buf.getvalue()
        assert "Performance" in text
        assert "p50" in text
        assert "req/s" in text

    def test_report_skips_performance_when_all_errors(self):
        events = [
            LLMEvent(
                provider="openai",
                model="gpt-4o",
                tokens=TokenUsage(output_tokens=50),
                latency_ms=200.0,
                error="boom",
            ),
        ]
        buf = io.StringIO()
        print_report(events, file=buf)
        text = buf.getvalue()
        assert "Performance" not in text

    def test_report_shows_errors_line(self):
        events = [
            LLMEvent(
                provider="openai",
                model="gpt-4o",
                tokens=TokenUsage(output_tokens=50),
                latency_ms=200.0,
            ),
            LLMEvent(
                provider="openai",
                model="gpt-4o",
                tokens=TokenUsage(output_tokens=50),
                latency_ms=400.0,
                error="boom",
            ),
        ]
        buf = io.StringIO()
        print_report(events, file=buf)
        text = buf.getvalue()
        assert "Errors" in text
