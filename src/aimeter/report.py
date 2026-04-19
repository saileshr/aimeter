"""Screenshot-ready terminal report for AIMeter data.

Renders a rich, formatted cost comparison report to the terminal
using only stdlib (no rich/click dependency).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import Any

from aimeter.performance import compute_performance
from aimeter.types import LLMEvent

# ANSI color codes
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_BLUE = "\033[94m"
_CYAN = "\033[96m"
_WHITE = "\033[97m"
_RESET = "\033[0m"
_BG_DARK = "\033[48;5;235m"

_BAR_CHAR = "█"
_BAR_HALF = "▌"


def _fmt_cost(cost: float) -> str:
    """Format a cost value for display."""
    if cost < 0.001:
        return f"${cost:.6f}"
    if cost < 0.01:
        return f"${cost:.5f}"
    if cost < 1.0:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def _fmt_tokens(tokens: int) -> str:
    """Format token count with commas."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    return str(tokens)


def _bar(value: float, max_value: float, width: int = 30) -> str:
    """Render a horizontal bar."""
    if max_value == 0:
        return ""
    ratio = min(value / max_value, 1.0)
    full_blocks = int(ratio * width)
    remainder = (ratio * width) - full_blocks
    bar = _BAR_CHAR * full_blocks
    if remainder >= 0.5:
        bar += _BAR_HALF
    return bar


def _color_for_rank(rank: int, total: int) -> str:
    """Color based on cost rank (cheapest=green, most expensive=red)."""
    if total <= 1:
        return _WHITE
    if rank == 0:
        return _GREEN
    if rank == total - 1:
        return _RED
    return _YELLOW


def _print_performance(events: list[LLMEvent], out: Any) -> None:
    """Render a Performance section (global + per-provider)."""
    perf = compute_performance(events)
    g = perf.get("global")
    if g is None:
        return

    out.write(f"  {_BOLD}{_WHITE}Performance{_RESET}\n")
    out.write(f"  {_DIM}{'─' * 64}{_RESET}\n")

    lat = g["latency_ms"]
    out.write(
        f"  {_DIM}Latency (ms){_RESET}  "
        f"p50 {_BOLD}{lat['p50']:.0f}{_RESET}  "
        f"p95 {_BOLD}{lat['p95']:.0f}{_RESET}  "
        f"p99 {_BOLD}{lat['p99']:.0f}{_RESET}  "
        f"{_DIM}(min {lat['min']:.0f} / mean {lat['mean']:.0f} / max {lat['max']:.0f}){_RESET}\n"
    )

    tp = g["throughput"]
    rps = f"{tp['requests_per_sec']:.2f}" if tp["requests_per_sec"] is not None else "n/a"
    tps = f"{tp['output_tokens_per_sec']:.1f}" if tp["output_tokens_per_sec"] is not None else "n/a"
    out.write(
        f"  {_DIM}Throughput{_RESET}    "
        f"{_BOLD}{rps}{_RESET} req/s  "
        f"{_BOLD}{tps}{_RESET} out-tok/s  "
        f"{_DIM}(window {tp['window_seconds']:.1f}s){_RESET}\n"
    )

    err = perf["errors"]
    if err["count"]:
        out.write(
            f"  {_DIM}Errors{_RESET}        "
            f"{_RED}{err['count']}{_RESET} "
            f"{_DIM}({err['rate'] * 100:.1f}% of non-outcome events){_RESET}\n"
        )

    by_prov = perf["by_provider"]
    if len(by_prov) > 1:
        out.write("\n")
        out.write(f"  {_DIM}By provider{_RESET}\n")
        for name, s in sorted(by_prov.items()):
            plat = s["latency_ms"]
            out.write(
                f"    {name:<20} "
                f"p50 {plat['p50']:.0f}ms  p95 {plat['p95']:.0f}ms  "
                f"{_DIM}({s['count']} calls){_RESET}\n"
            )

    out.write("\n")


def _center(text: str, width: int) -> str:
    """Center text within width (accounting for ANSI codes)."""
    # Strip ANSI for length calculation
    stripped = text
    for code in [_BOLD, _DIM, _RED, _GREEN, _YELLOW, _BLUE, _CYAN, _WHITE, _RESET, _BG_DARK]:
        stripped = stripped.replace(code, "")
    pad = max(0, width - len(stripped))
    left = pad // 2
    right = pad - left
    return " " * left + text + " " * right


def print_report(
    events: list[LLMEvent],
    title: str = "AIMeter Cost Report",
    file: Any = None,
) -> None:
    """Print a screenshot-ready cost comparison report.

    Args:
        events: List of LLMEvents to analyze.
        title: Report title.
        file: Output file (defaults to sys.stdout).
    """
    out = file or sys.stdout

    if not events:
        out.write(f"{_DIM}No events to report.{_RESET}\n")
        return

    # Group events by model
    by_model: dict[str, list[LLMEvent]] = defaultdict(list)
    for e in events:
        if e.event_type == "outcome":
            continue
        label = f"{e.provider}/{e.model}" if e.provider else e.model
        by_model[label].append(e)

    if not by_model:
        out.write(f"{_DIM}No LLM events to report.{_RESET}\n")
        return

    # Calculate stats per model
    stats: list[dict[str, Any]] = []
    for model_label, model_events in sorted(by_model.items()):
        total_cost = sum(e.cost.total_cost_usd for e in model_events)
        total_input = sum(e.tokens.input_tokens for e in model_events)
        total_output = sum(e.tokens.output_tokens for e in model_events)
        total_tokens = sum(e.tokens.total_tokens for e in model_events)
        total_latency = sum(e.latency_ms for e in model_events)
        avg_latency = total_latency / len(model_events) if model_events else 0
        avg_cost = total_cost / len(model_events) if model_events else 0

        stats.append({
            "model": model_label,
            "calls": len(model_events),
            "total_cost": total_cost,
            "avg_cost": avg_cost,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_tokens,
            "avg_latency_ms": avg_latency,
        })

    # Sort by total cost (most expensive first)
    stats.sort(key=lambda s: s["total_cost"], reverse=True)
    max_cost = stats[0]["total_cost"] if stats else 0
    cheapest_cost = stats[-1]["total_cost"] if stats else 0

    # Calculate multipliers relative to cheapest
    for s in stats:
        if cheapest_cost > 0:
            s["multiplier"] = s["total_cost"] / cheapest_cost
        else:
            s["multiplier"] = 0

    # --- Render ---

    width = 72
    out.write("\n")
    out.write(f"{_BOLD}{_CYAN}{'─' * width}{_RESET}\n")
    out.write(f"{_BOLD}{_CYAN}{_center(title, width)}{_RESET}\n")
    out.write(f"{_BOLD}{_CYAN}{'─' * width}{_RESET}\n")
    out.write("\n")

    # Summary line
    total_events = sum(s["calls"] for s in stats)
    total_spend = sum(s["total_cost"] for s in stats)
    total_tok = sum(s["total_tokens"] for s in stats)
    out.write(
        f"  {_DIM}Total:{_RESET} {_BOLD}{total_events}{_RESET} calls  "
        f"{_DIM}|{_RESET}  {_BOLD}{_fmt_tokens(total_tok)}{_RESET} tokens  "
        f"{_DIM}|{_RESET}  {_BOLD}{_fmt_cost(total_spend)}{_RESET} total spend\n"
    )
    out.write("\n")

    # Cost comparison table header
    out.write(f"  {_BOLD}{_WHITE}{'Model':<32} {'Calls':>5}  {'Total Cost':>11}  "
              f"{'Avg/Call':>11}  {'vs Cheapest':>11}{_RESET}\n")
    out.write(f"  {_DIM}{'─' * 32} {'─' * 5}  {'─' * 11}  {'─' * 11}  {'─' * 11}{_RESET}\n")

    for i, s in enumerate(stats):
        rank = len(stats) - 1 - i  # 0 = most expensive
        color = _color_for_rank(rank, len(stats))

        multiplier_str = ""
        if s["multiplier"] > 1.05:
            multiplier_str = f"{_RED}{s['multiplier']:.1f}x{_RESET}"
        elif s["multiplier"] < 0.95 or len(stats) == 1:
            multiplier_str = f"{_GREEN}baseline{_RESET}"
        else:
            multiplier_str = f"{_YELLOW}{s['multiplier']:.1f}x{_RESET}"

        out.write(
            f"  {color}{s['model']:<32}{_RESET} "
            f"{s['calls']:>5}  "
            f"{color}{_fmt_cost(s['total_cost']):>11}{_RESET}  "
            f"{_fmt_cost(s['avg_cost']):>11}  "
            f"{multiplier_str:>20}\n"
        )

    out.write("\n")

    # Cost bar chart
    out.write(f"  {_BOLD}{_WHITE}Cost Distribution{_RESET}\n")
    out.write(f"  {_DIM}{'─' * 64}{_RESET}\n")

    for i, s in enumerate(stats):
        rank = len(stats) - 1 - i
        color = _color_for_rank(rank, len(stats))
        bar = _bar(s["total_cost"], max_cost, width=30)
        model_short = s["model"]
        if len(model_short) > 28:
            model_short = model_short[:25] + "..."
        out.write(
            f"  {model_short:<28} {color}{bar}{_RESET} {_fmt_cost(s['total_cost'])}\n"
        )

    out.write("\n")

    # Latency comparison
    out.write(f"  {_BOLD}{_WHITE}Avg Latency per Call{_RESET}\n")
    out.write(f"  {_DIM}{'─' * 64}{_RESET}\n")

    max_latency = max(s["avg_latency_ms"] for s in stats) if stats else 0
    for s in stats:
        bar = _bar(s["avg_latency_ms"], max_latency, width=30)
        model_short = s["model"]
        if len(model_short) > 28:
            model_short = model_short[:25] + "..."
        out.write(
            f"  {model_short:<28} {_BLUE}{bar}{_RESET} {s['avg_latency_ms']:.0f}ms\n"
        )

    out.write("\n")

    _print_performance(events, out)

    # Token breakdown
    out.write(f"  {_BOLD}{_WHITE}Token Breakdown{_RESET}\n")
    out.write(f"  {_DIM}{'─' * 64}{_RESET}\n")
    out.write(f"  {_DIM}{'Model':<32} {'Input':>8}  {'Output':>8}  {'Total':>8}{_RESET}\n")

    for s in stats:
        model_short = s["model"]
        if len(model_short) > 30:
            model_short = model_short[:27] + "..."
        out.write(
            f"  {model_short:<32} "
            f"{_fmt_tokens(s['input_tokens']):>8}  "
            f"{_fmt_tokens(s['output_tokens']):>8}  "
            f"{_BOLD}{_fmt_tokens(s['total_tokens']):>8}{_RESET}\n"
        )

    out.write("\n")

    # Key insight
    if len(stats) >= 2 and cheapest_cost > 0:
        most_expensive = stats[0]
        cheapest = stats[-1]
        multiplier = most_expensive["total_cost"] / cheapest_cost

        out.write(f"  {_BOLD}{_YELLOW}⚡ Key Insight{_RESET}\n")
        out.write(
            f"  {most_expensive['model']} costs "
            f"{_BOLD}{_RED}{multiplier:.1f}x{_RESET} more than "
            f"{cheapest['model']} for the same tasks.\n"
        )

        monthly_diff = (most_expensive["avg_cost"] - cheapest["avg_cost"]) * 1000 * 30
        if monthly_diff > 0:
            out.write(
                f"  At 1,000 calls/day, switching saves "
                f"{_BOLD}{_GREEN}{_fmt_cost(monthly_diff)}/month{_RESET}.\n"
            )

    out.write(f"\n{_BOLD}{_CYAN}{'─' * width}{_RESET}\n")
    out.write(f"{_DIM}  Powered by AIMeter — aimeter.ai{_RESET}\n\n")
