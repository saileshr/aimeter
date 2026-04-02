#!/usr/bin/env python3
"""AIMeter Model Comparison Demo

Runs the same set of realistic agent prompts across multiple LLM models
(OpenAI + Anthropic) and generates a screenshot-ready cost report.

This generates real data for blog posts and content pieces showing
the true cost differences between models.

Requirements:
    pip install aimeter[openai] anthropic

    Set environment variables:
        OPENAI_API_KEY=sk-...
        ANTHROPIC_API_KEY=sk-ant-...

Usage:
    python examples/model_comparison.py
"""

from __future__ import annotations

import os
import sys
import time

# --- Check API keys early ---

openai_key = os.environ.get("OPENAI_API_KEY", "")
anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

if not openai_key and not anthropic_key:
    print("Error: Set at least one of OPENAI_API_KEY or ANTHROPIC_API_KEY")
    sys.exit(1)

# --- Imports ---

from aimeter import MemoryExporter, configure, record_outcome
from aimeter.report import print_report

# Only import adapters for available keys
if openai_key:
    import openai
    from aimeter import track_openai

if anthropic_key:
    import anthropic
    from aimeter import track_anthropic

# --- Configuration ---

# Realistic agent prompts — the kind of tasks production agents handle
PROMPTS = [
    {
        "task": "Extract customer intent",
        "prompt": "Classify the following customer message into one of these categories: billing_issue, technical_support, feature_request, cancellation, general_inquiry. Message: 'I've been charged twice this month and I need a refund immediately. This is unacceptable.' Respond with just the category and a one-sentence explanation.",
        "outcome": "intent_classified",
        "value_usd": 0.50,
    },
    {
        "task": "Summarize support ticket",
        "prompt": "Summarize the following support ticket in 2-3 sentences for an engineering team handoff:\n\nTicket #4721: Customer reports that the export feature generates corrupted CSV files when the dataset exceeds 10,000 rows. They've tested with Chrome and Firefox on macOS. The issue started after last Tuesday's deployment. They need this resolved urgently as it's blocking their quarterly reporting. They've tried clearing cache and using incognito mode with no improvement.",
        "outcome": "ticket_summarized",
        "value_usd": 1.00,
    },
    {
        "task": "Generate API response",
        "prompt": "Generate a JSON response for a REST API endpoint GET /api/users/123/activity that returns the user's recent activity. Include realistic fields: activity_id, type (login/purchase/settings_change/support_ticket), timestamp, metadata. Return 5 activity items. Output valid JSON only.",
        "outcome": "api_response_generated",
        "value_usd": 0.25,
    },
    {
        "task": "Code review feedback",
        "prompt": "Review this Python function and provide 2-3 specific, actionable suggestions:\n\n```python\ndef process_orders(orders):\n    results = []\n    for order in orders:\n        if order['status'] == 'pending':\n            total = 0\n            for item in order['items']:\n                total = total + item['price'] * item['quantity']\n            if total > 100:\n                order['discount'] = total * 0.1\n                total = total - order['discount']\n            order['total'] = total\n            order['status'] = 'processed'\n            results.append(order)\n    return results\n```",
        "outcome": "code_reviewed",
        "value_usd": 2.00,
    },
    {
        "task": "Draft email response",
        "prompt": "Draft a professional but friendly email response to a customer who is asking for a feature that doesn't exist yet (dark mode). Acknowledge their request, explain it's on the roadmap but no ETA, and suggest a workaround (browser extension). Keep it under 100 words.",
        "outcome": "email_drafted",
        "value_usd": 0.75,
    },
    {
        "task": "Data extraction",
        "prompt": "Extract all company names, their roles, and any dollar amounts from this text:\n\n'In Q3 2025, Acme Corp invested $2.5M in their AI infrastructure, partnering with DataFlow Inc for pipeline management. Their competitor, NexGen Solutions, spent approximately $4.1M on similar technology through a deal with CloudBridge Systems. Meanwhile, TechStart LLC secured a $750K grant from the Innovation Fund to develop open-source alternatives.'",
        "outcome": "data_extracted",
        "value_usd": 1.50,
    },
    {
        "task": "SQL query generation",
        "prompt": "Write a SQL query for PostgreSQL that finds the top 10 customers by total spending in the last 90 days, including their name, email, total spend, number of orders, and average order value. Tables: customers(id, name, email, created_at), orders(id, customer_id, total_amount, status, created_at). Only include orders with status='completed'. Output the query only.",
        "outcome": "query_generated",
        "value_usd": 1.00,
    },
    {
        "task": "Sentiment analysis",
        "prompt": "Analyze the sentiment of each of these 5 product reviews and rate them from 1 (very negative) to 5 (very positive). Provide a one-word sentiment label for each.\n\n1. 'Absolutely love this product! Best purchase I've made all year.'\n2. 'It works okay I guess. Nothing special but gets the job done.'\n3. 'Terrible quality. Broke after two days. Want my money back.'\n4. 'Surprisingly good for the price. A few minor issues but overall solid.'\n5. 'DO NOT BUY. Scam product, nothing like the description.'",
        "outcome": "sentiment_analyzed",
        "value_usd": 0.30,
    },
    {
        "task": "Meeting notes summary",
        "prompt": "Generate structured meeting notes from this transcript excerpt:\n\n'So we agreed that the launch date moves to March 15th. Sarah will handle the marketing assets by the 10th, and Mike's team needs to finish the API integration by the 8th. We're cutting the advanced analytics feature from v1 — it'll go in v1.1. Budget is approved at $45K for the launch campaign. Next check-in is Thursday at 2pm. Oh and John mentioned we need legal review on the new terms of service before launch.'",
        "outcome": "notes_generated",
        "value_usd": 0.50,
    },
    {
        "task": "Error diagnosis",
        "prompt": "Diagnose the most likely cause of this error and suggest a fix:\n\n```\nTraceback (most recent call last):\n  File \"app/api/handlers.py\", line 142, in process_webhook\n    payload = json.loads(request.body)\n  File \"/usr/lib/python3.11/json/__init__.py\", line 346, in loads\n    return _default_decoder.decode(s)\n  File \"/usr/lib/python3.11/json/decoder.py\", line 337, in decode\n    obj, end = self.raw_decode(s, idx=_w(s, 0).end())\n  File \"/usr/lib/python3.11/json/decoder.py\", line 355, in raw_decode\n    raise JSONDecodeError(\"Expecting value\", s, err.value) from None\njson.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)\n```\n\nContext: This happens intermittently on a webhook endpoint that receives POST requests from Stripe.",
        "outcome": "error_diagnosed",
        "value_usd": 2.00,
    },
]

# Models to compare
OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"]
ANTHROPIC_MODELS = ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"]


def run_openai_model(model: str, mem: MemoryExporter) -> None:
    """Run all prompts through an OpenAI model."""
    client = track_openai(
        openai.OpenAI(),
        project="model-comparison",
        run_id=f"openai-{model}",
        tags={"demo": "model-comparison"},
    )

    print(f"  Running {model}...", end=" ", flush=True)
    for p in PROMPTS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Be concise."},
                    {"role": "user", "content": p["prompt"]},
                ],
                max_tokens=500,
            )
            # Record the business outcome
            record_outcome(
                run_id=f"openai-{model}",
                outcome=p["outcome"],
                value_usd=p["value_usd"],
                project="model-comparison",
                metadata={"task": p["task"], "model": model},
            )
        except Exception as e:
            print(f"\n    Warning: {model} failed on '{p['task']}': {e}")

    print("done")


def run_anthropic_model(model: str, mem: MemoryExporter) -> None:
    """Run all prompts through an Anthropic model."""
    client = track_anthropic(
        anthropic.Anthropic(),
        project="model-comparison",
        run_id=f"anthropic-{model}",
        tags={"demo": "model-comparison"},
    )

    print(f"  Running {model}...", end=" ", flush=True)
    for p in PROMPTS:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=500,
                system="You are a helpful AI assistant. Be concise.",
                messages=[
                    {"role": "user", "content": p["prompt"]},
                ],
            )
            record_outcome(
                run_id=f"anthropic-{model}",
                outcome=p["outcome"],
                value_usd=p["value_usd"],
                project="model-comparison",
                metadata={"task": p["task"], "model": model},
            )
        except Exception as e:
            print(f"\n    Warning: {model} failed on '{p['task']}': {e}")

    print("done")


def main() -> None:
    # Set up AIMeter with memory exporter
    mem = MemoryExporter()
    configure(project="model-comparison", exporters=[mem])

    print()
    print("=" * 60)
    print("  AIMeter Model Comparison Demo")
    print(f"  Running {len(PROMPTS)} realistic agent tasks across models")
    print("=" * 60)
    print()

    # Run OpenAI models
    if openai_key:
        print(f"OpenAI ({len(OPENAI_MODELS)} models):")
        for model in OPENAI_MODELS:
            run_openai_model(model, mem)
        print()
    else:
        print("Skipping OpenAI (no API key)")

    # Run Anthropic models
    if anthropic_key:
        print(f"Anthropic ({len(ANTHROPIC_MODELS)} models):")
        for model in ANTHROPIC_MODELS:
            run_anthropic_model(model, mem)
        print()
    else:
        print("Skipping Anthropic (no API key)")

    # Filter to only LLM events (not outcomes) for the report
    llm_events = [e for e in mem.events if e.event_type == "llm.call"]

    print(f"Tracked {len(llm_events)} LLM calls across {len(mem.events) - len(llm_events)} outcomes")

    # Print the screenshot-ready report
    print_report(
        llm_events,
        title="AIMeter — Model Cost Comparison",
    )

    # Print outcome attribution summary
    _print_outcome_summary(mem)


def _print_outcome_summary(mem: MemoryExporter) -> None:
    """Print cost-per-outcome breakdown."""
    from collections import defaultdict

    # Group LLM costs by run_id
    cost_by_run: dict[str, float] = defaultdict(float)
    for e in mem.events:
        if e.event_type == "llm.call" and e.run_id:
            cost_by_run[e.run_id] += e.cost.total_cost_usd

    # Group outcomes by run_id
    outcomes_by_run: dict[str, list] = defaultdict(list)
    for e in mem.events:
        if e.event_type == "outcome" and e.run_id:
            outcomes_by_run[e.run_id].append(e)

    if not cost_by_run:
        return

    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    print(f"  {BOLD}Cost-Per-Outcome Attribution{RESET}")
    print(f"  {DIM}{'─' * 64}{RESET}")
    print(f"  {DIM}{'Model Run':<36} {'Agent Cost':>10}  {'Outcomes':>8}  {'$/Outcome':>10}{RESET}")

    for run_id in sorted(cost_by_run.keys()):
        cost = cost_by_run[run_id]
        outcome_count = len(outcomes_by_run.get(run_id, []))
        cost_per = cost / outcome_count if outcome_count > 0 else 0

        # Clean up run_id for display
        display = run_id.replace("openai-", "OpenAI ").replace("anthropic-", "Anthropic ")

        print(
            f"  {display:<36} "
            f"${cost:>9.4f}  "
            f"{outcome_count:>8}  "
            f"{GREEN}${cost_per:>9.6f}{RESET}"
        )

    total_value = sum(
        e.metadata.get("value_usd", 0) or 0
        for e in mem.events
        if e.event_type == "outcome"
    )
    total_cost = sum(cost_by_run.values())
    roi = (total_value / total_cost) if total_cost > 0 else 0

    print()
    print(f"  {BOLD}Total agent cost:{RESET}  ${total_cost:.4f}")
    print(f"  {BOLD}Total outcome value:{RESET} ${total_value:.2f}")
    print(f"  {BOLD}ROI:{RESET} {GREEN}{roi:.0f}x{RESET} (${total_value:.2f} value / ${total_cost:.4f} cost)")
    print()
    print(f"{BOLD}{CYAN}{'─' * 72}{RESET}")
    print(f"{DIM}  Powered by AIMeter — aimeter.ai{RESET}")
    print()


if __name__ == "__main__":
    main()
