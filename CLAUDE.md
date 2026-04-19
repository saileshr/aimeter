# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What AIMeter is

A Python SDK that wraps LLM SDK clients (OpenAI, Anthropic, …) to record every call's cost, tokens, latency, and tool-call names into a process-wide tracker, then fan them out to exporters. Zero dependencies in the core; framework SDKs are optional extras.

See `README.md` for the user-facing pitch and `CONTRIBUTING.md` for contributor workflow — don't duplicate them here.

## Commands

```bash
pip install -e ".[dev]"         # dev install (src layout — editable required for imports to work)
pytest                          # full suite
pytest tests/test_cost.py -v    # single file
pytest -k "test_openai"         # by name pattern
ruff check src/ tests/          # lint
ruff format src/ tests/         # format
```

All tests mock the underlying SDKs — no API keys required, no network.

## Architecture

Layout is **src-style**: package is `src/aimeter/`, tests import as `from aimeter import ...`.

Event flow for every tracked LLM call:

```
user code → adapter (wraps SDK client)
          → builds LLMEvent (model, tokens, tool_call names, latency)
          → tracker.record(event)
              → CostRegistry enriches with USD cost
              → fans out to each configured Exporter
```

Key modules and their single responsibility:

- `types.py` — core dataclasses: `LLMEvent`, `TokenUsage`, `ToolCall`, `Outcome`. Everything else flows through these. Use `slots=True` and `from __future__ import annotations`.
- `cost.py` — `CostRegistry` + `_BUILTIN_PRICING_RAW` dict (prices per 1K tokens). **Pricing is inline Python, not YAML** — edit the dict directly to update/add models.
- `config.py` — `configure(project=..., exporters=[...], tags=...)` sets process-wide state. Also reads `AIMETER_*` env vars.
- `tracker.py` — global singleton tracker. `record()` enriches with cost and dispatches to exporters. `reset()` is the test hook — call it in `teardown_method` to clear state between tests.
- `outcome.py` — `record_outcome(run_id=..., outcome=..., value_usd=..., metadata=...)` links a prior event to a business outcome.
- `report.py` — terminal summary formatter. Reads events, delegates perf aggregation to `performance.compute_performance`.
- `performance.py` — pure aggregator over a list of events: latency percentiles (nearest-rank), throughput over first→last timestamp span, error rate; broken down by model/provider/project/tag key. Stdlib-only. Consumed by `MemoryExporter.summary()` (which returns the dict under a `"performance"` key) and by `report.py`.
- `adapters/` — one file per upstream SDK. Each is a **thin (~20 line) extraction wrapper**: wrap the client's entry point, call through, extract `model` / token counts / tool-call names, build `LLMEvent`, pass to tracker. No proxying, no feature recreation. `openai.py` is the reference pattern; `generic.py` provides the `track_llm_call` context manager for SDKs without a dedicated adapter.
- `exporters/` — implement `export(events: list[LLMEvent])` and `shutdown()`. `_base.py` has the protocol; `console.py` writes to stderr, `memory.py` keeps events in a list for tests and local reports.

## Invariants (don't break these)

- **Zero deps in core.** `src/aimeter/` (excluding `adapters/`) must import only stdlib. Adapter modules may import their target SDK, but must be gated behind the extras in `pyproject.toml` (`aimeter[openai]`, etc.) and must not be imported eagerly from `__init__.py`.
- **Privacy.** Never capture message content, prompt text, or tool-call *arguments*. Tool/function *names* are OK. This is a product-level promise, surfaced in the README.
- **Thin adapters.** If an adapter starts recreating SDK features, it's wrong — extract and emit, nothing more.
- **Non-blocking.** Tracking must never raise into user code. Swallow and log (behind `AIMETER_DEBUG`) rather than bubble up.
- **`from __future__ import annotations`** in every file; 3.10+ union syntax (`X | Y`); `slots=True` on dataclasses.

## Repo wiring

- Package name on PyPI: `aimeter`. GitHub repo: `saileshr/aimeter`.
- `ROADMAP.md` is the source of truth for what's planned vs. shipped. Current shipped adapters: OpenAI, Anthropic, generic. LangChain / CrewAI / AutoGen are planned, not implemented — don't reference them as if they exist.
