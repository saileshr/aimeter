# Contributing to AgentMeter

We're building the financial infrastructure for AI agents. Every contribution matters — from fixing a typo to adding a new framework adapter.

## Getting Started

```bash
git clone https://github.com/saileshr/agentmeter-sdk.git
cd agentmeter-sdk
pip install -e ".[dev]"
pytest
```

That's it. Zero external services needed.

## Running Tests

```bash
pytest                          # run all tests
pytest tests/test_cost.py -v    # run a specific test file
pytest -k "test_openai"         # run tests matching a pattern
ruff check src/ tests/          # lint
```

All tests use mocks — no API keys required. Tests must pass before merging.

## Project Structure

```
src/agentmeter/
├── types.py          # Core data model (LLMEvent, TokenUsage, etc.)
├── cost.py           # Cost registry and calculation
├── config.py         # Configuration
├── tracker.py        # Core engine
├── outcome.py        # Outcome attribution
├── report.py         # Terminal report formatter
├── adapters/         # Framework adapters (one file per framework)
│   ├── openai.py
│   ├── anthropic.py
│   └── generic.py
└── exporters/        # Event exporters (one file per destination)
    ├── console.py
    └── memory.py
```

## Good First Issues

These are great places to start:

### Add a framework adapter

Each adapter is ~20 lines of extraction logic. The pattern:

1. Wrap the client/framework entry point
2. Call the underlying method
3. Extract: `model`, `tokens` (input/output/cached), `tool_calls` (names only)
4. Create an `LLMEvent` and call `tracker.record(event)`

Look at `src/agentmeter/adapters/openai.py` for the reference implementation.

**Wanted adapters:**
- LangChain (callback handler pattern)
- CrewAI
- AutoGen
- Cohere
- Mistral
- LiteLLM

### Add an exporter

Exporters implement a simple protocol: `export(events: list[LLMEvent])` and `shutdown()`.

**Wanted exporters:**
- File exporter (JSON-lines to disk)
- HTTP exporter (POST events to an endpoint)
- OpenTelemetry exporter (emit as OTEL spans)

### Update model pricing

Edit the `_BUILTIN_PRICING_RAW` dict in `src/agentmeter/cost.py`. Prices are per 1,000 tokens. Include the source (provider pricing page URL) in your PR description.

## Code Style

- Python 3.10+ (use `X | Y` union syntax, `slots=True` on dataclasses)
- Lint with `ruff` (config in `pyproject.toml`)
- No external dependencies in core — only stdlib
- Framework SDKs are optional extras (`pip install agentmeter[openai]`)
- Use `from __future__ import annotations` in every file
- Privacy by default — never capture message content

## Writing Tests

- Every new feature needs tests
- Use `MemoryExporter` to capture events in tests
- Call `reset()` in `teardown_method` to clean up the global tracker
- Mock external SDK objects (don't require real API keys)
- See `tests/adapters/test_openai.py` for the mock pattern

## Pull Request Process

1. Fork the repo and create a branch
2. Make your changes
3. Run `pytest` and `ruff check src/ tests/`
4. Submit a PR with a clear description of what and why

We aim to review PRs within 48 hours.

## Design Principles

- **Zero dependencies in core** — the SDK must install with no extras
- **Privacy by default** — track costs, not content
- **Thin adapters** — 20 lines of extraction, not 200 lines of proxy. Easy to maintain when SDKs change.
- **Boring technology** — stdlib, dataclasses, simple functions. No magic.
- **Works offline** — everything runs locally without any cloud service
