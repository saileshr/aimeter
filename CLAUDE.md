# CLAUDE.md — AgentMeter Development Guide

## What is AgentMeter?

AgentMeter is open-source financial observability for AI agents. It tracks the real cost of running AI agents in production — per-agent, per-task, per-conversation — across any framework.

Think of it as **the missing cost layer for agentic AI**: not just token counting, but full economic visibility including tool calls, retries, latency, and cost-per-outcome attribution.

---

## Project Structure

```
agentmeter-sdk/
├── agentmeter/              # Core Python SDK
│   ├── __init__.py          # Public API exports
│   ├── tracker.py           # Core tracking engine
│   ├── cost_engine.py       # Token → dollar cost calculation
│   ├── cost_models.yaml     # LLM pricing registry (OpenAI, Anthropic, Google, etc.)
│   ├── events.py            # Event schema and serialization
│   ├── adapters/            # Framework-specific adapters
│   │   ├── langchain.py     # LangChain integration
│   │   ├── openai_adapter.py # OpenAI SDK wrapper
│   │   ├── anthropic.py     # Anthropic SDK wrapper
│   │   ├── crewai.py        # CrewAI integration
│   │   └── autogen.py       # AutoGen integration
│   ├── exporters/           # Where metered data goes
│   │   ├── console.py       # Pretty-print to terminal
│   │   ├── json_file.py     # JSON file output
│   │   └── http.py          # POST to AgentMeter server or custom endpoint
│   └── outcomes.py          # Outcome attribution (cost-per-outcome mapping)
├── server/                  # Optional ingest + dashboard backend
│   ├── api/                 # FastAPI endpoints
│   ├── storage/             # SQLite (local) / ClickHouse (production)
│   └── dashboard/           # React dashboard
├── examples/                # Working examples for each framework
├── tests/                   # Test suite
├── docs/                    # Documentation
└── pyproject.toml           # Package configuration
```

> **Note**: The actual file tree may have evolved since this was written. Run `find . -type f -name "*.py" | head -40` to see current state.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Agent Runtime                      │
│  (LangChain / CrewAI / AutoGen / Custom)             │
│                                                       │
│  ┌─────────────────────────────────────────────┐     │
│  │         AgentMeter SDK (lightweight)         │     │
│  │  - Wraps LLM calls, tool calls, agent steps │     │
│  │  - Captures: tokens, latency, cost, outcome │     │
│  │  - Async, non-blocking, <1ms overhead        │     │
│  └──────────────────┬──────────────────────────┘     │
└─────────────────────┼───────────────────────────────┘
                      │ events (batched, async)
                      ▼
┌─────────────────────────────────────────────────────┐
│              Ingest Layer                             │
│  - Schema validation & enrichment                    │
│  - Cost calculation engine (maps tokens → $)         │
│  - Outcome correlation engine                        │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│              Data Layer                               │
│  - SQLite (local dev) / ClickHouse (production)      │
│  - Time-series metrics & event store                 │
│  - Aggregation engine for dashboards                 │
│  - Cost model registry (pricing per model/provider)  │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│              Application Layer                        │
│  - Dashboard (React + Recharts)                      │
│  - REST API (FastAPI)                                │
│  - Alerting engine                                   │
│  - Reports & exports                                 │
└─────────────────────────────────────────────────────┘
```

---

## Design Principles

These are non-negotiable. Every PR should respect them.

1. **Zero-config start.** `pip install agentmeter` + 2 lines of code must produce useful output. No API keys, no server setup, no config files required for basic local usage.

2. **Never block the agent.** All telemetry is async and fire-and-forget. AgentMeter must add <1ms overhead to any agent invocation. If it can't send an event, it silently drops it. The agent's job is more important than our metrics.

3. **Privacy by default.** AgentMeter captures cost metadata (tokens, model, latency, tool names), NOT conversation content. Prompt/response capture is opt-in only and clearly documented.

4. **Framework-agnostic core.** The core tracker and cost engine know nothing about LangChain, OpenAI, or any specific framework. Adapters are thin wrappers that translate framework-specific events into AgentMeter's unified event schema.

5. **Outcome-aware.** This is what differentiates AgentMeter from token counters. We don't just track "this call used 1,500 tokens." We enable mapping costs to business outcomes: "this agent spent $0.47 to resolve this support ticket."

6. **Accuracy over estimation.** Cost calculations must use actual token counts from API responses, not estimates. The cost model registry must reflect real, current provider pricing.

---

## SDK Target API

This is the developer experience we're building toward. All code changes should move us closer to this:

```python
# LangChain — 2 lines to add
from agentmeter import track
from langchain.agents import AgentExecutor

agent = AgentExecutor(agent=my_agent, tools=my_tools)
tracked_agent = track(agent, project="customer-support", tags={"team": "cx"})

result = tracked_agent.invoke({"input": "Help me reset my password"})
# All costs, latency, tool calls now tracked automatically.
```

```python
# OpenAI direct
from agentmeter import track_openai
import openai

client = track_openai(openai.OpenAI(), project="sales-agent")
# Every completion, tool call, and cost is metered transparently.
```

```python
# Outcome attribution
from agentmeter import record_outcome

record_outcome(
    agent_run_id=result.run_id,
    outcome="ticket_resolved",
    value_usd=12.50,
    metadata={"ticket_id": "T-1234", "resolution_time_min": 3}
)
```

```python
# Local report (no server needed)
from agentmeter import report

report(last="7d")
# Prints a cost breakdown to the terminal: per-agent, per-model, per-project
```

---

## Cost Model Registry

The file `cost_models.yaml` contains current LLM pricing. It must be kept up to date.

```yaml
# Format:
provider:
  model-name:
    input_per_1k: <float>    # USD per 1K input tokens
    output_per_1k: <float>   # USD per 1K output tokens
    cached_input_per_1k: <float>  # optional, for providers that support prompt caching
```

When updating pricing:
- Always include the date of the change in the git commit message
- Link to the provider's pricing page as a source
- Do not remove old models — mark them as deprecated with a comment

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| SDK | Python 3.10+ | Primary SDK language. TypeScript SDK is a future goal. |
| API | FastAPI | Async, auto-generates OpenAPI docs |
| Local storage | SQLite | Zero-config local dev experience |
| Production storage | ClickHouse or TimescaleDB | For the hosted/self-hosted server |
| Dashboard | React + Recharts | Real-time, interactive |
| Package | PyPI (`agentmeter`) | Published via GitHub Actions |
| CI/CD | GitHub Actions | Lint, test, publish on tag |
| Testing | pytest + pytest-asyncio | All async code must have async tests |

---

## Development Setup

```bash
# Clone and install in dev mode
git clone https://github.com/<org>/agentmeter-sdk.git
cd agentmeter-sdk
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
ruff format .

# Run the example
python examples/langchain_basic.py
```

---

## How to Contribute

### Adding a new framework adapter

1. Create a new file in `agentmeter/adapters/`
2. Implement the adapter by wrapping the framework's LLM/tool calling interface
3. The adapter should emit events using `agentmeter.events.AgentEvent` — the unified schema
4. Add a working example in `examples/`
5. Add tests in `tests/adapters/`
6. Update the framework support matrix in README.md

The adapter should be a thin wrapper. All cost calculation, storage, and export logic lives in the core — adapters just capture raw events (model name, token counts, latency, tool calls).

### Updating the cost model registry

1. Edit `agentmeter/cost_models.yaml`
2. Include the provider's pricing page URL in your commit message
3. Run `pytest tests/test_cost_engine.py` to verify calculations still pass

### Working on the dashboard

The dashboard is in `server/dashboard/` and is a standard React app.

```bash
cd server/dashboard
npm install
npm run dev
```

### Commit conventions

- Use conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`
- Keep PRs focused — one feature or fix per PR
- Include tests for new functionality
- Update relevant examples if the API surface changes

---

## Roadmap

### MVP (current focus)
- Core tracking engine with LangChain and OpenAI adapters
- Local cost reporting (terminal + JSON export)
- Cost model registry with major providers
- Working examples and documentation

### V2 (next)
- Cost-per-outcome attribution engine
- FastAPI server for centralized collection
- React dashboard with real-time metrics
- Alerting for cost anomalies and runaway agents
- CrewAI, AutoGen, and Anthropic adapters

### V3 (future)
- Usage-based billing primitives for agent builders
- Multi-tenant metering
- Budget controls and spend limits per agent/project
- TypeScript SDK
- ClickHouse backend for production-scale deployments

---

## Key Files to Understand First

If you're new to the codebase, read these in order:

1. `agentmeter/events.py` — the unified event schema. Everything flows through this.
2. `agentmeter/cost_engine.py` — how tokens get converted to dollars.
3. `agentmeter/tracker.py` — the core tracking logic.
4. `agentmeter/adapters/langchain.py` — the most complete adapter, use as a reference.
5. `examples/langchain_basic.py` — the simplest end-to-end usage.

---

*AgentMeter is open-source under the Apache 2.0 license. Contributions welcome.*
