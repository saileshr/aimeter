<p align="center">
  <h1 align="center">AgentMeter</h1>
  <p align="center"><strong>Your AI agents are burning money. AgentMeter shows you exactly how much.</strong></p>
  <p align="center">
    <a href="https://github.com/saileshr/agentmeter-sdk/actions"><img src="https://github.com/saileshr/agentmeter-sdk/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://pypi.org/project/agentmeter-sdk/"><img src="https://img.shields.io/pypi/v/agentmeter-sdk" alt="PyPI"></a>
    <a href="https://pypi.org/project/agentmeter-sdk/"><img src="https://img.shields.io/pypi/pyversions/agentmeter-sdk" alt="Python"></a>
    <a href="https://github.com/saileshr/agentmeter-sdk/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  </p>
</p>

---

## The Problem

A typical AI agent setup using GPT-4o looks like it costs ~$50/month. **The real number is closer to $800.**

Hidden costs add up fast: verbose system prompts resent on every call, silent retries, tool calls that invoke expensive models, and zero visibility into what each agent actually spends.

We ran 10 identical tasks across 5 models. Here's what we found:

| Model | Cost for 10 tasks | vs. Cheapest |
|---|---|---|
| GPT-4o | $0.0617 | **16x** |
| Claude Sonnet 4 | $0.0912 | **24x** |
| GPT-4o-mini | $0.0038 | 1x (baseline) |
| Claude Haiku 4.5 | $0.0041 | 1.1x |
| GPT-4.1-nano | $0.0024 | **baseline** |

> At 1,000 calls/day, choosing GPT-4o over GPT-4.1-nano costs an extra **$131/month** for the same tasks.

<p align="center">
  <img src="images/agentmeter-sdk-output.png" alt="AgentMeter cost comparison report" width="700">
</p>

AgentMeter is a lightweight Python SDK that tracks every LLM call, calculates the real cost, and connects it to business outcomes. Zero dependencies. Two lines of code. Works offline.

## Quickstart (60 seconds)

```bash
pip install agentmeter-sdk[openai]
```

```python
import openai
from agentmeter import track_openai, MemoryExporter, configure

# 1. Set up tracking
mem = MemoryExporter()
configure(project="my-agent", exporters=[mem])

# 2. Wrap your client (one line)
client = track_openai(openai.OpenAI(), project="my-agent")

# 3. Use it normally — costs are tracked automatically
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize this support ticket..."}],
)

# See what you spent
print(f"Cost: ${mem.total_cost:.4f}")
print(f"Tokens: {mem.total_tokens}")
print(mem.summary())
```

## What it tracks

Every LLM call automatically records:

| | |
|---|---|
| **Token costs** | Input, output, and cached token counts with USD breakdown |
| **Model & provider** | Which model handled each call (GPT-4o, Claude Sonnet 4, etc.) |
| **Latency** | Per-call duration in milliseconds |
| **Tool calls** | Function/tool names invoked (names only — never arguments, for privacy) |
| **Errors** | Failed calls with error messages and cost of retries |
| **Outcomes** | Link agent costs to business results: "this call resolved a $12.50 ticket" |

**Privacy by default** — AgentMeter tracks cost metadata only. No message content, prompts, or tool arguments are ever captured.

## Framework Support

| Framework | Status | Adapter |
|---|---|---|
| **OpenAI SDK** | Supported | `track_openai()` |
| **Anthropic SDK** | Supported | `track_anthropic()` |
| **Any LLM** | Supported | `track_llm_call()` context manager |
| **LangChain** | Planned | Callback handler |
| **CrewAI** | Planned | |
| **AutoGen** | Planned | |

```python
# OpenAI
from agentmeter import track_openai
client = track_openai(openai.OpenAI(), project="support-agent")

# Anthropic
from agentmeter import track_anthropic
client = track_anthropic(anthropic.Anthropic(), project="research-agent")

# Any LLM (manual instrumentation)
from agentmeter import track_llm_call
with track_llm_call(provider="cohere", model="command-r-plus") as call:
    response = my_llm_call(...)
    call.input_tokens = response.meta.tokens.input_tokens
    call.output_tokens = response.meta.tokens.output_tokens
```

## Cost-Per-Outcome Attribution

This is what makes AgentMeter different. Not just "how much did I spend?" but **"how much did each business result cost?"**

```python
from agentmeter import record_outcome

# After your agent resolves a support ticket
record_outcome(
    run_id="run-123",
    outcome="ticket_resolved",
    value_usd=12.50,
    metadata={"ticket_id": "T-1234", "resolution_time_min": 3},
)

# Now you know: this ticket cost $0.05 in LLM calls
# and delivered $12.50 in value. ROI: 250x.
```

## Live Pricing for 300+ Models

AgentMeter ships with built-in pricing for OpenAI, Anthropic, Google, and Mistral. Need more?

```python
from agentmeter import CostRegistry

registry = CostRegistry()

# Pull 300+ models from litellm's community-maintained registry
registry.update_from_litellm()

# Or fetch from your own endpoint
registry.update_from_url("https://agentmeter.ai/api/pricing.json")

# Or set manually
registry.register("mycloud", "my-model", ModelPricing(
    input_per_1k=0.001, output_per_1k=0.002
))
```

The SDK never phones home by default. Remote pricing updates are always opt-in.

## Architecture

```
┌─────────────────────────────────────────────┐
│           Your Agent Code                    │
│  (OpenAI / Anthropic / LangChain / Custom)  │
│                                              │
│  client = track_openai(openai.OpenAI())     │  <- 1 line to add
└──────────────────┬───────────────────────────┘
                   │ records LLMEvent (tokens, cost, latency)
                   ▼
┌─────────────────────────────────────────────┐
│          AgentMeter SDK (in-process)         │
│                                              │
│  ┌──────────┐ ┌───────────┐ ┌────────────┐ │
│  │ Cost     │ │ Tracker   │ │ Outcome    │ │
│  │ Registry │ │ (enrich + │ │ Attribution│ │
│  │ (300+    │ │  export)  │ │            │ │
│  │ models)  │ │           │ │            │ │
│  └──────────┘ └─────┬─────┘ └────────────┘ │
└─────────────────────┼───────────────────────┘
                      │
            ┌─────────┼─────────┐
            ▼         ▼         ▼
       ┌────────┐ ┌────────┐ ┌────────┐
       │Console │ │Memory  │ │ HTTP   │
       │(stderr)│ │(local) │ │(cloud) │  <- future
       └────────┘ └────────┘ └────────┘
```

**Zero dependencies.** The core SDK uses only Python stdlib. Framework adapters (OpenAI, Anthropic) are optional extras.

```bash
pip install agentmeter-sdk            # core only — zero deps
pip install agentmeter-sdk[openai]    # + OpenAI SDK
pip install agentmeter-sdk[anthropic] # + Anthropic SDK
pip install agentmeter-sdk[all]       # everything
```

## Configuration

```python
from agentmeter import configure, MemoryExporter

mem = MemoryExporter()
configure(
    project="my-agent",
    tags={"team": "cx", "env": "prod"},
    exporters=[mem],
)
```

Or via environment variables:

```bash
export AGENTMETER_PROJECT=my-agent
export AGENTMETER_EXPORT=console   # or "memory"
export AGENTMETER_DEBUG=true       # log unknown models
export AGENTMETER_ENABLED=false    # kill switch
```

## Examples

See the [`examples/`](examples/) directory:

- **[Model Comparison](examples/model_comparison.py)** — Run the same tasks across GPT-4o, GPT-4o-mini, GPT-4.1-nano, Claude Sonnet 4, and Claude Haiku 4.5. Generates a screenshot-ready cost report.

## Contributing

We're building the financial infrastructure for AI agents — the Datadog + Stripe of the agentic era. And we'd love your help.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started. Good first issues:

- Add a new framework adapter (LangChain, CrewAI, AutoGen)
- Add a new exporter (file, HTTP, OpenTelemetry)
- Update model pricing in the cost registry
- Improve the terminal report formatting

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full plan.

**Now:** SDK with cost tracking, outcome attribution, OpenAI + Anthropic adapters
**Next:** LangChain/CrewAI adapters, streaming support, file exporter, CLI report command
**Later:** Hosted dashboard, billing-as-a-service, agent marketplace economics

## License

[Apache 2.0](LICENSE) — use it in production, fork it, build on it. No strings attached.
