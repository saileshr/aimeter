# AIMeter Roadmap

## MVP (Now)

The open-source SDK for agent cost tracking. Works locally, zero dependencies, two lines of code.

- [x] Core data model (LLMEvent, TokenUsage, CostBreakdown, Outcome)
- [x] Cost registry with built-in pricing for OpenAI, Anthropic, Google, Mistral
- [x] Dynamic pricing updates (litellm integration, custom URLs, runtime override)
- [x] OpenAI adapter (`track_openai`)
- [x] Anthropic adapter (`track_anthropic`)
- [x] Generic adapter (`track_llm_call` context manager)
- [x] Outcome attribution (`record_outcome`)
- [x] Console exporter (JSON-lines to stderr)
- [x] Memory exporter (in-memory with query helpers)
- [x] Terminal cost report (screenshot-ready)
- [x] Configuration via code and environment variables
- [x] Model comparison demo

## V1.1 (Next)

More adapters, streaming, and the CLI.

- [ ] LangChain adapter (callback handler)
- [ ] CrewAI adapter
- [ ] OpenAI streaming support
- [ ] Anthropic streaming support
- [ ] File exporter (JSON-lines to disk with rotation)
- [ ] CLI: `aimeter report` — terminal cost summary from event files
- [ ] CLI: `aimeter models` — list all known models and pricing
- [ ] OpenTelemetry exporter (emit as OTEL spans)
- [ ] Async adapter support (AsyncOpenAI, AsyncAnthropic)

## V2 (Later)

Cost-per-outcome analytics and the hosted dashboard.

- [ ] AgentRun aggregation (group events into runs with totals)
- [ ] Cost-per-outcome analytics (query historical cost/value ratios)
- [ ] Hosted dashboard (real-time cost per agent, per project, fleet overview)
- [ ] HTTP exporter (POST events to AIMeter cloud)
- [ ] Cost anomaly detection (alert on spend spikes, runaway agents)
- [ ] Budget limits and alerts (hard caps, warnings at thresholds)
- [ ] Model recommendation engine (suggest cheaper models for equivalent tasks)
- [ ] Team/organization support (shared projects, RBAC)

## V3 (Future)

Billing infrastructure for agent builders.

- [ ] Billing-as-a-service for agent builders to monetize their products
- [ ] Usage-based billing primitives (per-call, per-outcome, per-minute, hybrid)
- [ ] Multi-tenant metering for SaaS companies running agents on behalf of customers
- [ ] Agent marketplace economics (revenue sharing, usage tiers)
- [ ] Embeddable cost dashboards for end users
- [ ] SOC 2, HIPAA compliance features

---

Have ideas? [Open a feature request](https://github.com/saileshr/agentmeter-sdk/issues/new?template=feature_request.yml) or start a discussion.
