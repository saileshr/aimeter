"""AgentMeter — cost tracking and attribution for AI agents.

Usage:
    # OpenAI
    from agentmeter import track_openai
    client = track_openai(openai.OpenAI(), project="my-agent")

    # Any LLM (manual)
    from agentmeter import track_llm_call
    with track_llm_call(provider="openai", model="gpt-4o") as call:
        response = my_llm_call(...)
        call.input_tokens = response.usage.input_tokens

    # Business outcome attribution
    from agentmeter import record_outcome
    record_outcome(run_id="run-123", outcome="ticket_resolved", value_usd=12.50)
"""

from agentmeter._version import __version__
from agentmeter.adapters.generic import track_llm_call
from agentmeter.config import AgentMeterConfig
from agentmeter.cost import CostRegistry, ModelPricing
from agentmeter.exporters.console import ConsoleExporter
from agentmeter.exporters.memory import MemoryExporter
from agentmeter.outcome import record_outcome
from agentmeter.tracker import configure, get_tracker, reset
from agentmeter.types import CostBreakdown, LLMEvent, Outcome, TokenUsage

# Lazy imports for optional-dependency adapters
_LAZY_IMPORTS = {
    "track_openai": ("agentmeter.adapters.openai", "track_openai"),
    "track_anthropic": ("agentmeter.adapters.anthropic", "track_anthropic"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        try:
            import importlib

            module = importlib.import_module(module_path)
            return getattr(module, attr_name)
        except ImportError as e:
            package = name.replace("track_", "")
            raise ImportError(
                f"agentmeter.{name} requires the '{package}' package. "
                f"Install it with: pip install agentmeter-sdk[{package}]"
            ) from e
    raise AttributeError(f"module 'agentmeter' has no attribute {name}")


__all__ = [
    "__version__",
    # Config
    "AgentMeterConfig",
    "configure",
    "get_tracker",
    "reset",
    # Types
    "LLMEvent",
    "TokenUsage",
    "CostBreakdown",
    "Outcome",
    # Cost
    "CostRegistry",
    "ModelPricing",
    # Adapters
    "track_openai",
    "track_anthropic",
    "track_llm_call",
    # Outcome
    "record_outcome",
    # Exporters
    "ConsoleExporter",
    "MemoryExporter",
]
