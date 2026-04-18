"""AIMeter — cost tracking and attribution for AI agents.

Usage:
    # OpenAI
    from aimeter import track_openai
    client = track_openai(openai.OpenAI(), project="my-agent")

    # Any LLM (manual)
    from aimeter import track_llm_call
    with track_llm_call(provider="openai", model="gpt-4o") as call:
        response = my_llm_call(...)
        call.input_tokens = response.usage.input_tokens

    # Business outcome attribution
    from aimeter import record_outcome
    record_outcome(run_id="run-123", outcome="ticket_resolved", value_usd=12.50)
"""

from aimeter._version import __version__
from aimeter.adapters.generic import track_llm_call
from aimeter.config import AIMeterConfig
from aimeter.cost import CostRegistry, ModelPricing
from aimeter.exporters.console import ConsoleExporter
from aimeter.exporters.memory import MemoryExporter
from aimeter.outcome import record_outcome
from aimeter.tracker import configure, get_tracker, reset
from aimeter.types import CostBreakdown, LLMEvent, Outcome, TokenUsage

# Lazy imports for optional-dependency adapters
_LAZY_IMPORTS = {
    "track_openai": ("aimeter.adapters.openai", "track_openai"),
    "track_anthropic": ("aimeter.adapters.anthropic", "track_anthropic"),
    "track_gemini": ("aimeter.adapters.gemini", "track_gemini"),
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
                f"aimeter.{name} requires the '{package}' package. "
                f"Install it with: pip install aimeter[{package}]"
            ) from e
    raise AttributeError(f"module 'aimeter' has no attribute {name}")


__all__ = [
    "__version__",
    # Config
    "AIMeterConfig",
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
    "track_gemini",
    "track_llm_call",
    # Outcome
    "record_outcome",
    # Exporters
    "ConsoleExporter",
    "MemoryExporter",
]
