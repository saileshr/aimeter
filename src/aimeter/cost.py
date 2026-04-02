"""Cost registry — maps (provider, model, tokens) to USD.

Bundled pricing ships as a Python dict (zero dependencies, works offline).
For up-to-date pricing, use:
  - registry.update_from_url(url) — fetch from any JSON endpoint
  - registry.update_from_litellm() — fetch from litellm's community-maintained registry
  - registry.update_from_dict(data) — load from a dict at runtime

The SDK never phones home by default. Remote fetching is always opt-in.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from aimeter.types import CostBreakdown, TokenUsage

logger = logging.getLogger("aimeter")


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-1K-token pricing for a model."""

    input_per_1k: float
    output_per_1k: float
    cached_input_per_1k: float = 0.0


# Pricing as of April 2026. Prices are per 1,000 tokens.
# Source: provider pricing pages.
# Raw pricing — keyed by exact model names as providers use them.
# The registry constructor normalizes these keys for lookup.
_BUILTIN_PRICING_RAW: list[tuple[str, str, ModelPricing]] = [
    # OpenAI
    ("openai", "gpt-4o", ModelPricing(0.0025, 0.01, 0.00125)),
    ("openai", "gpt-4o-mini", ModelPricing(0.00015, 0.0006, 0.000075)),
    ("openai", "gpt-4.1", ModelPricing(0.002, 0.008, 0.0005)),
    ("openai", "gpt-4.1-mini", ModelPricing(0.0004, 0.0016, 0.0001)),
    ("openai", "gpt-4.1-nano", ModelPricing(0.0001, 0.0004, 0.000025)),
    ("openai", "o3", ModelPricing(0.01, 0.04, 0.0025)),
    ("openai", "o3-mini", ModelPricing(0.0011, 0.0044, 0.000275)),
    ("openai", "o4-mini", ModelPricing(0.0011, 0.0044, 0.000275)),
    # Anthropic
    ("anthropic", "claude-sonnet-4-20250514", ModelPricing(0.003, 0.015, 0.0003)),
    ("anthropic", "claude-opus-4-20250514", ModelPricing(0.015, 0.075, 0.0015)),
    ("anthropic", "claude-haiku-4-5-20251001", ModelPricing(0.0008, 0.004, 0.00008)),
    # Google
    ("google", "gemini-2.5-pro", ModelPricing(0.00125, 0.01, 0.000315)),
    ("google", "gemini-2.5-flash", ModelPricing(0.00015, 0.0006, 0.0000375)),
    # Mistral
    ("mistral", "mistral-large", ModelPricing(0.002, 0.006)),
]

# Date-suffix pattern: strip "-YYYYMMDD" or "-YYYY-MM-DD" from model names
_DATE_SUFFIX = re.compile(r"-\d{4}-?\d{2}-?\d{2}$")


def _normalize_model(model: str) -> str:
    """Normalize model name for registry lookup.

    Strips date suffixes (e.g., 'gpt-4o-2024-08-06' -> 'gpt-4o')
    and lowercases.
    """
    model = model.lower().strip()
    return _DATE_SUFFIX.sub("", model)


class CostRegistry:
    """Calculates USD cost from provider + model + token counts.

    Uses built-in pricing by default. Users can register custom models
    or override existing pricing at runtime.
    """

    def __init__(self) -> None:
        self._models: dict[tuple[str, str], ModelPricing] = {}
        for provider, model, pricing in _BUILTIN_PRICING_RAW:
            self._register_both(provider, model, pricing)

    def _register_both(self, provider: str, model: str, pricing: ModelPricing) -> None:
        """Register under both the exact name and the normalized name."""
        p = provider.lower()
        m = model.lower()
        self._models[(p, m)] = pricing
        normalized = _normalize_model(m)
        if normalized != m:
            self._models[(p, normalized)] = pricing

    def register(self, provider: str, model: str, pricing: ModelPricing) -> None:
        """Register or override pricing for a model."""
        self._register_both(provider, model, pricing)

    def calculate(self, provider: str, model: str, tokens: TokenUsage) -> CostBreakdown:
        """Calculate cost for an LLM call.

        Returns zero cost if the model is not in the registry (with a debug warning).
        """
        key = (provider.lower(), _normalize_model(model))
        pricing = self._models.get(key)

        if pricing is None:
            logger.debug(
                "aimeter: unknown model '%s/%s' — cost will be $0.00. "
                "Use registry.register() to add custom pricing.",
                provider,
                model,
            )
            return CostBreakdown()

        input_cost = (tokens.input_tokens / 1000) * pricing.input_per_1k
        output_cost = (tokens.output_tokens / 1000) * pricing.output_per_1k
        cached_cost = (tokens.cached_tokens / 1000) * pricing.cached_input_per_1k

        return CostBreakdown(
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            cached_cost_usd=cached_cost,
        )

    def has_model(self, provider: str, model: str) -> bool:
        """Check if a model is in the registry."""
        return (provider.lower(), _normalize_model(model)) in self._models

    def list_models(self) -> list[tuple[str, str]]:
        """List all registered (provider, model) pairs."""
        return sorted(self._models.keys())

    def update_from_dict(self, data: dict[str, dict[str, dict[str, Any]]]) -> int:
        """Update registry from a nested dict.

        Expected shape: {"provider": {"model": {"input_per_1k": ..., "output_per_1k": ...}}}
        Returns the number of models added/updated.
        """
        count = 0
        for provider, models in data.items():
            for model, prices in models.items():
                self.register(
                    provider,
                    model,
                    ModelPricing(
                        input_per_1k=prices["input_per_1k"],
                        output_per_1k=prices["output_per_1k"],
                        cached_input_per_1k=prices.get("cached_input_per_1k", 0.0),
                    ),
                )
                count += 1
        return count

    def update_from_url(self, url: str, timeout: float = 10.0) -> int:
        """Fetch pricing JSON from a URL and update the registry.

        The JSON must be in the aimeter dict format:
        {"provider": {"model": {"input_per_1k": ..., "output_per_1k": ...}}}

        Args:
            url: URL to fetch JSON pricing data from.
            timeout: Request timeout in seconds.

        Returns:
            Number of models added/updated.

        Raises:
            urllib.error.URLError: If the request fails.
            json.JSONDecodeError: If the response is not valid JSON.
        """
        req = urllib.request.Request(url, headers={"User-Agent": "aimeter"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return self.update_from_dict(data)

    def update_from_litellm(self, timeout: float = 10.0) -> int:
        """Fetch pricing from litellm's community-maintained model registry.

        This pulls from:
        https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json

        litellm's format uses per-token pricing; this method converts to per-1K.
        Only models with both input and output pricing are imported.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            Number of models added/updated.
        """
        url = (
            "https://raw.githubusercontent.com/BerriAI/litellm/main/"
            "model_prices_and_context_window.json"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "aimeter"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        return self._import_litellm_format(raw)

    def _import_litellm_format(self, raw: dict[str, Any]) -> int:
        """Parse litellm's pricing format into our registry.

        litellm format per entry:
            "provider/model-name": {
                "input_cost_per_token": 2.5e-06,
                "output_cost_per_token": 1.0e-05,
                "cache_read_input_token_cost": 1.25e-06,  (optional)
                "litellm_provider": "openai",
            }

        We convert per-token to per-1K-token and normalize provider names.
        """
        # Map litellm provider names to our canonical names
        provider_map = {
            "openai": "openai",
            "anthropic": "anthropic",
            "vertex_ai-language-models": "google",
            "vertex_ai-chat-models": "google",
            "gemini": "google",
            "mistral": "mistral",
            "cohere": "cohere",
            "cohere_chat": "cohere",
            "bedrock": "aws-bedrock",
            "azure": "azure",
            "groq": "groq",
            "together_ai": "together",
            "fireworks_ai": "fireworks",
            "deepseek": "deepseek",
            "perplexity": "perplexity",
        }

        count = 0
        for model_key, info in raw.items():
            if not isinstance(info, dict):
                continue

            input_cost = info.get("input_cost_per_token")
            output_cost = info.get("output_cost_per_token")

            # Skip entries without pricing
            if input_cost is None or output_cost is None:
                continue

            cached_cost = info.get("cache_read_input_token_cost", 0.0) or 0.0

            # Determine provider
            litellm_provider = info.get("litellm_provider", "")
            provider = provider_map.get(litellm_provider, litellm_provider)

            # Extract model name: strip "provider/" prefix if present
            model_name = model_key.split("/", 1)[1] if "/" in model_key else model_key

            # Convert per-token to per-1K-token
            pricing = ModelPricing(
                input_per_1k=input_cost * 1000,
                output_per_1k=output_cost * 1000,
                cached_input_per_1k=cached_cost * 1000,
            )

            self._register_both(provider, model_name, pricing)
            count += 1

        logger.info("aimeter: loaded %d models from litellm registry", count)
        return count

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, dict[str, Any]]]) -> CostRegistry:
        """Create a registry from a nested dict.

        Expected shape: {"provider": {"model": {"input_per_1k": ..., "output_per_1k": ...}}}
        """
        registry = cls()
        registry.update_from_dict(data)
        return registry
