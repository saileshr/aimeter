"""Tests for agentmeter.cost — cost registry and calculation."""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

from agentmeter.cost import CostRegistry, ModelPricing, _normalize_model
from agentmeter.types import CostBreakdown, TokenUsage


class TestNormalizeModel:
    def test_strips_date_suffix(self):
        assert _normalize_model("gpt-4o-2024-08-06") == "gpt-4o"

    def test_strips_compact_date(self):
        assert _normalize_model("claude-sonnet-4-20250514") == "claude-sonnet-4"

    def test_lowercases(self):
        assert _normalize_model("GPT-4o") == "gpt-4o"

    def test_strips_whitespace(self):
        assert _normalize_model("  gpt-4o  ") == "gpt-4o"

    def test_no_date_unchanged(self):
        assert _normalize_model("mistral-large") == "mistral-large"

    def test_preserves_version_numbers(self):
        assert _normalize_model("gpt-4.1-mini") == "gpt-4.1-mini"


class TestCostRegistry:
    def test_builtin_openai_pricing(self):
        registry = CostRegistry()
        tokens = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = registry.calculate("openai", "gpt-4o", tokens)
        assert abs(cost.input_cost_usd - 0.0025) < 1e-9
        assert abs(cost.output_cost_usd - 0.005) < 1e-9
        assert abs(cost.total_cost_usd - 0.0075) < 1e-9

    def test_builtin_anthropic_pricing(self):
        registry = CostRegistry()
        tokens = TokenUsage(input_tokens=2000, output_tokens=1000)
        cost = registry.calculate("anthropic", "claude-sonnet-4-20250514", tokens)
        assert abs(cost.input_cost_usd - 0.006) < 1e-9
        assert abs(cost.output_cost_usd - 0.015) < 1e-9

    def test_cached_tokens(self):
        registry = CostRegistry()
        tokens = TokenUsage(input_tokens=500, output_tokens=200, cached_tokens=300)
        cost = registry.calculate("openai", "gpt-4o", tokens)
        expected_cached = (300 / 1000) * 0.00125
        assert abs(cost.cached_cost_usd - expected_cached) < 1e-9

    def test_unknown_model_returns_zero(self):
        registry = CostRegistry()
        tokens = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = registry.calculate("openai", "nonexistent-model", tokens)
        assert cost == CostBreakdown()
        assert cost.total_cost_usd == 0.0

    def test_unknown_model_logs_debug(self, caplog):
        registry = CostRegistry()
        tokens = TokenUsage(input_tokens=100)
        with caplog.at_level(logging.DEBUG, logger="agentmeter"):
            registry.calculate("openai", "fake-model", tokens)
        assert "unknown model" in caplog.text
        assert "fake-model" in caplog.text

    def test_date_suffix_normalization(self):
        registry = CostRegistry()
        tokens = TokenUsage(input_tokens=1000, output_tokens=500)
        # gpt-4o with a date suffix should still match
        cost = registry.calculate("openai", "gpt-4o-2024-08-06", tokens)
        assert cost.total_cost_usd > 0

    def test_case_insensitive(self):
        registry = CostRegistry()
        tokens = TokenUsage(input_tokens=1000)
        cost = registry.calculate("OpenAI", "GPT-4o", tokens)
        assert cost.input_cost_usd > 0

    def test_register_custom_model(self):
        registry = CostRegistry()
        registry.register("custom", "my-model", ModelPricing(0.001, 0.002))
        tokens = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = registry.calculate("custom", "my-model", tokens)
        assert abs(cost.input_cost_usd - 0.001) < 1e-9
        assert abs(cost.output_cost_usd - 0.001) < 1e-9

    def test_override_existing_pricing(self):
        registry = CostRegistry()
        registry.register("openai", "gpt-4o", ModelPricing(0.999, 0.999))
        tokens = TokenUsage(input_tokens=1000)
        cost = registry.calculate("openai", "gpt-4o", tokens)
        assert abs(cost.input_cost_usd - 0.999) < 1e-9

    def test_has_model(self):
        registry = CostRegistry()
        assert registry.has_model("openai", "gpt-4o")
        assert registry.has_model("openai", "gpt-4o-2024-08-06")  # normalized
        assert not registry.has_model("openai", "nonexistent")

    def test_list_models(self):
        registry = CostRegistry()
        models = registry.list_models()
        assert ("openai", "gpt-4o") in models
        assert ("anthropic", "claude-sonnet-4-20250514") in models
        assert len(models) > 10

    def test_from_dict(self):
        data = {
            "mycloud": {
                "my-llm": {"input_per_1k": 0.01, "output_per_1k": 0.02},
                "my-llm-fast": {
                    "input_per_1k": 0.005,
                    "output_per_1k": 0.01,
                    "cached_input_per_1k": 0.001,
                },
            }
        }
        registry = CostRegistry.from_dict(data)
        assert registry.has_model("mycloud", "my-llm")
        tokens = TokenUsage(input_tokens=1000, output_tokens=1000)
        cost = registry.calculate("mycloud", "my-llm", tokens)
        assert abs(cost.input_cost_usd - 0.01) < 1e-9
        assert abs(cost.output_cost_usd - 0.02) < 1e-9

    def test_zero_tokens(self):
        registry = CostRegistry()
        cost = registry.calculate("openai", "gpt-4o", TokenUsage())
        assert cost.total_cost_usd == 0.0


class TestUpdateFromDict:
    def test_adds_new_models(self):
        registry = CostRegistry()
        count = registry.update_from_dict({
            "newprovider": {
                "model-a": {"input_per_1k": 0.01, "output_per_1k": 0.02},
                "model-b": {"input_per_1k": 0.005, "output_per_1k": 0.01},
            }
        })
        assert count == 2
        assert registry.has_model("newprovider", "model-a")
        assert registry.has_model("newprovider", "model-b")

    def test_overrides_existing(self):
        registry = CostRegistry()
        registry.update_from_dict({
            "openai": {
                "gpt-4o": {"input_per_1k": 0.999, "output_per_1k": 0.888},
            }
        })
        tokens = TokenUsage(input_tokens=1000)
        cost = registry.calculate("openai", "gpt-4o", tokens)
        assert abs(cost.input_cost_usd - 0.999) < 1e-9

    def test_with_cached_pricing(self):
        registry = CostRegistry()
        registry.update_from_dict({
            "test": {
                "cached-model": {
                    "input_per_1k": 0.01,
                    "output_per_1k": 0.02,
                    "cached_input_per_1k": 0.005,
                },
            }
        })
        tokens = TokenUsage(input_tokens=1000, cached_tokens=500)
        cost = registry.calculate("test", "cached-model", tokens)
        assert abs(cost.cached_cost_usd - 0.0025) < 1e-9


class TestUpdateFromUrl:
    """Tests using a local HTTP server to avoid external dependencies."""

    def _serve_json(self, data: dict) -> tuple[HTTPServer, int]:
        """Start a local HTTP server that serves JSON."""
        payload = json.dumps(data).encode()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format, *args):
                pass  # suppress logs

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = Thread(target=server.handle_request, daemon=True)
        thread.start()
        return server, port

    def test_fetches_and_updates(self):
        data = {
            "remote-provider": {
                "remote-model": {"input_per_1k": 0.05, "output_per_1k": 0.10},
            }
        }
        server, port = self._serve_json(data)
        try:
            registry = CostRegistry()
            count = registry.update_from_url(f"http://127.0.0.1:{port}/pricing.json")
            assert count == 1
            assert registry.has_model("remote-provider", "remote-model")
            tokens = TokenUsage(input_tokens=1000, output_tokens=1000)
            cost = registry.calculate("remote-provider", "remote-model", tokens)
            assert abs(cost.input_cost_usd - 0.05) < 1e-9
        finally:
            server.server_close()

    def test_bad_url_raises(self):
        registry = CostRegistry()
        try:
            registry.update_from_url("http://127.0.0.1:1/nonexistent", timeout=1.0)
            assert False, "Should have raised"
        except (OSError, ConnectionError):
            pass


class TestImportLitellmFormat:
    def test_parses_litellm_entries(self):
        raw = {
            "gpt-4o": {
                "input_cost_per_token": 2.5e-06,
                "output_cost_per_token": 1.0e-05,
                "cache_read_input_token_cost": 1.25e-06,
                "litellm_provider": "openai",
                "mode": "chat",
            },
            "anthropic/claude-sonnet-4-20250514": {
                "input_cost_per_token": 3.0e-06,
                "output_cost_per_token": 1.5e-05,
                "cache_read_input_token_cost": 3.0e-07,
                "litellm_provider": "anthropic",
                "mode": "chat",
            },
        }
        registry = CostRegistry()
        count = registry._import_litellm_format(raw)
        assert count == 2

        # Check per-token was converted to per-1K correctly
        tokens = TokenUsage(input_tokens=1000, output_tokens=1000)
        cost = registry.calculate("openai", "gpt-4o", tokens)
        assert abs(cost.input_cost_usd - 0.0025) < 1e-6
        assert abs(cost.output_cost_usd - 0.01) < 1e-6

    def test_strips_provider_prefix(self):
        raw = {
            "anthropic/my-model": {
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 2e-06,
                "litellm_provider": "anthropic",
            },
        }
        registry = CostRegistry()
        registry._import_litellm_format(raw)
        assert registry.has_model("anthropic", "my-model")

    def test_skips_entries_without_pricing(self):
        raw = {
            "some-model": {
                "mode": "chat",
                "litellm_provider": "openai",
                # no cost fields
            },
            "priced-model": {
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 2e-06,
                "litellm_provider": "openai",
            },
        }
        registry = CostRegistry()
        count = registry._import_litellm_format(raw)
        assert count == 1
        assert not registry.has_model("openai", "some-model")
        assert registry.has_model("openai", "priced-model")

    def test_maps_provider_names(self):
        raw = {
            "gemini/gemini-2.5-pro": {
                "input_cost_per_token": 1.25e-06,
                "output_cost_per_token": 1.0e-05,
                "litellm_provider": "gemini",
            },
            "groq/llama-3": {
                "input_cost_per_token": 5e-07,
                "output_cost_per_token": 1e-06,
                "litellm_provider": "groq",
            },
        }
        registry = CostRegistry()
        registry._import_litellm_format(raw)
        assert registry.has_model("google", "gemini-2.5-pro")
        assert registry.has_model("groq", "llama-3")

    def test_handles_missing_cached_cost(self):
        raw = {
            "some-model": {
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 2e-06,
                "litellm_provider": "openai",
                # no cache_read_input_token_cost
            },
        }
        registry = CostRegistry()
        registry._import_litellm_format(raw)
        tokens = TokenUsage(input_tokens=1000, cached_tokens=500)
        cost = registry.calculate("openai", "some-model", tokens)
        assert cost.cached_cost_usd == 0.0
