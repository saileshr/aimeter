"""Tests for agentmeter.config."""

import os

from agentmeter.config import AgentMeterConfig
from agentmeter.exporters.console import ConsoleExporter
from agentmeter.exporters.memory import MemoryExporter


class TestAgentMeterConfig:
    def test_defaults(self):
        config = AgentMeterConfig()
        assert config.project == "default"
        assert config.tags == {}
        assert config.enabled is True
        assert config.debug is False
        assert len(config.exporters) == 1
        assert isinstance(config.exporters[0], ConsoleExporter)

    def test_explicit_project(self):
        config = AgentMeterConfig(project="my-project")
        assert config.project == "my-project"

    def test_explicit_exporters(self):
        mem = MemoryExporter()
        config = AgentMeterConfig(exporters=[mem])
        assert config.exporters == [mem]

    def test_env_project(self, monkeypatch):
        monkeypatch.setenv("AGENTMETER_PROJECT", "env-project")
        config = AgentMeterConfig()
        assert config.project == "env-project"

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("AGENTMETER_PROJECT", "env-project")
        config = AgentMeterConfig(project="explicit")
        assert config.project == "explicit"

    def test_env_disabled(self, monkeypatch):
        monkeypatch.setenv("AGENTMETER_ENABLED", "false")
        config = AgentMeterConfig()
        assert config.enabled is False

    def test_env_debug(self, monkeypatch):
        monkeypatch.setenv("AGENTMETER_DEBUG", "true")
        config = AgentMeterConfig()
        assert config.debug is True

    def test_env_export_memory(self, monkeypatch):
        monkeypatch.setenv("AGENTMETER_EXPORT", "memory")
        config = AgentMeterConfig()
        assert len(config.exporters) == 1
        assert isinstance(config.exporters[0], MemoryExporter)

    def test_disabled_config(self):
        config = AgentMeterConfig(enabled=False)
        assert config.enabled is False
