"""Tests for aimeter.config."""


from aimeter.config import AIMeterConfig
from aimeter.exporters.console import ConsoleExporter
from aimeter.exporters.memory import MemoryExporter


class TestAIMeterConfig:
    def test_defaults(self):
        config = AIMeterConfig()
        assert config.project == "default"
        assert config.tags == {}
        assert config.enabled is True
        assert config.debug is False
        assert len(config.exporters) == 1
        assert isinstance(config.exporters[0], ConsoleExporter)

    def test_explicit_project(self):
        config = AIMeterConfig(project="my-project")
        assert config.project == "my-project"

    def test_explicit_exporters(self):
        mem = MemoryExporter()
        config = AIMeterConfig(exporters=[mem])
        assert config.exporters == [mem]

    def test_env_project(self, monkeypatch):
        monkeypatch.setenv("AIMETER_PROJECT", "env-project")
        config = AIMeterConfig()
        assert config.project == "env-project"

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("AIMETER_PROJECT", "env-project")
        config = AIMeterConfig(project="explicit")
        assert config.project == "explicit"

    def test_env_disabled(self, monkeypatch):
        monkeypatch.setenv("AIMETER_ENABLED", "false")
        config = AIMeterConfig()
        assert config.enabled is False

    def test_env_debug(self, monkeypatch):
        monkeypatch.setenv("AIMETER_DEBUG", "true")
        config = AIMeterConfig()
        assert config.debug is True

    def test_env_export_memory(self, monkeypatch):
        monkeypatch.setenv("AIMETER_EXPORT", "memory")
        config = AIMeterConfig()
        assert len(config.exporters) == 1
        assert isinstance(config.exporters[0], MemoryExporter)

    def test_disabled_config(self):
        config = AIMeterConfig(enabled=False)
        assert config.enabled is False
