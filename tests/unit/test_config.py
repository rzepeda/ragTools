"""
Unit tests for Configuration Management System.

This module contains comprehensive unit tests for configuration loading,
validation, environment overrides, hot-reload, and configuration access.
"""

import json
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from rag_factory.config import (
    ConfigFileHandler,
    ConfigManager,
    ConfigurationError,
    GlobalConfigSchema,
    PipelineConfigSchema,
    RAGConfigSchema,
    StrategyConfigSchema,
)


# Test helper to reset singleton between tests
@pytest.fixture(autouse=True)
def reset_config_manager():
    """Reset ConfigManager singleton between tests."""
    ConfigManager._instance = None
    ConfigManager._config = None
    ConfigManager._config_path = None
    ConfigManager._observers = []
    ConfigManager._callbacks = []
    yield
    # Cleanup after test
    if ConfigManager._instance:
        ConfigManager._instance.disable_hot_reload()


# TC4.1: Configuration Loading Tests
def test_load_from_yaml(tmp_path: Path) -> None:
    """Test loading configuration from YAML file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
global_settings:
  log_level: DEBUG
strategies:
  test_strategy:
    chunk_size: 1024
""")

    config = ConfigManager()
    config.load(str(config_file))

    assert config.get("global_settings.log_level") == "DEBUG"
    assert config.get("strategies.test_strategy.chunk_size") == 1024


def test_load_from_json(tmp_path: Path) -> None:
    """Test loading configuration from JSON file."""
    config_file = tmp_path / "config.json"
    config_file.write_text('{"global_settings": {"log_level": "ERROR"}}')

    config = ConfigManager()
    config.load(str(config_file))

    assert config.get("global_settings.log_level") == "ERROR"


def test_load_from_dict() -> None:
    """Test loading configuration from dictionary."""
    config_dict = {
        "global_settings": {"log_level": "INFO"},
        "strategies": {}
    }

    config = ConfigManager()
    config.load(config_dict=config_dict)

    assert config.get("global_settings.log_level") == "INFO"


def test_file_not_found_raises_error() -> None:
    """Test loading non-existent file raises error."""
    config = ConfigManager()

    with pytest.raises(ConfigurationError, match="not found"):
        config.load("/nonexistent/config.yaml")


def test_invalid_yaml_raises_error(tmp_path: Path) -> None:
    """Test invalid YAML raises error."""
    config_file = tmp_path / "bad_config.yaml"
    config_file.write_text("invalid: yaml: content:")

    config = ConfigManager()

    with pytest.raises(ConfigurationError, match="Invalid YAML"):
        config.load(str(config_file))


def test_invalid_json_raises_error(tmp_path: Path) -> None:
    """Test invalid JSON raises error."""
    config_file = tmp_path / "bad_config.json"
    config_file.write_text('{"invalid": json}')

    config = ConfigManager()

    with pytest.raises(ConfigurationError, match="Invalid JSON"):
        config.load(str(config_file))


def test_unsupported_file_format_raises_error(tmp_path: Path) -> None:
    """Test unsupported file format raises error."""
    config_file = tmp_path / "config.txt"
    config_file.write_text("some content")

    config = ConfigManager()

    with pytest.raises(ConfigurationError, match="Unsupported file format"):
        config.load(str(config_file))


def test_empty_yaml_loads_as_empty_dict(tmp_path: Path) -> None:
    """Test empty YAML file loads as empty dict."""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")

    config = ConfigManager()
    config.load(str(config_file))

    # Should load with defaults
    assert config.config is not None


# TC4.2: Validation Tests
def test_validation_enforces_chunk_size_range() -> None:
    """Test chunk_size must be within valid range."""
    config_dict = {
        "strategies": {
            "test": {
                "chunk_size": -1  # Invalid: must be >= 1
            }
        }
    }

    config = ConfigManager()

    with pytest.raises(ConfigurationError):
        config.load(config_dict=config_dict)


def test_validation_enforces_chunk_size_max() -> None:
    """Test chunk_size must not exceed maximum."""
    config_dict = {
        "strategies": {
            "test": {
                "chunk_size": 10000  # Invalid: must be <= 8192
            }
        }
    }

    config = ConfigManager()

    with pytest.raises(ConfigurationError):
        config.load(config_dict=config_dict)


def test_validation_enforces_top_k_min() -> None:
    """Test top_k must be positive."""
    config_dict = {
        "strategies": {
            "test": {
                "top_k": 0  # Invalid: must be >= 1
            }
        }
    }

    config = ConfigManager()

    with pytest.raises(ConfigurationError):
        config.load(config_dict=config_dict)


def test_validation_enforces_log_level() -> None:
    """Test log_level must be valid."""
    config_dict = {
        "global_settings": {
            "log_level": "INVALID_LEVEL"
        }
    }

    config = ConfigManager()

    with pytest.raises(ConfigurationError, match="Invalid log_level"):
        config.load(config_dict=config_dict)


def test_validation_accepts_valid_log_levels() -> None:
    """Test all valid log levels accepted."""
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    for level in valid_levels:
        config = ConfigManager()
        config_dict = {
            "global_settings": {
                "log_level": level
            }
        }
        config.load(config_dict=config_dict)
        assert config.get("global_settings.log_level") == level


def test_validation_normalizes_log_level_case() -> None:
    """Test log level case normalized to uppercase."""
    config_dict = {
        "global_settings": {
            "log_level": "debug"
        }
    }

    config = ConfigManager()
    config.load(config_dict=config_dict)

    assert config.get("global_settings.log_level") == "DEBUG"


# TC4.3: Default Configuration Tests
def test_default_config_loaded() -> None:
    """Test default configuration loaded when no config provided."""
    config = ConfigManager()
    config.load()

    assert config.get("global_settings.environment") == "development"
    assert config.get("global_settings.cache_enabled") is True
    assert config.get("global_settings.log_level") == "INFO"
    assert config.get("global_settings.cache_ttl") == 3600


def test_user_config_merged_with_defaults() -> None:
    """Test user configuration merged with defaults."""
    config_dict = {
        "global_settings": {
            "log_level": "DEBUG"
            # cache_enabled should come from defaults
        }
    }

    config = ConfigManager()
    config.load(config_dict=config_dict)

    assert config.get("global_settings.log_level") == "DEBUG"
    assert config.get("global_settings.cache_enabled") is True  # From default


def test_get_strategy_config_returns_default() -> None:
    """Test getting config for unregistered strategy returns defaults."""
    config = ConfigManager()
    config.load()

    strategy_config = config.get_strategy_config("nonexistent_strategy")

    assert strategy_config.chunk_size == 512  # Default value
    assert strategy_config.top_k == 5  # Default value


# TC4.4: Environment Configuration Tests
def test_environment_from_env_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test environment determined from environment variable."""
    base_config = tmp_path / "config.yaml"
    base_config.write_text("global_settings:\n  log_level: INFO\n")

    prod_config = tmp_path / "config.production.yaml"
    prod_config.write_text("global_settings:\n  log_level: ERROR\n")

    monkeypatch.setenv("RAG_ENV", "production")

    config = ConfigManager()
    config.load(str(base_config))

    assert config.get("global_settings.log_level") == "ERROR"


def test_environment_override_merges_correctly(tmp_path: Path) -> None:
    """Test environment-specific config merged with base."""
    base_config = tmp_path / "config.yaml"
    base_config.write_text("""
global_settings:
  log_level: INFO
  cache_enabled: true
  cache_ttl: 3600
""")

    env_config = tmp_path / "config.test.yaml"
    env_config.write_text("""
global_settings:
  log_level: DEBUG
""")

    config = ConfigManager()
    config.load(str(base_config), environment="test")

    assert config.get("global_settings.log_level") == "DEBUG"  # Overridden
    assert config.get("global_settings.cache_enabled") is True  # From base
    assert config.get("global_settings.cache_ttl") == 3600  # From base


def test_environment_override_nonexistent_file(tmp_path: Path) -> None:
    """Test environment override when env-specific file doesn't exist."""
    base_config = tmp_path / "config.yaml"
    base_config.write_text("global_settings:\n  log_level: INFO\n")

    config = ConfigManager()
    config.load(str(base_config), environment="nonexistent")

    # Should use base config
    assert config.get("global_settings.log_level") == "INFO"


def test_deep_merge_nested_dicts(tmp_path: Path) -> None:
    """Test deep merge preserves nested structure."""
    base_config = tmp_path / "config.yaml"
    base_config.write_text("""
strategies:
  strategy1:
    chunk_size: 512
    top_k: 5
    metadata:
      model: "base_model"
      param1: "value1"
""")

    env_config = tmp_path / "config.prod.yaml"
    env_config.write_text("""
strategies:
  strategy1:
    chunk_size: 1024
    metadata:
      model: "prod_model"
""")

    config = ConfigManager()
    config.load(str(base_config), environment="prod")

    assert config.get("strategies.strategy1.chunk_size") == 1024  # Overridden
    assert config.get("strategies.strategy1.top_k") == 5  # Preserved
    assert config.get("strategies.strategy1.metadata.model") == "prod_model"  # Overridden
    assert config.get("strategies.strategy1.metadata.param1") == "value1"  # Preserved


# TC4.5: Configuration Access Tests
def test_get_with_dot_notation() -> None:
    """Test accessing nested config with dot notation."""
    config_dict = {
        "global_settings": {
            "cache_enabled": True
        }
    }

    config = ConfigManager()
    config.load(config_dict=config_dict)

    assert config.get("global_settings.cache_enabled") is True


def test_get_missing_key_returns_default() -> None:
    """Test getting non-existent key returns default."""
    config = ConfigManager()
    config.load()

    value = config.get("nonexistent.key", default="default_value")
    assert value == "default_value"


def test_get_strategy_config() -> None:
    """Test getting strategy-specific configuration."""
    config_dict = {
        "strategies": {
            "my_strategy": {
                "chunk_size": 2048,
                "top_k": 15
            }
        }
    }

    config = ConfigManager()
    config.load(config_dict=config_dict)

    strategy_config = config.get_strategy_config("my_strategy")

    assert strategy_config.chunk_size == 2048
    assert strategy_config.top_k == 15


def test_get_before_load_raises_error() -> None:
    """Test accessing config before loading raises error."""
    config = ConfigManager()

    with pytest.raises(ConfigurationError, match="not loaded"):
        config.get("some.key")


def test_to_dict_exports_config() -> None:
    """Test exporting configuration as dictionary."""
    config_dict = {
        "global_settings": {"log_level": "INFO"},
        "strategies": {}
    }

    config = ConfigManager()
    config.load(config_dict=config_dict)

    exported = config.to_dict()

    assert isinstance(exported, dict)
    assert exported["global_settings"]["log_level"] == "INFO"


def test_to_dict_before_load_raises_error() -> None:
    """Test to_dict before loading raises error."""
    config = ConfigManager()

    with pytest.raises(ConfigurationError, match="not loaded"):
        config.to_dict()


def test_config_property_returns_schema() -> None:
    """Test config property returns RAGConfigSchema."""
    config = ConfigManager()
    config.load()

    assert isinstance(config.config, RAGConfigSchema)


# TC4.6: Hot-Reload Tests
def test_hot_reload_detects_file_changes(tmp_path: Path) -> None:
    """Test configuration reloaded when file changes."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("global_settings:\n  log_level: INFO\n")

    config = ConfigManager()
    config.load(str(config_file))

    callback_called: list[bool] = []

    def callback(new_config: RAGConfigSchema) -> None:
        callback_called.append(True)

    config.enable_hot_reload(callback)

    # Give observer time to start
    time.sleep(0.2)

    # Modify file
    config_file.write_text("global_settings:\n  log_level: DEBUG\n")

    # Wait for file watcher to detect change
    time.sleep(1.0)

    assert len(callback_called) > 0
    assert config.get("global_settings.log_level") == "DEBUG"

    config.disable_hot_reload()


def test_hot_reload_without_config_path_raises_error() -> None:
    """Test enabling hot-reload without config path raises error."""
    config = ConfigManager()
    config.load(config_dict={"global_settings": {}})

    with pytest.raises(ConfigurationError, match="without config file path"):
        config.enable_hot_reload()


def test_disable_hot_reload() -> None:
    """Test hot-reload can be disabled."""
    config_file = Path("test_config.yaml")
    # This test just verifies the method works
    config = ConfigManager()

    # Call disable even without enabling (should not error)
    config.disable_hot_reload()


def test_reload_method_reloads_config(tmp_path: Path) -> None:
    """Test reload method reloads configuration."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("global_settings:\n  log_level: INFO\n")

    config = ConfigManager()
    config.load(str(config_file))

    assert config.get("global_settings.log_level") == "INFO"

    # Modify file
    config_file.write_text("global_settings:\n  log_level: ERROR\n")

    # Manually reload
    config.reload()

    assert config.get("global_settings.log_level") == "ERROR"


# Additional tests for schema classes
def test_strategy_config_schema_defaults() -> None:
    """Test StrategyConfigSchema default values."""
    schema = StrategyConfigSchema()

    assert schema.chunk_size == 512
    assert schema.chunk_overlap == 50
    assert schema.top_k == 5
    assert schema.strategy_name == ""
    assert schema.metadata == {}


def test_global_config_schema_defaults() -> None:
    """Test GlobalConfigSchema default values."""
    schema = GlobalConfigSchema()

    assert schema.environment == "development"
    assert schema.log_level == "INFO"
    assert schema.cache_enabled is True
    assert schema.cache_ttl == 3600


def test_pipeline_config_schema_defaults() -> None:
    """Test PipelineConfigSchema default values."""
    schema = PipelineConfigSchema()

    assert schema.mode == "sequential"
    assert schema.stages == []
    assert schema.timeout is None


def test_rag_config_schema_defaults() -> None:
    """Test RAGConfigSchema default values."""
    schema = RAGConfigSchema()

    assert isinstance(schema.global_settings, GlobalConfigSchema)
    assert schema.strategies == {}
    assert schema.pipeline is None


# Singleton pattern tests
def test_config_manager_singleton() -> None:
    """Test ConfigManager implements singleton pattern."""
    config1 = ConfigManager()
    config2 = ConfigManager()

    assert config1 is config2


def test_config_file_handler_initialization() -> None:
    """Test ConfigFileHandler can be initialized."""
    config = ConfigManager()
    handler = ConfigFileHandler(config)

    assert handler.config_manager is config
