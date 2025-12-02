# Story 1.4: Create Configuration Management System

**Story ID:** 1.4
**Epic:** Epic 1 - Core Infrastructure & Factory Pattern
**Story Points:** 5
**Priority:** High
**Dependencies:** Story 1.1 (RAG Strategy Interface)

---

## User Story

**As a** developer
**I want** centralized configuration for all RAG strategies
**So that** I can easily tune and experiment with different settings

---

## Detailed Requirements

### Functional Requirements

1. **Configuration File Support**
   - Support YAML configuration files
   - Support JSON configuration files
   - Support loading from file path or file-like objects
   - Support multiple configuration files with inheritance/merging

2. **Configuration Schema**
   - Define schema for global settings
   - Define schema for strategy-specific settings
   - Define schema for pipeline configurations
   - Include environment-specific overrides

3. **Validation**
   - Validate configuration against schema
   - Check required fields are present
   - Validate data types and value ranges
   - Provide clear error messages for invalid configs

4. **Environment-Specific Configurations**
   - Support dev, test, prod environments
   - Allow environment-specific overrides
   - Load configuration based on environment variable
   - Merge base config with environment-specific config

5. **Default Configurations**
   - Provide sensible defaults for all strategies
   - Allow strategies to define their own defaults
   - Merge user config with defaults
   - Document all default values

6. **Hot-Reload Capability**
   - Watch configuration files for changes
   - Reload configuration automatically in development
   - Notify when configuration changes
   - Validate before applying new configuration

7. **Configuration Access**
   - Simple API to get configuration values
   - Support dot notation for nested keys (e.g., `config.get("strategy.chunk_size")`)
   - Type-safe configuration access
   - Cache configuration for performance

### Non-Functional Requirements

1. **Performance**
   - Configuration loading < 100ms
   - Configuration access < 1ms (cached)
   - Minimal memory footprint

2. **Usability**
   - Clear configuration structure
   - Well-documented configuration options
   - Good error messages for invalid configs

3. **Maintainability**
   - Centralized configuration logic
   - Easy to add new configuration options
   - Version configuration schema

4. **Security**
   - Support for environment variables for sensitive data
   - Don't log sensitive configuration values
   - Validate configuration to prevent injection attacks

---

## Acceptance Criteria

### AC1: File Format Support
- [ ] Can load configuration from YAML files
- [ ] Can load configuration from JSON files
- [ ] Handles file not found errors gracefully
- [ ] Handles malformed file syntax errors with clear messages

### AC2: Schema Definition
- [ ] Configuration schema defined (using pydantic or similar)
- [ ] Schema includes all common configuration options
- [ ] Schema supports strategy-specific configurations
- [ ] Schema documented with descriptions and examples

### AC3: Validation
- [ ] Configuration validated against schema on load
- [ ] Invalid configurations raise validation errors
- [ ] Error messages indicate which fields are invalid
- [ ] Validation includes type checking and range checking

### AC4: Environment Support
- [ ] Supports loading different configs for dev/test/prod
- [ ] Environment determined by environment variable (e.g., `RAG_ENV`)
- [ ] Environment-specific configs override base config
- [ ] Falls back to default environment if not specified

### AC5: Default Values
- [ ] Default configuration provided for all strategies
- [ ] User configuration merged with defaults
- [ ] Defaults documented in code and docs
- [ ] Can access defaults programmatically

### AC6: Hot-Reload
- [ ] Configuration files watched for changes (dev mode)
- [ ] Configuration reloaded automatically when files change
- [ ] Callback mechanism to notify of configuration changes
- [ ] Hot-reload can be disabled for production

### AC7: Configuration Access API
- [ ] Simple `get()` method for accessing values
- [ ] Supports nested key access with dot notation
- [ ] Returns type-safe values
- [ ] Raises error for missing required keys

---

## Technical Specifications

### File Location
`rag_factory/config.py`

### Dependencies
```python
from typing import Any, Dict, Optional, List, Callable
from pathlib import Path
import os
import yaml
import json
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field, validator
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
```

### Configuration Schema
```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class StrategyConfigSchema(BaseModel):
    """Schema for strategy configuration."""
    chunk_size: int = Field(512, ge=1, le=8192, description="Size of text chunks")
    chunk_overlap: int = Field(50, ge=0, le=500, description="Overlap between chunks")
    top_k: int = Field(5, ge=1, le=100, description="Number of results to retrieve")
    strategy_name: str = Field("", description="Strategy identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

class PipelineConfigSchema(BaseModel):
    """Schema for pipeline configuration."""
    mode: str = Field("sequential", description="Execution mode")
    stages: List[Dict[str, Any]] = Field(default_factory=list, description="Pipeline stages")
    timeout: Optional[int] = Field(None, description="Pipeline timeout in seconds")

class GlobalConfigSchema(BaseModel):
    """Schema for global configuration."""
    environment: str = Field("development", description="Environment name")
    log_level: str = Field("INFO", description="Logging level")
    cache_enabled: bool = Field(True, description="Enable result caching")
    cache_ttl: int = Field(3600, description="Cache TTL in seconds")

class RAGConfigSchema(BaseModel):
    """Complete RAG configuration schema."""
    global_settings: GlobalConfigSchema = Field(default_factory=GlobalConfigSchema)
    strategies: Dict[str, StrategyConfigSchema] = Field(default_factory=dict)
    pipeline: Optional[PipelineConfigSchema] = None
```

### Configuration Manager Implementation
```python
class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass

class ConfigManager:
    """Manages RAG configuration with validation and hot-reload."""

    _instance = None
    _config: Optional[RAGConfigSchema] = None
    _config_path: Optional[Path] = None
    _observers: List[Observer] = []
    _callbacks: List[Callable] = []

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict] = None,
        environment: Optional[str] = None
    ) -> None:
        """Load configuration from file or dict."""
        if config_path:
            self._config_path = Path(config_path)
            config_data = self._load_file(self._config_path)
        elif config_dict:
            config_data = config_dict
        else:
            config_data = self._get_default_config()

        # Apply environment-specific overrides
        env = environment or os.getenv("RAG_ENV", "development")
        config_data = self._apply_environment_overrides(config_data, env)

        # Validate and parse
        try:
            self._config = RAGConfigSchema(**config_data)
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration: {e}")

    def _load_file(self, file_path: Path) -> Dict:
        """Load configuration from file."""
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, 'r') as f:
                if file_path.suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                elif file_path.suffix == '.json':
                    return json.load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported file format: {file_path.suffix}"
                    )
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML: {e}")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON: {e}")

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "global_settings": {
                "environment": "development",
                "log_level": "INFO",
                "cache_enabled": True,
                "cache_ttl": 3600
            },
            "strategies": {},
            "pipeline": None
        }

    def _apply_environment_overrides(
        self,
        config: Dict,
        environment: str
    ) -> Dict:
        """Apply environment-specific configuration overrides."""
        # Look for environment-specific config file
        if self._config_path:
            env_config_path = (
                self._config_path.parent /
                f"{self._config_path.stem}.{environment}{self._config_path.suffix}"
            )
            if env_config_path.exists():
                env_config = self._load_file(env_config_path)
                config = self._deep_merge(config, env_config)

        return config

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        if self._config is None:
            raise ConfigurationError("Configuration not loaded")

        keys = key.split('.')
        value = self._config.dict()

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_strategy_config(self, strategy_name: str) -> StrategyConfigSchema:
        """Get configuration for a specific strategy."""
        if self._config is None:
            raise ConfigurationError("Configuration not loaded")

        if strategy_name in self._config.strategies:
            return self._config.strategies[strategy_name]
        else:
            # Return default config
            return StrategyConfigSchema(strategy_name=strategy_name)

    def enable_hot_reload(self, callback: Optional[Callable] = None) -> None:
        """Enable hot-reloading of configuration files."""
        if not self._config_path:
            raise ConfigurationError("Cannot hot-reload without config file path")

        if callback:
            self._callbacks.append(callback)

        handler = ConfigFileHandler(self)
        observer = Observer()
        observer.schedule(
            handler,
            str(self._config_path.parent),
            recursive=False
        )
        observer.start()
        self._observers.append(observer)

    def disable_hot_reload(self) -> None:
        """Disable hot-reloading."""
        for observer in self._observers:
            observer.stop()
            observer.join()
        self._observers.clear()

    def reload(self) -> None:
        """Reload configuration from file."""
        if self._config_path:
            self.load(str(self._config_path))
            # Notify callbacks
            for callback in self._callbacks:
                callback(self._config)

    @property
    def config(self) -> Optional[RAGConfigSchema]:
        """Get the current configuration."""
        return self._config

    def to_dict(self) -> Dict:
        """Export configuration as dictionary."""
        if self._config is None:
            raise ConfigurationError("Configuration not loaded")
        return self._config.dict()


class ConfigFileHandler(FileSystemEventHandler):
    """Handle configuration file changes for hot-reload."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    def on_modified(self, event):
        if not event.is_directory and event.src_path == str(self.config_manager._config_path):
            try:
                self.config_manager.reload()
                print(f"Configuration reloaded from {event.src_path}")
            except ConfigurationError as e:
                print(f"Failed to reload configuration: {e}")
```

### Example Configuration Files

**config.yaml** (base configuration):
```yaml
global_settings:
  environment: development
  log_level: INFO
  cache_enabled: true
  cache_ttl: 3600

strategies:
  reranking:
    chunk_size: 512
    chunk_overlap: 50
    top_k: 5
    metadata:
      model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"

  semantic_search:
    chunk_size: 256
    top_k: 10
    metadata:
      embedding_model: "all-MiniLM-L6-v2"

pipeline:
  mode: sequential
  timeout: 30
  stages:
    - strategy: semantic_search
      name: initial_retrieval
      config:
        top_k: 20
    - strategy: reranking
      name: rerank
      config:
        top_k: 5
```

**config.production.yaml** (production overrides):
```yaml
global_settings:
  environment: production
  log_level: WARNING
  cache_ttl: 7200

strategies:
  reranking:
    chunk_size: 1024
```

---

## Unit Tests

### Test File Location
`tests/unit/test_config.py`

### Test Cases

#### TC4.1: Configuration Loading Tests
```python
def test_load_from_yaml(tmp_path):
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

def test_load_from_json(tmp_path):
    """Test loading configuration from JSON file."""
    config_file = tmp_path / "config.json"
    config_file.write_text('{"global_settings": {"log_level": "ERROR"}}')

    config = ConfigManager()
    config.load(str(config_file))

    assert config.get("global_settings.log_level") == "ERROR"

def test_load_from_dict():
    """Test loading configuration from dictionary."""
    config_dict = {
        "global_settings": {"log_level": "INFO"},
        "strategies": {}
    }

    config = ConfigManager()
    config.load(config_dict=config_dict)

    assert config.get("global_settings.log_level") == "INFO"

def test_file_not_found_raises_error():
    """Test loading non-existent file raises error."""
    config = ConfigManager()

    with pytest.raises(ConfigurationError, match="not found"):
        config.load("/nonexistent/config.yaml")

def test_invalid_yaml_raises_error(tmp_path):
    """Test invalid YAML raises error."""
    config_file = tmp_path / "bad_config.yaml"
    config_file.write_text("invalid: yaml: content:")

    config = ConfigManager()

    with pytest.raises(ConfigurationError, match="Invalid YAML"):
        config.load(str(config_file))
```

#### TC4.2: Validation Tests
```python
def test_validation_enforces_schema():
    """Test configuration validated against schema."""
    config_dict = {
        "global_settings": {
            "log_level": "INVALID_LEVEL"  # Invalid enum value
        }
    }

    config = ConfigManager()
    # Should validate and potentially raise error or use default
    # Depending on schema strictness

def test_chunk_size_range_validation():
    """Test chunk_size must be within valid range."""
    config_dict = {
        "strategies": {
            "test": {
                "chunk_size": -1  # Invalid
            }
        }
    }

    config = ConfigManager()

    with pytest.raises(ConfigurationError):
        config.load(config_dict=config_dict)

def test_required_fields_validation():
    """Test required fields must be present."""
    # Test depends on which fields are actually required
    pass
```

#### TC4.3: Default Configuration Tests
```python
def test_default_config_loaded():
    """Test default configuration loaded when no config provided."""
    config = ConfigManager()
    config.load()

    assert config.get("global_settings.environment") == "development"
    assert config.get("global_settings.cache_enabled") is True

def test_user_config_merged_with_defaults():
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

def test_get_strategy_config_returns_default():
    """Test getting config for unregistered strategy returns defaults."""
    config = ConfigManager()
    config.load()

    strategy_config = config.get_strategy_config("nonexistent_strategy")

    assert strategy_config.chunk_size == 512  # Default value
```

#### TC4.4: Environment Configuration Tests
```python
def test_environment_from_env_variable(tmp_path, monkeypatch):
    """Test environment determined from environment variable."""
    base_config = tmp_path / "config.yaml"
    base_config.write_text("global_settings:\n  log_level: INFO")

    prod_config = tmp_path / "config.production.yaml"
    prod_config.write_text("global_settings:\n  log_level: ERROR")

    monkeypatch.setenv("RAG_ENV", "production")

    config = ConfigManager()
    config.load(str(base_config))

    assert config.get("global_settings.log_level") == "ERROR"

def test_environment_override_merges_correctly(tmp_path):
    """Test environment-specific config merged with base."""
    base_config = tmp_path / "config.yaml"
    base_config.write_text("""
    global_settings:
      log_level: INFO
      cache_enabled: true
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
```

#### TC4.5: Configuration Access Tests
```python
def test_get_with_dot_notation():
    """Test accessing nested config with dot notation."""
    config_dict = {
        "global_settings": {
            "cache_enabled": True
        }
    }

    config = ConfigManager()
    config.load(config_dict=config_dict)

    assert config.get("global_settings.cache_enabled") is True

def test_get_missing_key_returns_default():
    """Test getting non-existent key returns default."""
    config = ConfigManager()
    config.load()

    value = config.get("nonexistent.key", default="default_value")
    assert value == "default_value"

def test_get_strategy_config():
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
```

#### TC4.6: Hot-Reload Tests
```python
def test_hot_reload_detects_file_changes(tmp_path):
    """Test configuration reloaded when file changes."""
    import time

    config_file = tmp_path / "config.yaml"
    config_file.write_text("global_settings:\n  log_level: INFO")

    config = ConfigManager()
    config.load(str(config_file))

    callback_called = []

    def callback(new_config):
        callback_called.append(True)

    config.enable_hot_reload(callback)

    # Modify file
    time.sleep(0.1)
    config_file.write_text("global_settings:\n  log_level: DEBUG")
    time.sleep(0.5)  # Wait for file watcher

    assert len(callback_called) > 0
    assert config.get("global_settings.log_level") == "DEBUG"

    config.disable_hot_reload()

def test_disable_hot_reload():
    """Test hot-reload can be disabled."""
    # Implementation depends on hot-reload mechanism
    pass
```

---

## Integration Tests

### Test File Location
`tests/integration/test_config_integration.py`

### Test Scenarios

#### IS4.1: Full Configuration Lifecycle
```python
@pytest.mark.integration
def test_load_validate_use_config(tmp_path):
    """Test complete configuration lifecycle."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
    global_settings:
      log_level: INFO
      cache_enabled: true

    strategies:
      test_strategy:
        chunk_size: 512
        top_k: 5

    pipeline:
      mode: sequential
      stages:
        - strategy: test_strategy
          name: stage1
    """)

    # Load
    config = ConfigManager()
    config.load(str(config_file))

    # Access global settings
    assert config.get("global_settings.log_level") == "INFO"

    # Access strategy config
    strategy_config = config.get_strategy_config("test_strategy")
    assert strategy_config.chunk_size == 512

    # Access pipeline config
    pipeline_config = config.get("pipeline")
    assert pipeline_config["mode"] == "sequential"
```

#### IS4.2: Multi-Environment Configuration
```python
@pytest.mark.integration
def test_multi_environment_configuration(tmp_path):
    """Test configuration across multiple environments."""
    # Create base config
    base_config = tmp_path / "config.yaml"
    base_config.write_text("""
    global_settings:
      log_level: INFO
      cache_ttl: 3600
    strategies:
      strategy1:
        chunk_size: 512
    """)

    # Create dev override
    dev_config = tmp_path / "config.development.yaml"
    dev_config.write_text("""
    global_settings:
      log_level: DEBUG
    """)

    # Create prod override
    prod_config = tmp_path / "config.production.yaml"
    prod_config.write_text("""
    global_settings:
      log_level: ERROR
      cache_ttl: 7200
    strategies:
      strategy1:
        chunk_size: 1024
    """)

    # Test dev environment
    config = ConfigManager()
    config.load(str(base_config), environment="development")
    assert config.get("global_settings.log_level") == "DEBUG"
    assert config.get("global_settings.cache_ttl") == 3600  # From base

    # Test prod environment
    config_prod = ConfigManager()
    config_prod.load(str(base_config), environment="production")
    assert config_prod.get("global_settings.log_level") == "ERROR"
    assert config_prod.get("global_settings.cache_ttl") == 7200
    assert config_prod.get("strategies.strategy1.chunk_size") == 1024
```

#### IS4.3: Integration with Factory
```python
@pytest.mark.integration
def test_config_with_factory(tmp_path):
    """Test configuration used with RAGFactory."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
    strategies:
      test_strategy:
        chunk_size: 1024
        top_k: 10
        strategy_name: test_strategy
    """)

    # Load config
    config = ConfigManager()
    config.load(str(config_file))

    # Create strategy using config
    class TestStrategy(IRAGStrategy):
        def initialize(self, config):
            self.config = config

    RAGFactory.register_strategy("test_strategy", TestStrategy)

    strategy_config = config.get_strategy_config("test_strategy")
    strategy = RAGFactory.create_strategy(
        "test_strategy",
        strategy_config.dict()
    )

    assert strategy.config.chunk_size == 1024
```

---

## Definition of Done

- [ ] All code passes type checking with mypy
- [ ] All unit tests pass (>95% coverage of config.py)
- [ ] All integration tests pass
- [ ] Code reviewed by at least one team member
- [ ] Documentation complete with examples
- [ ] Example configuration files provided
- [ ] No linting errors
- [ ] Integration with other components verified
- [ ] Changes committed to feature branch

---

## Testing Checklist

### Unit Testing
- [ ] YAML file loading works
- [ ] JSON file loading works
- [ ] Dictionary loading works
- [ ] File not found handled
- [ ] Invalid syntax handled
- [ ] Schema validation works
- [ ] Default configuration provided
- [ ] Environment overrides work
- [ ] Dot notation access works
- [ ] Hot-reload works

### Integration Testing
- [ ] Full lifecycle test passes
- [ ] Multi-environment configuration works
- [ ] Integration with Factory works
- [ ] Integration with Pipeline works
- [ ] Real-world configuration scenarios work

### Code Quality
- [ ] Clear error messages
- [ ] Good documentation
- [ ] Thread-safe if needed
- [ ] No security vulnerabilities

---

## Notes for Developers

1. **Start with schema**: Define pydantic models first
2. **Test validation thoroughly**: Edge cases are important
3. **Document all options**: Configuration should be self-documenting
4. **Think about usability**: Make common cases easy
5. **Security first**: Be careful with sensitive data
6. **Performance matters**: Cache configuration access
7. **Hot-reload is optional**: Production shouldn't use it

### Recommended Implementation Order
1. Define configuration schema with pydantic
2. Implement basic file loading (YAML/JSON)
3. Add validation
4. Implement configuration access API
5. Add default configurations
6. Implement environment-specific overrides
7. Add hot-reload capability
8. Test integration with other components

### Configuration Best Practices
- Use environment variables for secrets
- Provide sensible defaults
- Validate early and fail fast
- Document all configuration options
- Version your configuration schema
- Keep production configs simple
- Test all environment configurations
