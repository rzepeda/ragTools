# Story 17.1: Design Service Registry and Configuration Schema

**Story ID:** 17.1
**Epic:** Epic 17 - Strategy Pair Configuration System
**Story Points:** 8
**Priority:** High
**Dependencies:** Epic 11 (Dependency Injection), Epic 12 (Pipeline Separation), Epic 16 (Database Consolidation)

---

## User Story

**As a** developer
**I want** well-defined YAML schemas for services and strategy pairs
**So that** configurations are consistent, validatable, and support environment variables

---

## Detailed Requirements

### Functional Requirements

1. **Service Registry YAML Schema**
   - Define services.yaml format for LLM, embedding, and database services
   - Support service naming with unique identifiers
   - Include all required service parameters (URL, API keys, models, etc.)
   - Support optional parameters with defaults
   - Allow service-specific configuration (provider-dependent)
   - Human-readable and self-documenting structure
   - Versioning support for backward compatibility

2. **Strategy Pair YAML Schema**
   - Define strategy-pair.yaml format for indexing/retrieval pairs
   - Reference services from service registry using `$service_name` syntax
   - Support inline service configuration as alternative
   - Include database table and field mapping configuration
   - Strategy-specific configuration parameters
   - Migration references (Alembic revision IDs)
   - Expected schema validation
   - Optional tags for discovery and filtering

3. **Environment Variable Resolution**
   - Support `${VAR_NAME}` syntax for environment variables
   - Support `${VAR_NAME:-default}` for optional variables with defaults
   - Support `${VAR_NAME:?error message}` for required variables with custom errors
   - Recursive variable resolution in nested structures
   - Clear error messages for missing variables
   - Security: never log resolved secrets

4. **Service Reference Validation**
   - Validate `$service_name` references exist in registry
   - Check service types match strategy requirements
   - Ensure referenced services are compatible
   - Detect circular dependencies
   - Validate inline service configurations

5. **Schema Validation Utilities**
   - JSON Schema definitions for both YAML formats
   - Validation function for services.yaml
   - Validation function for strategy-pair.yaml
   - Clear, actionable error messages
   - Warnings for deprecated configurations
   - Schema version checking

6. **Configuration Documentation**
   - Comprehensive documentation with examples
   - All configuration options documented
   - Best practices guide
   - Migration guide from manual configuration
   - Troubleshooting common issues
   - Real-world configuration examples

### Non-Functional Requirements

1. **Usability**
   - YAML format easy to read and write by hand
   - Sensible defaults minimize required configuration
   - Self-documenting with clear parameter names
   - Examples provided for all service types
   - IDE support (YAML schema for autocomplete)

2. **Validation Performance**
   - Schema validation <100ms for typical configurations
   - Environment variable resolution <50ms
   - Service reference validation <50ms
   - Lazy loading of schemas (only validate when needed)

3. **Error Handling**
   - Validation errors point to exact line/field in YAML
   - Suggest corrections for common mistakes
   - Distinguish between errors and warnings
   - No crashes on malformed YAML (graceful degradation)

4. **Maintainability**
   - Schema definitions separate from validation code
   - Easy to add new service types
   - Easy to add new configuration parameters
   - Version migration path documented
   - Schema evolution strategy defined

5. **Security**
   - Environment variables for all secrets
   - No secrets in configuration files
   - Warn if API keys detected in plaintext
   - Validate environment variable names (no injection)

---

## Acceptance Criteria

### AC1: Service Registry Schema
- [ ] Services.yaml JSON Schema defined
- [ ] Supports LLM service configuration (URL, API key, model, etc.)
- [ ] Supports embedding service configuration (provider, model, dimensions)
- [ ] Supports database service configuration (connection string, pooling)
- [ ] Environment variable syntax supported
- [ ] Schema validation function implemented
- [ ] Example services.yaml files created

### AC2: Strategy Pair Schema
- [ ] Strategy-pair.yaml JSON Schema defined
- [ ] Service reference syntax `$service_name` supported
- [ ] Inline service configuration supported
- [ ] Database table/field mapping configuration defined
- [ ] Strategy-specific config section defined
- [ ] Migration references section defined
- [ ] Expected schema validation section defined
- [ ] Tags for filtering supported
- [ ] Schema validation function implemented
- [ ] Example strategy-pair.yaml files created

### AC3: Environment Variable Resolution
- [ ] `${VAR}` syntax resolves from environment
- [ ] `${VAR:-default}` provides default values
- [ ] `${VAR:?error}` provides custom error messages
- [ ] Recursive resolution works in nested structures
- [ ] Missing required variables throw clear errors
- [ ] Validation prevents variable injection attacks

### AC4: Service Reference Validation
- [ ] References to non-existent services detected
- [ ] Service type compatibility checked
- [ ] Circular dependencies detected
- [ ] Inline service configs validated
- [ ] Clear error messages for reference issues

### AC5: Validation Utilities
- [ ] ConfigValidator class implemented
- [ ] validate_services_yaml() function working
- [ ] validate_strategy_pair_yaml() function working
- [ ] JSON Schema validation integrated
- [ ] Error messages include file location and field name
- [ ] Warnings generated for deprecated configs

### AC6: Documentation
- [ ] Complete schema reference documentation
- [ ] Example configurations for all service types
- [ ] Example configurations for all strategies
- [ ] Best practices documented
- [ ] Troubleshooting guide created
- [ ] Migration guide from manual config

### AC7: Testing
- [ ] Unit tests for schema validation (valid cases)
- [ ] Unit tests for schema validation (invalid cases)
- [ ] Unit tests for environment variable resolution
- [ ] Unit tests for service reference validation
- [ ] Integration tests with real configuration files
- [ ] Performance benchmarks for validation

---

## Technical Specifications

### File Structure
```
rag_factory/
├── config/
│   ├── __init__.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── service_registry_schema.json    # JSON Schema for services.yaml
│   │   ├── strategy_pair_schema.json       # JSON Schema for strategy-pair.yaml
│   │   └── version.py                      # Schema version tracking
│   ├── validator.py                         # ConfigValidator class
│   ├── env_resolver.py                      # Environment variable resolution
│   └── examples/
│       ├── services.yaml
│       ├── semantic-local-pair.yaml
│       └── README.md

docs/
├── configuration/
│   ├── service-registry-schema.md
│   ├── strategy-pair-schema.md
│   ├── environment-variables.md
│   └── examples/
│       └── *.yaml

tests/
├── unit/
│   └── config/
│       ├── test_validator.py
│       ├── test_env_resolver.py
│       └── test_schemas.py
```

### Dependencies
```python
# requirements.txt additions
pyyaml>=6.0           # YAML parsing
jsonschema>=4.0       # Schema validation
python-dotenv>=1.0    # .env file loading
```

### Service Registry Schema (JSON Schema)
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Service Registry Configuration",
  "type": "object",
  "required": ["services"],
  "properties": {
    "version": {
      "type": "string",
      "pattern": "^\\d+\\.\\d+\\.\\d+$",
      "description": "Schema version (semver)"
    },
    "services": {
      "type": "object",
      "description": "Service definitions",
      "patternProperties": {
        "^[a-zA-Z0-9_]+$": {
          "oneOf": [
            { "$ref": "#/definitions/llm_service" },
            { "$ref": "#/definitions/embedding_service" },
            { "$ref": "#/definitions/database_service" }
          ]
        }
      },
      "additionalProperties": false
    }
  },
  "definitions": {
    "llm_service": {
      "type": "object",
      "required": ["name"],
      "properties": {
        "name": { "type": "string", "description": "Human-readable service name" },
        "url": { "type": "string", "format": "uri", "description": "LLM API endpoint" },
        "api_key": { "type": "string", "description": "API key (use ${ENV_VAR})" },
        "model": { "type": "string", "description": "Model identifier" },
        "temperature": { "type": "number", "minimum": 0, "maximum": 2, "default": 0.7 },
        "max_tokens": { "type": "integer", "minimum": 1 },
        "timeout": { "type": "integer", "minimum": 1, "default": 30 }
      }
    },
    "embedding_service": {
      "type": "object",
      "required": ["name", "provider"],
      "properties": {
        "name": { "type": "string" },
        "provider": {
          "type": "string",
          "enum": ["onnx", "openai", "cohere", "huggingface"]
        },
        "model": { "type": "string", "description": "Model name/path" },
        "cache_dir": { "type": "string", "description": "Model cache directory" },
        "batch_size": { "type": "integer", "minimum": 1, "default": 32 },
        "dimensions": { "type": "integer", "minimum": 1, "description": "Embedding dimensions" },
        "api_key": { "type": "string", "description": "API key for cloud providers" }
      }
    },
    "database_service": {
      "type": "object",
      "required": ["name", "type"],
      "properties": {
        "name": { "type": "string" },
        "type": {
          "type": "string",
          "enum": ["postgres", "neo4j", "mongodb"]
        },
        "connection_string": { "type": "string", "description": "Full connection string" },
        "host": { "type": "string" },
        "port": { "type": "integer" },
        "database": { "type": "string" },
        "user": { "type": "string" },
        "password": { "type": "string", "description": "Use ${ENV_VAR}" },
        "pool_size": { "type": "integer", "minimum": 1, "default": 10 },
        "max_overflow": { "type": "integer", "minimum": 0, "default": 20 }
      }
    }
  }
}
```

### Strategy Pair Schema (JSON Schema)
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Strategy Pair Configuration",
  "type": "object",
  "required": ["strategy_name", "version", "indexer", "retriever"],
  "properties": {
    "strategy_name": {
      "type": "string",
      "pattern": "^[a-z0-9-]+$",
      "description": "Unique strategy pair identifier"
    },
    "version": {
      "type": "string",
      "pattern": "^\\d+\\.\\d+\\.\\d+$",
      "description": "Configuration version (semver)"
    },
    "description": {
      "type": "string",
      "description": "Human-readable description"
    },
    "indexer": { "$ref": "#/definitions/strategy_config" },
    "retriever": { "$ref": "#/definitions/strategy_config" },
    "migrations": {
      "type": "object",
      "properties": {
        "required_revisions": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Alembic revision IDs required"
        }
      }
    },
    "expected_schema": {
      "type": "object",
      "properties": {
        "tables": {
          "type": "array",
          "items": { "type": "string" }
        },
        "indexes": {
          "type": "array",
          "items": { "type": "string" }
        },
        "extensions": {
          "type": "array",
          "items": { "type": "string" }
        }
      }
    },
    "tags": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Tags for discovery/filtering"
    }
  },
  "definitions": {
    "strategy_config": {
      "type": "object",
      "required": ["strategy", "services"],
      "properties": {
        "strategy": {
          "type": "string",
          "description": "Strategy class name"
        },
        "services": {
          "type": "object",
          "description": "Service references or inline configs",
          "patternProperties": {
            "^[a-zA-Z_]+$": {
              "oneOf": [
                {
                  "type": "string",
                  "pattern": "^\\$[a-zA-Z0-9_]+$",
                  "description": "Service reference"
                },
                {
                  "type": "object",
                  "description": "Inline service config"
                }
              ]
            }
          }
        },
        "db_config": {
          "type": "object",
          "properties": {
            "tables": {
              "type": "object",
              "description": "Logical to physical table mapping",
              "patternProperties": {
                "^[a-zA-Z_]+$": { "type": "string" }
              }
            },
            "fields": {
              "type": "object",
              "description": "Logical to physical field mapping",
              "patternProperties": {
                "^[a-zA-Z_]+$": { "type": "string" }
              }
            }
          }
        },
        "config": {
          "type": "object",
          "description": "Strategy-specific configuration",
          "additionalProperties": true
        }
      }
    }
  }
}
```

### Configuration Validator Implementation
```python
# rag_factory/config/validator.py
from typing import Dict, Any, List
import yaml
import jsonschema
import os
from pathlib import Path

class ConfigValidationError(Exception):
    """Configuration validation error with detailed context."""

    def __init__(self, message: str, file_path: str = None, field: str = None):
        self.message = message
        self.file_path = file_path
        self.field = field
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [self.message]
        if self.file_path:
            parts.append(f"File: {self.file_path}")
        if self.field:
            parts.append(f"Field: {self.field}")
        return "\n".join(parts)


class ConfigValidator:
    """Validates service registry and strategy pair configurations."""

    def __init__(self, schemas_dir: str = None):
        """
        Initialize validator with JSON schemas.

        Args:
            schemas_dir: Directory containing JSON schema files
        """
        if schemas_dir is None:
            schemas_dir = Path(__file__).parent / "schemas"

        self.schemas_dir = Path(schemas_dir)
        self._schemas = {}
        self._load_schemas()

    def _load_schemas(self):
        """Load JSON schemas from files."""
        schema_files = {
            "service_registry": "service_registry_schema.json",
            "strategy_pair": "strategy_pair_schema.json"
        }

        for name, filename in schema_files.items():
            schema_path = self.schemas_dir / filename
            with open(schema_path, 'r') as f:
                import json
                self._schemas[name] = json.load(f)

    def validate_services_yaml(
        self,
        config: Dict[str, Any],
        file_path: str = None
    ) -> List[str]:
        """
        Validate services.yaml configuration.

        Args:
            config: Parsed YAML configuration
            file_path: Path to YAML file (for error messages)

        Returns:
            List of warning messages (empty if no warnings)

        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            # JSON Schema validation
            jsonschema.validate(
                instance=config,
                schema=self._schemas["service_registry"]
            )
        except jsonschema.ValidationError as e:
            raise ConfigValidationError(
                message=f"Schema validation failed: {e.message}",
                file_path=file_path,
                field=".".join(str(p) for p in e.path)
            )

        warnings = []

        # Check for plaintext secrets
        warnings.extend(self._check_for_plaintext_secrets(config, file_path))

        # Check for deprecated configurations
        warnings.extend(self._check_deprecated_configs(config))

        return warnings

    def validate_strategy_pair_yaml(
        self,
        config: Dict[str, Any],
        service_registry: Dict[str, Any] = None,
        file_path: str = None
    ) -> List[str]:
        """
        Validate strategy-pair.yaml configuration.

        Args:
            config: Parsed YAML configuration
            service_registry: Service registry for reference validation
            file_path: Path to YAML file (for error messages)

        Returns:
            List of warning messages

        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            # JSON Schema validation
            jsonschema.validate(
                instance=config,
                schema=self._schemas["strategy_pair"]
            )
        except jsonschema.ValidationError as e:
            raise ConfigValidationError(
                message=f"Schema validation failed: {e.message}",
                file_path=file_path,
                field=".".join(str(p) for p in e.path)
            )

        warnings = []

        # Validate service references if registry provided
        if service_registry:
            warnings.extend(
                self._validate_service_references(config, service_registry, file_path)
            )

        return warnings

    def _check_for_plaintext_secrets(
        self,
        config: Dict[str, Any],
        file_path: str
    ) -> List[str]:
        """Check for potential plaintext secrets in configuration."""
        warnings = []
        secret_fields = ["api_key", "password", "secret", "token"]

        def check_dict(d: dict, path: str = ""):
            for key, value in d.items():
                current_path = f"{path}.{key}" if path else key

                if isinstance(value, dict):
                    check_dict(value, current_path)
                elif isinstance(value, str):
                    # Check if this is a secret field
                    if any(field in key.lower() for field in secret_fields):
                        # Check if it uses environment variable syntax
                        if not value.startswith("${"):
                            warnings.append(
                                f"WARNING: Potential plaintext secret in {current_path}. "
                                f"Consider using environment variable: ${{ENV_VAR}}"
                            )

        check_dict(config)
        return warnings

    def _check_deprecated_configs(self, config: Dict[str, Any]) -> List[str]:
        """Check for deprecated configuration options."""
        warnings = []
        # Placeholder for future deprecations
        return warnings

    def _validate_service_references(
        self,
        config: Dict[str, Any],
        service_registry: Dict[str, Any],
        file_path: str
    ) -> List[str]:
        """
        Validate that service references exist in registry.

        Args:
            config: Strategy pair configuration
            service_registry: Service registry configuration
            file_path: Path to configuration file

        Returns:
            List of warnings

        Raises:
            ConfigValidationError: If referenced service doesn't exist
        """
        warnings = []
        available_services = set(service_registry.get("services", {}).keys())

        # Check indexer and retriever services
        for component in ["indexer", "retriever"]:
            if component not in config:
                continue

            services = config[component].get("services", {})
            for service_type, service_ref in services.items():
                # Check if it's a reference (starts with $)
                if isinstance(service_ref, str) and service_ref.startswith("$"):
                    service_name = service_ref[1:]  # Remove $

                    if service_name not in available_services:
                        raise ConfigValidationError(
                            message=(
                                f"Service reference '${service_name}' not found in registry. "
                                f"Available services: {sorted(available_services)}"
                            ),
                            file_path=file_path,
                            field=f"{component}.services.{service_type}"
                        )

        return warnings


def load_yaml_with_validation(
    file_path: str,
    config_type: str,
    service_registry: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Load and validate YAML configuration file.

    Args:
        file_path: Path to YAML file
        config_type: "service_registry" or "strategy_pair"
        service_registry: Service registry for strategy pair validation

    Returns:
        Validated configuration dictionary

    Raises:
        ConfigValidationError: If validation fails
    """
    # Load YAML
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate
    validator = ConfigValidator()

    if config_type == "service_registry":
        warnings = validator.validate_services_yaml(config, file_path)
    elif config_type == "strategy_pair":
        warnings = validator.validate_strategy_pair_yaml(
            config,
            service_registry=service_registry,
            file_path=file_path
        )
    else:
        raise ValueError(f"Unknown config type: {config_type}")

    # Print warnings
    for warning in warnings:
        print(f"WARNING: {warning}")

    return config
```

### Environment Variable Resolver
```python
# rag_factory/config/env_resolver.py
import os
import re
from typing import Any, Dict

class EnvironmentVariableError(Exception):
    """Error resolving environment variable."""
    pass


class EnvResolver:
    """Resolves environment variables in configuration."""

    # Pattern: ${VAR}, ${VAR:-default}, ${VAR:?error}
    ENV_VAR_PATTERN = re.compile(r'\$\{([^}:]+)(?::-([^}]+)|:\?([^}]+))?\}')

    @classmethod
    def resolve(cls, value: Any) -> Any:
        """
        Recursively resolve environment variables in value.

        Args:
            value: Value to resolve (can be str, dict, list, etc.)

        Returns:
            Value with environment variables resolved

        Raises:
            EnvironmentVariableError: If required variable is missing
        """
        if isinstance(value, str):
            return cls._resolve_string(value)
        elif isinstance(value, dict):
            return {k: cls.resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [cls.resolve(item) for item in value]
        else:
            return value

    @classmethod
    def _resolve_string(cls, value: str) -> str:
        """Resolve environment variables in a string."""
        def replacer(match):
            var_name = match.group(1)
            default = match.group(2)
            error_msg = match.group(3)

            env_value = os.getenv(var_name)

            if env_value is not None:
                return env_value
            elif default is not None:
                return default
            elif error_msg is not None:
                raise EnvironmentVariableError(
                    f"Environment variable ${{{var_name}}}: {error_msg}"
                )
            else:
                raise EnvironmentVariableError(
                    f"Required environment variable ${{{var_name}}} is not set"
                )

        return cls.ENV_VAR_PATTERN.sub(replacer, value)
```

---

## Unit Tests

### Test File Locations
- `tests/unit/config/test_validator.py`
- `tests/unit/config/test_env_resolver.py`
- `tests/unit/config/test_schemas.py`

### Test Cases

#### TC17.1.1: Schema Validation Tests
```python
import pytest
from rag_factory.config.validator import ConfigValidator, ConfigValidationError

@pytest.fixture
def validator():
    return ConfigValidator()

def test_valid_service_registry(validator):
    """Test validation of valid service registry."""
    config = {
        "services": {
            "llm1": {
                "name": "test-llm",
                "url": "http://localhost:1234/v1",
                "api_key": "${API_KEY}",
                "model": "test-model"
            }
        }
    }

    warnings = validator.validate_services_yaml(config)
    assert isinstance(warnings, list)

def test_invalid_service_registry_missing_required(validator):
    """Test validation fails with missing required field."""
    config = {
        "services": {
            "llm1": {
                # Missing 'name'
                "url": "http://localhost:1234/v1"
            }
        }
    }

    with pytest.raises(ConfigValidationError) as exc_info:
        validator.validate_services_yaml(config)

    assert "name" in str(exc_info.value).lower()

def test_plaintext_secret_warning(validator):
    """Test warning for plaintext secrets."""
    config = {
        "services": {
            "llm1": {
                "name": "test",
                "api_key": "sk-123456"  # Plaintext!
            }
        }
    }

    warnings = validator.validate_services_yaml(config)
    assert len(warnings) > 0
    assert "plaintext" in warnings[0].lower()

def test_valid_strategy_pair(validator):
    """Test validation of valid strategy pair."""
    config = {
        "strategy_name": "test-pair",
        "version": "1.0.0",
        "indexer": {
            "strategy": "TestIndexer",
            "services": {
                "embedding": "$embedding1"
            }
        },
        "retriever": {
            "strategy": "TestRetriever",
            "services": {
                "embedding": "$embedding1"
            }
        }
    }

    service_registry = {
        "services": {
            "embedding1": {"name": "test", "provider": "onnx"}
        }
    }

    warnings = validator.validate_strategy_pair_yaml(config, service_registry)
    assert isinstance(warnings, list)

def test_invalid_service_reference(validator):
    """Test validation fails with invalid service reference."""
    config = {
        "strategy_name": "test-pair",
        "version": "1.0.0",
        "indexer": {
            "strategy": "TestIndexer",
            "services": {
                "embedding": "$nonexistent"  # Doesn't exist!
            }
        },
        "retriever": {
            "strategy": "TestRetriever",
            "services": {
                "embedding": "$embedding1"
            }
        }
    }

    service_registry = {
        "services": {
            "embedding1": {"name": "test"}
        }
    }

    with pytest.raises(ConfigValidationError) as exc_info:
        validator.validate_strategy_pair_yaml(config, service_registry)

    assert "nonexistent" in str(exc_info.value)
```

#### TC17.1.2: Environment Variable Resolution Tests
```python
import pytest
import os
from rag_factory.config.env_resolver import EnvResolver, EnvironmentVariableError

def test_resolve_simple_variable():
    """Test resolving simple environment variable."""
    os.environ["TEST_VAR"] = "test_value"

    result = EnvResolver.resolve("${TEST_VAR}")
    assert result == "test_value"

    del os.environ["TEST_VAR"]

def test_resolve_with_default():
    """Test resolving with default value."""
    # Variable doesn't exist
    result = EnvResolver.resolve("${NONEXISTENT:-default_value}")
    assert result == "default_value"

def test_resolve_required_missing():
    """Test error for missing required variable."""
    with pytest.raises(EnvironmentVariableError) as exc_info:
        EnvResolver.resolve("${REQUIRED_VAR}")

    assert "REQUIRED_VAR" in str(exc_info.value)

def test_resolve_with_custom_error():
    """Test custom error message."""
    with pytest.raises(EnvironmentVariableError) as exc_info:
        EnvResolver.resolve("${REQUIRED_VAR:?Custom error message}")

    assert "Custom error message" in str(exc_info.value)

def test_resolve_in_dict():
    """Test recursive resolution in dictionary."""
    os.environ["KEY1"] = "value1"
    os.environ["KEY2"] = "value2"

    config = {
        "field1": "${KEY1}",
        "field2": "${KEY2}",
        "field3": "normal_value",
        "nested": {
            "field4": "${KEY1}"
        }
    }

    result = EnvResolver.resolve(config)

    assert result["field1"] == "value1"
    assert result["field2"] == "value2"
    assert result["field3"] == "normal_value"
    assert result["nested"]["field4"] == "value1"

    del os.environ["KEY1"]
    del os.environ["KEY2"]

def test_resolve_in_list():
    """Test resolution in list."""
    os.environ["ITEM"] = "item_value"

    config = ["${ITEM}", "normal", "${ITEM}"]
    result = EnvResolver.resolve(config)

    assert result == ["item_value", "normal", "item_value"]

    del os.environ["ITEM"]

def test_partial_string_replacement():
    """Test partial string replacement."""
    os.environ["HOST"] = "localhost"
    os.environ["PORT"] = "5432"

    connection_string = "postgresql://user:pass@${HOST}:${PORT}/db"
    result = EnvResolver.resolve(connection_string)

    assert result == "postgresql://user:pass@localhost:5432/db"

    del os.environ["HOST"]
    del os.environ["PORT"]
```

---

## Integration Tests

```python
# tests/integration/config/test_config_integration.py

@pytest.mark.integration
def test_load_real_services_yaml(tmp_path):
    """Test loading real services.yaml file."""
    services_yaml = tmp_path / "services.yaml"
    services_yaml.write_text("""
services:
  embedding_local:
    name: "local-onnx-minilm"
    provider: "onnx"
    model: "Xenova/all-MiniLM-L6-v2"
    cache_dir: "./models/embeddings"
    batch_size: 32
    dimensions: 384

  db_main:
    name: "main-postgres"
    type: "postgres"
    connection_string: "${DATABASE_URL}"
    pool_size: 10
    max_overflow: 20
""")

    os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost:5432/test"

    from rag_factory.config.validator import load_yaml_with_validation
    from rag_factory.config.env_resolver import EnvResolver

    # Load and validate
    config = load_yaml_with_validation(str(services_yaml), "service_registry")

    # Resolve environment variables
    config = EnvResolver.resolve(config)

    assert "services" in config
    assert "embedding_local" in config["services"]
    assert config["services"]["db_main"]["connection_string"] == "postgresql://user:pass@localhost:5432/test"

    del os.environ["DATABASE_URL"]

@pytest.mark.integration
def test_load_real_strategy_pair_yaml(tmp_path):
    """Test loading real strategy pair YAML."""
    # Create services file
    services_yaml = tmp_path / "services.yaml"
    services_yaml.write_text("""
services:
  embedding1:
    name: "test-embedding"
    provider: "onnx"
    model: "test-model"
""")

    # Create strategy pair file
    strategy_yaml = tmp_path / "test-pair.yaml"
    strategy_yaml.write_text("""
strategy_name: "test-pair"
version: "1.0.0"
description: "Test strategy pair"

indexer:
  strategy: "TestIndexer"
  services:
    embedding: "$embedding1"
  db_config:
    tables:
      chunks: "test_chunks"
  config:
    chunk_size: 512

retriever:
  strategy: "TestRetriever"
  services:
    embedding: "$embedding1"
  config:
    top_k: 5
""")

    from rag_factory.config.validator import load_yaml_with_validation
    import yaml

    # Load service registry
    with open(services_yaml) as f:
        service_registry = yaml.safe_load(f)

    # Load and validate strategy pair
    config = load_yaml_with_validation(
        str(strategy_yaml),
        "strategy_pair",
        service_registry=service_registry
    )

    assert config["strategy_name"] == "test-pair"
    assert config["indexer"]["services"]["embedding"] == "$embedding1"
```

---

## Definition of Done

- [ ] Service registry JSON Schema defined and saved
- [ ] Strategy pair JSON Schema defined and saved
- [ ] ConfigValidator class implemented
- [ ] EnvResolver class implemented
- [ ] Environment variable syntax supported (${VAR}, ${VAR:-default}, ${VAR:?error})
- [ ] Service reference validation working
- [ ] Plaintext secret detection working
- [ ] All unit tests pass (>90% coverage)
- [ ] All integration tests pass
- [ ] Example services.yaml created
- [ ] Example strategy-pair.yaml files created
- [ ] Schema documentation written
- [ ] Configuration best practices documented
- [ ] Troubleshooting guide created
- [ ] Code reviewed
- [ ] No linting errors

---

## Setup Instructions

### Installation

```bash
# Install dependencies
pip install pyyaml jsonschema python-dotenv
```

### Usage Example

```python
from rag_factory.config.validator import load_yaml_with_validation
from rag_factory.config.env_resolver import EnvResolver

# Load service registry
services_config = load_yaml_with_validation(
    "config/services.yaml",
    config_type="service_registry"
)

# Resolve environment variables
services_config = EnvResolver.resolve(services_config)

# Load strategy pair
strategy_config = load_yaml_with_validation(
    "strategies/semantic-local-pair.yaml",
    config_type="strategy_pair",
    service_registry=services_config
)

strategy_config = EnvResolver.resolve(strategy_config)
```

---

## Notes for Developers

1. **Schema Evolution**: When adding new fields, make them optional initially for backward compatibility.

2. **Error Messages**: Always include file path and field name in validation errors for easy debugging.

3. **Environment Variables**: Document all required environment variables in README or .env.example.

4. **Security**: Never log resolved secret values. Log variable names only.

5. **Validation Performance**: Lazy-load schemas. Don't validate on every access.

6. **IDE Support**: Provide YAML schema files for IDE autocomplete support.

7. **Migration Path**: Provide scripts to migrate from manual configuration to YAML.

8. **Testing**: Test with real YAML files, not just in-memory dictionaries.

9. **Documentation**: Keep schema documentation in sync with JSON Schema files.

10. **Defaults**: Use sensible defaults to minimize required configuration.
