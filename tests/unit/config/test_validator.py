"""Unit tests for configuration validator."""

import pytest
import json
from pathlib import Path
from rag_factory.config.validator import (
    ConfigValidator,
    ConfigValidationError,
    load_yaml_with_validation,
)


@pytest.fixture
def validator():
    """Create a ConfigValidator instance."""
    return ConfigValidator()


class TestServiceRegistryValidation:
    """Test service registry validation."""

    def test_valid_service_registry(self, validator):
        """Test validation of valid service registry."""
        config = {
            "services": {
                "llm1": {
                    "name": "test-llm",
                    "type": "llm",
                    "url": "http://localhost:1234/v1",
                    "api_key": "${API_KEY}",
                    "model": "test-model"
                }
            }
        }

        warnings = validator.validate_services_yaml(config)
        assert isinstance(warnings, list)

    def test_valid_embedding_service(self, validator):
        """Test validation of embedding service."""
        config = {
            "services": {
                "embedding1": {
                    "name": "test-embedding",
                    "type": "embedding",
                    "provider": "onnx",
                    "model": "test-model",
                    "dimensions": 384
                }
            }
        }

        warnings = validator.validate_services_yaml(config)
        assert isinstance(warnings, list)

    def test_valid_database_service(self, validator):
        """Test validation of database service."""
        config = {
            "services": {
                "db1": {
                    "name": "test-db",
                    "type": "postgres",
                    "connection_string": "${DATABASE_URL}",
                    "pool_size": 10
                }
            }
        }

        warnings = validator.validate_services_yaml(config)
        assert isinstance(warnings, list)

    def test_invalid_service_registry_missing_required(self, validator):
        """Test validation fails with missing required field."""
        config = {
            "services": {
                "llm1": {
                    # Missing 'name' and 'type'
                    "url": "http://localhost:1234/v1"
                }
            }
        }

        with pytest.raises(ConfigValidationError) as exc_info:
            validator.validate_services_yaml(config)

        error_msg = str(exc_info.value).lower()
        assert "validation failed" in error_msg

    def test_invalid_service_type(self, validator):
        """Test validation fails with invalid service type."""
        config = {
            "services": {
                "llm1": {
                    "name": "test",
                    "type": "invalid_type"
                }
            }
        }

        with pytest.raises(ConfigValidationError):
            validator.validate_services_yaml(config)

    def test_invalid_embedding_provider(self, validator):
        """Test validation fails with invalid embedding provider."""
        config = {
            "services": {
                "embedding1": {
                    "name": "test",
                    "type": "embedding",
                    "provider": "invalid_provider"
                }
            }
        }

        with pytest.raises(ConfigValidationError):
            validator.validate_services_yaml(config)

    def test_plaintext_secret_warning(self, validator):
        """Test warning for plaintext secrets."""
        config = {
            "services": {
                "llm1": {
                    "name": "test",
                    "type": "llm",
                    "api_key": "sk-123456"  # Plaintext!
                }
            }
        }

        warnings = validator.validate_services_yaml(config)
        assert len(warnings) > 0
        assert any("plaintext" in w.lower() for w in warnings)

    def test_no_warning_for_env_var_secrets(self, validator):
        """Test no warning when using environment variables for secrets."""
        config = {
            "services": {
                "llm1": {
                    "name": "test",
                    "type": "llm",
                    "api_key": "${API_KEY}"
                }
            }
        }

        warnings = validator.validate_services_yaml(config)
        # Should not have plaintext secret warnings
        assert not any("plaintext" in w.lower() for w in warnings)


class TestStrategyPairValidation:
    """Test strategy pair validation."""

    def test_valid_strategy_pair(self, validator):
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
                "embedding1": {
                    "name": "test",
                    "type": "embedding",
                    "provider": "onnx"
                }
            }
        }

        warnings = validator.validate_strategy_pair_yaml(config, service_registry)
        assert isinstance(warnings, list)

    def test_strategy_pair_with_db_config(self, validator):
        """Test strategy pair with database configuration."""
        config = {
            "strategy_name": "test-pair",
            "version": "1.0.0",
            "indexer": {
                "strategy": "TestIndexer",
                "services": {
                    "embedding": "$embedding1"
                },
                "db_config": {
                    "tables": {
                        "chunks": "test_chunks"
                    },
                    "fields": {
                        "content": "chunk_content"
                    }
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
                "embedding1": {
                    "name": "test",
                    "type": "embedding",
                    "provider": "onnx"
                }
            }
        }

        warnings = validator.validate_strategy_pair_yaml(config, service_registry)
        assert isinstance(warnings, list)

    def test_strategy_pair_with_migrations(self, validator):
        """Test strategy pair with migration configuration."""
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
            },
            "migrations": {
                "required_revisions": [
                    "001_create_tables",
                    "002_add_indexes"
                ]
            }
        }

        service_registry = {
            "services": {
                "embedding1": {
                    "name": "test",
                    "type": "embedding",
                    "provider": "onnx"
                }
            }
        }

        warnings = validator.validate_strategy_pair_yaml(config, service_registry)
        assert isinstance(warnings, list)

    def test_invalid_strategy_name_format(self, validator):
        """Test validation fails with invalid strategy name format."""
        config = {
            "strategy_name": "Invalid_Name",  # Should be lowercase with hyphens
            "version": "1.0.0",
            "indexer": {
                "strategy": "TestIndexer",
                "services": {"embedding": "$embedding1"}
            },
            "retriever": {
                "strategy": "TestRetriever",
                "services": {"embedding": "$embedding1"}
            }
        }

        with pytest.raises(ConfigValidationError):
            validator.validate_strategy_pair_yaml(config)

    def test_invalid_version_format(self, validator):
        """Test validation fails with invalid version format."""
        config = {
            "strategy_name": "test-pair",
            "version": "1.0",  # Should be semver (1.0.0)
            "indexer": {
                "strategy": "TestIndexer",
                "services": {"embedding": "$embedding1"}
            },
            "retriever": {
                "strategy": "TestRetriever",
                "services": {"embedding": "$embedding1"}
            }
        }

        with pytest.raises(ConfigValidationError):
            validator.validate_strategy_pair_yaml(config)

    def test_missing_required_sections(self, validator):
        """Test validation fails with missing required sections."""
        config = {
            "strategy_name": "test-pair",
            "version": "1.0.0",
            # Missing indexer and retriever
        }

        with pytest.raises(ConfigValidationError):
            validator.validate_strategy_pair_yaml(config)


class TestServiceReferenceValidation:
    """Test service reference validation."""

    def test_invalid_service_reference(self, validator):
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
                "embedding1": {
                    "name": "test",
                    "type": "embedding",
                    "provider": "onnx"
                }
            }
        }

        with pytest.raises(ConfigValidationError) as exc_info:
            validator.validate_strategy_pair_yaml(config, service_registry)

        assert "nonexistent" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()

    def test_inline_service_config(self, validator):
        """Test inline service configuration (not a reference)."""
        config = {
            "strategy_name": "test-pair",
            "version": "1.0.0",
            "indexer": {
                "strategy": "TestIndexer",
                "services": {
                    "embedding": {
                        "name": "inline-embedding",
                        "type": "embedding",
                        "provider": "onnx",
                        "model": "test-model"
                    }
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
                "embedding1": {
                    "name": "test",
                    "type": "embedding",
                    "provider": "onnx"
                }
            }
        }

        # Should not raise error for inline config
        warnings = validator.validate_strategy_pair_yaml(config, service_registry)
        assert isinstance(warnings, list)

    def test_validation_without_service_registry(self, validator):
        """Test validation works without service registry (no reference checking)."""
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

        # Should not raise error when service_registry is None
        warnings = validator.validate_strategy_pair_yaml(config, service_registry=None)
        assert isinstance(warnings, list)


class TestConfigValidationError:
    """Test ConfigValidationError formatting."""

    def test_error_with_all_fields(self):
        """Test error formatting with all fields."""
        error = ConfigValidationError(
            message="Test error",
            file_path="/path/to/config.yaml",
            field="services.llm1.api_key"
        )

        error_str = str(error)
        assert "Test error" in error_str
        assert "/path/to/config.yaml" in error_str
        assert "services.llm1.api_key" in error_str

    def test_error_with_message_only(self):
        """Test error formatting with message only."""
        error = ConfigValidationError(message="Test error")

        error_str = str(error)
        assert "Test error" in error_str
        assert "File:" not in error_str
        assert "Field:" not in error_str


class TestLoadYamlWithValidation:
    """Test YAML loading and validation."""

    def test_load_valid_services_yaml(self, tmp_path):
        """Test loading valid services YAML file."""
        services_yaml = tmp_path / "services.yaml"
        services_yaml.write_text("""
services:
  embedding1:
    name: "test-embedding"
    type: "embedding"
    provider: "onnx"
    model: "test-model"
""")

        config = load_yaml_with_validation(
            str(services_yaml),
            config_type="service_registry"
        )

        assert "services" in config
        assert "embedding1" in config["services"]

    def test_load_valid_strategy_pair_yaml(self, tmp_path):
        """Test loading valid strategy pair YAML file."""
        strategy_yaml = tmp_path / "test-pair.yaml"
        strategy_yaml.write_text("""
strategy_name: "test-pair"
version: "1.0.0"

indexer:
  strategy: "TestIndexer"
  services:
    embedding: "$embedding1"

retriever:
  strategy: "TestRetriever"
  services:
    embedding: "$embedding1"
""")

        service_registry = {
            "services": {
                "embedding1": {
                    "name": "test",
                    "type": "embedding",
                    "provider": "onnx"
                }
            }
        }

        config = load_yaml_with_validation(
            str(strategy_yaml),
            config_type="strategy_pair",
            service_registry=service_registry
        )

        assert config["strategy_name"] == "test-pair"

    def test_load_invalid_config_type(self, tmp_path):
        """Test error with invalid config type."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("test: value")

        with pytest.raises(ValueError) as exc_info:
            load_yaml_with_validation(
                str(yaml_file),
                config_type="invalid_type"
            )

        assert "Unknown config type" in str(exc_info.value)

    def test_load_nonexistent_file(self):
        """Test error with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_yaml_with_validation(
                "/nonexistent/file.yaml",
                config_type="service_registry"
            )
