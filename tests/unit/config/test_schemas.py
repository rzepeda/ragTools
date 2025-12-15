"""Unit tests for JSON schemas."""

import pytest
import json
import jsonschema
from pathlib import Path


@pytest.fixture
def schemas_dir():
    """Get the schemas directory."""
    return Path(__file__).parent.parent.parent.parent / "rag_factory" / "config" / "schemas"


@pytest.fixture
def service_registry_schema(schemas_dir):
    """Load service registry schema."""
    schema_path = schemas_dir / "service_registry_schema.json"
    with open(schema_path) as f:
        return json.load(f)


@pytest.fixture
def strategy_pair_schema(schemas_dir):
    """Load strategy pair schema."""
    schema_path = schemas_dir / "strategy_pair_schema.json"
    with open(schema_path) as f:
        return json.load(f)


class TestServiceRegistrySchema:
    """Test service registry JSON schema."""

    def test_schema_is_valid_json_schema(self, service_registry_schema):
        """Test that the schema itself is valid JSON Schema."""
        # This will raise an exception if the schema is invalid
        jsonschema.Draft7Validator.check_schema(service_registry_schema)

    def test_minimal_valid_config(self, service_registry_schema):
        """Test minimal valid configuration."""
        config = {
            "services": {}
        }
        jsonschema.validate(instance=config, schema=service_registry_schema)

    def test_llm_service_valid(self, service_registry_schema):
        """Test valid LLM service configuration."""
        config = {
            "services": {
                "llm1": {
                    "name": "test-llm",
                    "type": "llm",
                    "url": "http://localhost:1234/v1",
                    "model": "test-model"
                }
            }
        }
        jsonschema.validate(instance=config, schema=service_registry_schema)

    def test_embedding_service_valid(self, service_registry_schema):
        """Test valid embedding service configuration."""
        config = {
            "services": {
                "embedding1": {
                    "name": "test-embedding",
                    "type": "embedding",
                    "provider": "onnx",
                    "model": "test-model"
                }
            }
        }
        jsonschema.validate(instance=config, schema=service_registry_schema)

    def test_database_service_valid(self, service_registry_schema):
        """Test valid database service configuration."""
        config = {
            "services": {
                "db1": {
                    "name": "test-db",
                    "type": "postgres",
                    "connection_string": "postgresql://localhost/test"
                }
            }
        }
        jsonschema.validate(instance=config, schema=service_registry_schema)

    def test_missing_services_invalid(self, service_registry_schema):
        """Test that missing 'services' key is invalid."""
        config = {}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=config, schema=service_registry_schema)

    def test_invalid_service_name(self, service_registry_schema):
        """Test that invalid service names are rejected."""
        config = {
            "services": {
                "invalid-name-with-hyphen": {  # Should only allow alphanumeric + underscore
                    "name": "test",
                    "type": "llm"
                }
            }
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=config, schema=service_registry_schema)

    def test_version_format(self, service_registry_schema):
        """Test version format validation."""
        # Valid version
        config = {
            "version": "1.0.0",
            "services": {}
        }
        jsonschema.validate(instance=config, schema=service_registry_schema)

        # Invalid version
        config = {
            "version": "1.0",  # Not semver
            "services": {}
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=config, schema=service_registry_schema)


class TestStrategyPairSchema:
    """Test strategy pair JSON schema."""

    def test_schema_is_valid_json_schema(self, strategy_pair_schema):
        """Test that the schema itself is valid JSON Schema."""
        jsonschema.Draft7Validator.check_schema(strategy_pair_schema)

    def test_minimal_valid_config(self, strategy_pair_schema):
        """Test minimal valid configuration."""
        config = {
            "strategy_name": "test-pair",
            "version": "1.0.0",
            "indexer": {
                "strategy": "TestIndexer",
                "services": {}
            },
            "retriever": {
                "strategy": "TestRetriever",
                "services": {}
            }
        }
        jsonschema.validate(instance=config, schema=strategy_pair_schema)

    def test_service_reference_format(self, strategy_pair_schema):
        """Test service reference format validation."""
        # Valid reference
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
        jsonschema.validate(instance=config, schema=strategy_pair_schema)

    def test_inline_service_config(self, strategy_pair_schema):
        """Test inline service configuration."""
        config = {
            "strategy_name": "test-pair",
            "version": "1.0.0",
            "indexer": {
                "strategy": "TestIndexer",
                "services": {
                    "embedding": {
                        "name": "inline",
                        "type": "embedding",
                        "provider": "onnx"
                    }
                }
            },
            "retriever": {
                "strategy": "TestRetriever",
                "services": {}
            }
        }
        jsonschema.validate(instance=config, schema=strategy_pair_schema)

    def test_db_config(self, strategy_pair_schema):
        """Test database configuration."""
        config = {
            "strategy_name": "test-pair",
            "version": "1.0.0",
            "indexer": {
                "strategy": "TestIndexer",
                "services": {},
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
                "services": {}
            }
        }
        jsonschema.validate(instance=config, schema=strategy_pair_schema)

    def test_migrations_config(self, strategy_pair_schema):
        """Test migrations configuration."""
        config = {
            "strategy_name": "test-pair",
            "version": "1.0.0",
            "indexer": {
                "strategy": "TestIndexer",
                "services": {}
            },
            "retriever": {
                "strategy": "TestRetriever",
                "services": {}
            },
            "migrations": {
                "required_revisions": [
                    "001_create_tables",
                    "002_add_indexes"
                ]
            }
        }
        jsonschema.validate(instance=config, schema=strategy_pair_schema)

    def test_expected_schema_config(self, strategy_pair_schema):
        """Test expected schema configuration."""
        config = {
            "strategy_name": "test-pair",
            "version": "1.0.0",
            "indexer": {
                "strategy": "TestIndexer",
                "services": {}
            },
            "retriever": {
                "strategy": "TestRetriever",
                "services": {}
            },
            "expected_schema": {
                "tables": ["documents", "chunks"],
                "indexes": ["idx_chunks_embedding"],
                "extensions": ["vector"]
            }
        }
        jsonschema.validate(instance=config, schema=strategy_pair_schema)

    def test_tags(self, strategy_pair_schema):
        """Test tags configuration."""
        config = {
            "strategy_name": "test-pair",
            "version": "1.0.0",
            "indexer": {
                "strategy": "TestIndexer",
                "services": {}
            },
            "retriever": {
                "strategy": "TestRetriever",
                "services": {}
            },
            "tags": ["semantic", "local", "production"]
        }
        jsonschema.validate(instance=config, schema=strategy_pair_schema)

    def test_invalid_strategy_name_format(self, strategy_pair_schema):
        """Test that invalid strategy name format is rejected."""
        config = {
            "strategy_name": "Invalid_Name",  # Should be lowercase with hyphens
            "version": "1.0.0",
            "indexer": {
                "strategy": "TestIndexer",
                "services": {}
            },
            "retriever": {
                "strategy": "TestRetriever",
                "services": {}
            }
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=config, schema=strategy_pair_schema)

    def test_missing_required_fields(self, strategy_pair_schema):
        """Test that missing required fields are rejected."""
        # Missing indexer
        config = {
            "strategy_name": "test-pair",
            "version": "1.0.0",
            "retriever": {
                "strategy": "TestRetriever",
                "services": {}
            }
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=config, schema=strategy_pair_schema)

        # Missing retriever
        config = {
            "strategy_name": "test-pair",
            "version": "1.0.0",
            "indexer": {
                "strategy": "TestIndexer",
                "services": {}
            }
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=config, schema=strategy_pair_schema)
