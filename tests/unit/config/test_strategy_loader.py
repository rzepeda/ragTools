import pytest
import json
import yaml
from pathlib import Path
from rag_factory.config.strategy_loader import StrategyPairLoader, ConfigurationError

# Minimal valid schema content for reference (no need to redefine if using real schema)
# We test against the real schema file if possible, or mock it.
# The class loads schema from disk relative to itself.

@pytest.fixture
def valid_config():
    return {
        "strategy_name": "test-strategy",
        "version": "1.0.0",
        "description": "A test strategy pair",
        "indexer": {
            "strategy": "rag_factory.strategies.loading.SimpleLoader",
            "services": {
                "llm": "$gpt4",
                "db": "$postgres"
            },
            "db_config": {
                "tables": {"chunks": "chunks_v1"}
            },
            "config": {
                "batch_size": 10
            }
        },
        "retriever": {
            "strategy": "rag_factory.strategies.retrieval.SimpleRetriever",
            "services": {
                "llm": "$gpt4",
                "embedding": "$text-embedding-3"
            },
            "config": {
                "top_k": 5
            }
        },
        "migrations": {
            "required_revisions": ["rev1", "rev2"]
        }
    }

@pytest.fixture
def loader(tmp_path):
    # We rely on the real schema file being present in the package. 
    # If the test environment doesn't have it installed in site-packages, 
    # we might need to point to the source location.
    
    # Check if schema exists where the code expects it
    # rag_factory/config/schemas/strategy_pair_schema.json
    
    # For unit tests, we can instantiate loader.
    loader = StrategyPairLoader()
    return loader

def test_load_valid_config(loader, valid_config, tmp_path):
    config_file = tmp_path / "valid_strategy.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_config, f)
        
    loaded_config = loader.load_config(config_file)
    assert loaded_config == valid_config

def test_load_invalid_yaml(loader, tmp_path):
    config_file = tmp_path / "invalid.yaml"
    with open(config_file, "w") as f:
        f.write("invalid: [yaml: { unclosed")
        
    with pytest.raises(ConfigurationError, match="Error parsing YAML"):
        loader.load_config(config_file)

def test_load_missing_file(loader, tmp_path):
    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        loader.load_config(tmp_path / "nonexistent.yaml")

def test_schema_validation_error(loader, valid_config, tmp_path):
    # Make config invalid (missing required field)
    del valid_config["indexer"]
    
    config_file = tmp_path / "invalid_schema.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_config, f)
        
    with pytest.raises(ConfigurationError, match="Configuration validation failed"):
        loader.load_config(config_file)

def test_schema_validation_types(loader, valid_config, tmp_path):
    # Invalid type for strategy_name
    valid_config["strategy_name"] = 123
    
    config_file = tmp_path / "invalid_type.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_config, f)
        
    with pytest.raises(ConfigurationError, match="validation failed"):
        loader.load_config(config_file)

def test_schema_pattern_validation(loader, valid_config, tmp_path):
    # Invalid service reference format
    valid_config["indexer"]["services"]["llm"] = "invalid_ref_without_dollar"
    # Actually wait, the schema says: 
    # "pattern": "^\\$[a-zA-Z0-9_]+$" OR object.
    # So "invalid_ref_without_dollar" matches neither if it's treated rigidly?
    # Wait, the schema allows "patternProperties": "^[a-zA-Z_]+$": { "oneOf": [ {pattern: ...}, {object} ] }
    # So "invalid" string would fail both string pattern and object check.
    
    config_file = tmp_path / "invalid_pattern.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_config, f)
        
    # Validation strictly checks the pattern
    with pytest.raises(ConfigurationError, match="validation failed"):
        loader.load_config(config_file)
