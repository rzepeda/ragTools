"""
Integration tests for Configuration Management System.

This module contains integration tests that verify complete configuration
lifecycle, multi-environment scenarios, and integration with other components.
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest

from rag_factory.legacy_config import ConfigManager, StrategyConfigSchema
from rag_factory.factory import RAGFactory
from rag_factory.pipeline import StrategyPipeline
from rag_factory.strategies.base import Chunk, IRAGStrategy


# Reset singleton between tests
@pytest.fixture(autouse=True)
def reset_config_manager():
    """Reset ConfigManager singleton between tests."""
    ConfigManager._instance = None
    ConfigManager._config = None
    ConfigManager._config_path = None
    ConfigManager._observers = []
    ConfigManager._callbacks = []
    yield
    if ConfigManager._instance:
        ConfigManager._instance.disable_hot_reload()


# Helper strategy for testing
class TestIntegrationStrategy(IRAGStrategy):
    """Test strategy for integration tests."""

    def __init__(self, config: Dict[str, Any], dependencies: Any) -> None:
        """Initialize with config and dependencies."""
        super().__init__(config, dependencies)

    def requires_services(self):
        """Declare required services."""
        return set()

    def initialize(self, config: Any) -> None:
        """Initialize strategy with config (legacy method)."""
        self.config = config

    def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
        """Prepare data."""
        return None

    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        """Retrieve chunks."""
        return [Chunk("Test", {}, 0.9, "doc1", "c1")]

    async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
        """Async retrieve."""
        return [Chunk("Test", {}, 0.9, "doc1", "c1")]

    def process_query(self, query: str, context: List[Chunk]) -> str:
        """Process query."""
        return "Test response"


# IS4.1: Full Configuration Lifecycle
@pytest.mark.integration
def test_load_validate_use_config(tmp_path: Path) -> None:
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
    assert config.get("global_settings.cache_enabled") is True

    # Access strategy config
    strategy_config = config.get_strategy_config("test_strategy")
    assert strategy_config.chunk_size == 512
    assert strategy_config.top_k == 5

    # Access pipeline config
    pipeline_config = config.get("pipeline")
    assert pipeline_config["mode"] == "sequential"
    assert len(pipeline_config["stages"]) == 1


# IS4.2: Multi-Environment Configuration
@pytest.mark.integration
def test_multi_environment_configuration(tmp_path: Path) -> None:
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
    config_dev = ConfigManager()
    config_dev.load(str(base_config), environment="development")
    assert config_dev.get("global_settings.log_level") == "DEBUG"
    assert config_dev.get("global_settings.cache_ttl") == 3600  # From base
    assert config_dev.get("strategies.strategy1.chunk_size") == 512

    # Reset for prod test
    ConfigManager._instance = None

    # Test prod environment
    config_prod = ConfigManager()
    config_prod.load(str(base_config), environment="production")
    assert config_prod.get("global_settings.log_level") == "ERROR"
    assert config_prod.get("global_settings.cache_ttl") == 7200
    assert config_prod.get("strategies.strategy1.chunk_size") == 1024


# IS4.3: Integration with Factory
@pytest.mark.integration
def test_config_with_factory(tmp_path: Path) -> None:
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

    # Register strategy
    RAGFactory.register_strategy(
        "test_strategy",
        TestIntegrationStrategy,
        override=True
    )

    # Create factory instance
    factory = RAGFactory()
    
    # Get strategy config and create strategy
    strategy_config = config.get_strategy_config("test_strategy")
    strategy = factory.create_strategy(
        "test_strategy",
        strategy_config.model_dump()
    )

    assert strategy.config["chunk_size"] == 1024
    assert strategy.config["top_k"] == 10


# IS4.4: Integration with Pipeline
@pytest.mark.integration
def test_config_with_pipeline(tmp_path: Path) -> None:
    """Test configuration used with StrategyPipeline."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
strategies:
  test_strategy:
    chunk_size: 512
    top_k: 5

pipeline:
  mode: sequential
  stages:
    - strategy: test_strategy
      name: stage1
      config:
        top_k: 10
""")

    # Load config
    config = ConfigManager()
    config.load(str(config_file))

    # Register strategy
    RAGFactory.register_strategy(
        "test_strategy",
        TestIntegrationStrategy,
        override=True
    )

    # Create pipeline from config
    pipeline_config = config.get("pipeline")
    pipeline = StrategyPipeline.from_config(pipeline_config)

    assert len(pipeline.stages) == 1
    assert pipeline.stages[0].name == "stage1"


# IS4.5: Complete Real-World Scenario
@pytest.mark.integration
def test_complete_real_world_scenario(tmp_path: Path) -> None:
    """Test a complete real-world configuration scenario."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
global_settings:
  environment: development
  log_level: INFO
  cache_enabled: true
  cache_ttl: 3600

strategies:
  semantic_search:
    chunk_size: 256
    chunk_overlap: 25
    top_k: 20
    strategy_name: semantic_search
    metadata:
      model: "all-MiniLM-L6-v2"

  reranking:
    chunk_size: 512
    chunk_overlap: 50
    top_k: 5
    strategy_name: reranking
    metadata:
      model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

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
""")

    # Load configuration
    config = ConfigManager()
    config.load(str(config_file))

    # Verify global settings
    assert config.get("global_settings.environment") == "development"
    assert config.get("global_settings.log_level") == "INFO"
    assert config.get("global_settings.cache_enabled") is True

    # Verify strategy configurations
    semantic_config = config.get_strategy_config("semantic_search")
    assert semantic_config.chunk_size == 256
    assert semantic_config.top_k == 20
    assert semantic_config.metadata["model"] == "all-MiniLM-L6-v2"

    reranking_config = config.get_strategy_config("reranking")
    assert reranking_config.chunk_size == 512
    assert reranking_config.top_k == 5
    assert reranking_config.metadata["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Verify pipeline configuration
    pipeline_config = config.get("pipeline")
    assert pipeline_config["mode"] == "sequential"
    assert pipeline_config["timeout"] == 30
    assert len(pipeline_config["stages"]) == 2


# IS4.6: Configuration Export and Re-import
@pytest.mark.integration
def test_config_export_and_reimport(tmp_path: Path) -> None:
    """Test exporting configuration and re-importing it."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
global_settings:
  log_level: DEBUG
strategies:
  test_strategy:
    chunk_size: 1024
""")

    # Load config
    config1 = ConfigManager()
    config1.load(str(config_file))

    # Export to dict
    exported = config1.to_dict()

    # Reset singleton
    ConfigManager._instance = None

    # Re-import from dict
    config2 = ConfigManager()
    config2.load(config_dict=exported)

    # Verify same values
    assert config2.get("global_settings.log_level") == "DEBUG"
    assert config2.get("strategies.test_strategy.chunk_size") == 1024


# IS4.7: Configuration with JSON
@pytest.mark.integration
def test_json_configuration_integration(tmp_path: Path) -> None:
    """Test complete integration with JSON configuration."""
    config_file = tmp_path / "config.json"
    config_file.write_text("""{
  "global_settings": {
    "log_level": "WARNING",
    "cache_enabled": false
  },
  "strategies": {
    "json_strategy": {
      "chunk_size": 2048,
      "top_k": 15
    }
  }
}""")

    # Load JSON config
    config = ConfigManager()
    config.load(str(config_file))

    # Verify
    assert config.get("global_settings.log_level") == "WARNING"
    assert config.get("global_settings.cache_enabled") is False

    json_strategy = config.get_strategy_config("json_strategy")
    assert json_strategy.chunk_size == 2048
    assert json_strategy.top_k == 15


# IS4.8: Nested Configuration Access
@pytest.mark.integration
def test_deeply_nested_configuration_access(tmp_path: Path) -> None:
    """Test accessing deeply nested configuration values."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
strategies:
  complex_strategy:
    chunk_size: 512
    metadata:
      model:
        name: "complex-model"
        version: "2.0"
        parameters:
          temperature: 0.7
          max_length: 1024
""")

    config = ConfigManager()
    config.load(str(config_file))

    # Access deeply nested values
    strategy_config = config.get_strategy_config("complex_strategy")
    assert strategy_config.metadata["model"]["name"] == "complex-model"
    assert strategy_config.metadata["model"]["version"] == "2.0"
    assert strategy_config.metadata["model"]["parameters"]["temperature"] == 0.7
    assert strategy_config.metadata["model"]["parameters"]["max_length"] == 1024
