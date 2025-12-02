"""
Unit tests for RAGFactory.

This module contains comprehensive unit tests for the RAGFactory class,
including tests for strategy registration, creation, configuration validation,
and the decorator pattern.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from rag_factory.factory import (
    RAGFactory,
    StrategyNotFoundError,
    ConfigurationError,
    register_rag_strategy,
)
from rag_factory.strategies.base import (
    IRAGStrategy,
    StrategyConfig,
    Chunk,
    PreparedData,
)


# Test fixtures and helper classes


class TestStrategy(IRAGStrategy):
    """Minimal test strategy for testing purposes."""

    def initialize(self, config: StrategyConfig) -> None:
        """Initialize with config."""
        self.config = config

    def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
        """Prepare test data."""
        chunks = [
            Chunk(
                text=doc.get("text", ""),
                metadata={},
                score=1.0,
                source_id=doc.get("id", "unknown"),
                chunk_id=f"chunk_{i}",
            )
            for i, doc in enumerate(documents)
        ]
        return PreparedData(chunks=chunks, embeddings=[], index_metadata={})

    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        """Retrieve test chunks."""
        return []

    async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
        """Async retrieve test chunks."""
        return self.retrieve(query, top_k)

    def process_query(self, query: str, context: List[Chunk]) -> str:
        """Process test query."""
        return "test answer"


class BrokenStrategy(IRAGStrategy):
    """Strategy that fails during initialization."""

    def initialize(self, config: StrategyConfig) -> None:
        """Initialize and raise error."""
        raise RuntimeError("Initialization failed")

    def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
        """Prepare data."""
        return PreparedData(chunks=[], embeddings=[], index_metadata={})

    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        """Retrieve chunks."""
        return []

    async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
        """Async retrieve chunks."""
        return []

    def process_query(self, query: str, context: List[Chunk]) -> str:
        """Process query."""
        return ""


@pytest.fixture
def factory() -> RAGFactory:
    """Create a fresh factory instance for each test."""
    # Clear registry before each test
    RAGFactory._registry = {}
    RAGFactory._dependencies = {}
    return RAGFactory()


# TC2.1: Factory Instantiation Tests


def test_factory_can_be_created(factory: RAGFactory) -> None:
    """Test factory can be instantiated."""
    assert isinstance(factory, RAGFactory)


def test_factory_has_empty_registry_initially(factory: RAGFactory) -> None:
    """Test factory starts with empty registry."""
    assert len(factory.list_strategies()) == 0


# TC2.2: Strategy Registration Tests


def test_register_strategy_adds_to_registry(factory: RAGFactory) -> None:
    """Test registering a strategy adds it to registry."""
    factory.register_strategy("test_strategy", TestStrategy)
    assert "test_strategy" in factory.list_strategies()


def test_register_duplicate_raises_error(factory: RAGFactory) -> None:
    """Test registering duplicate strategy name raises error."""
    factory.register_strategy("test", TestStrategy)

    with pytest.raises(ValueError, match="already registered"):
        factory.register_strategy("test", TestStrategy)


def test_register_duplicate_with_override(factory: RAGFactory) -> None:
    """Test override=True allows replacing strategy."""

    class StrategyV1(IRAGStrategy):
        """Version 1 strategy."""

        def initialize(self, config: StrategyConfig) -> None:
            """Initialize."""
            pass

        def prepare_data(
            self, documents: List[Dict[str, Any]]
        ) -> PreparedData:
            """Prepare data."""
            return PreparedData(chunks=[], embeddings=[], index_metadata={})

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    class StrategyV2(IRAGStrategy):
        """Version 2 strategy."""

        def initialize(self, config: StrategyConfig) -> None:
            """Initialize."""
            pass

        def prepare_data(
            self, documents: List[Dict[str, Any]]
        ) -> PreparedData:
            """Prepare data."""
            return PreparedData(chunks=[], embeddings=[], index_metadata={})

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    factory.register_strategy("strategy", StrategyV1)
    factory.register_strategy("strategy", StrategyV2, override=True)

    # Should use V2
    assert factory._registry["strategy"] == StrategyV2


def test_unregister_strategy(factory: RAGFactory) -> None:
    """Test unregistering a strategy removes it."""
    factory.register_strategy("test", TestStrategy)
    factory.unregister_strategy("test")

    assert "test" not in factory.list_strategies()


def test_unregister_nonexistent_strategy(factory: RAGFactory) -> None:
    """Test unregistering a nonexistent strategy raises error."""
    with pytest.raises(KeyError, match="not found"):
        factory.unregister_strategy("nonexistent")


def test_list_strategies_returns_all_names(factory: RAGFactory) -> None:
    """Test list_strategies returns all registered names."""

    class Strategy1(IRAGStrategy):
        """Strategy 1."""

        def initialize(self, config: StrategyConfig) -> None:
            """Initialize."""
            pass

        def prepare_data(
            self, documents: List[Dict[str, Any]]
        ) -> PreparedData:
            """Prepare data."""
            return PreparedData(chunks=[], embeddings=[], index_metadata={})

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    class Strategy2(IRAGStrategy):
        """Strategy 2."""

        def initialize(self, config: StrategyConfig) -> None:
            """Initialize."""
            pass

        def prepare_data(
            self, documents: List[Dict[str, Any]]
        ) -> PreparedData:
            """Prepare data."""
            return PreparedData(chunks=[], embeddings=[], index_metadata={})

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    factory.register_strategy("strategy1", Strategy1)
    factory.register_strategy("strategy2", Strategy2)

    strategies = factory.list_strategies()
    assert "strategy1" in strategies
    assert "strategy2" in strategies
    assert len(strategies) == 2


# TC2.3: Strategy Creation Tests


def test_create_strategy_returns_instance(factory: RAGFactory) -> None:
    """Test creating a strategy returns an instance."""
    factory.register_strategy("test", TestStrategy)

    strategy = factory.create_strategy("test")
    assert isinstance(strategy, TestStrategy)
    assert isinstance(strategy, IRAGStrategy)


def test_create_unknown_strategy_raises_error(factory: RAGFactory) -> None:
    """Test creating unknown strategy raises StrategyNotFoundError."""
    with pytest.raises(StrategyNotFoundError, match="not found"):
        factory.create_strategy("nonexistent")


def test_create_strategy_with_config(factory: RAGFactory) -> None:
    """Test creating strategy with configuration."""
    factory.register_strategy("test", TestStrategy)

    config = {"chunk_size": 1024, "top_k": 10}
    strategy = factory.create_strategy("test", config)

    assert strategy.config.chunk_size == 1024
    assert strategy.config.top_k == 10


def test_strategy_not_found_error_message_includes_available(
    factory: RAGFactory,
) -> None:
    """Test error message includes list of available strategies."""

    class Strategy1(IRAGStrategy):
        """Strategy 1."""

        def initialize(self, config: StrategyConfig) -> None:
            """Initialize."""
            pass

        def prepare_data(
            self, documents: List[Dict[str, Any]]
        ) -> PreparedData:
            """Prepare data."""
            return PreparedData(chunks=[], embeddings=[], index_metadata={})

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    factory.register_strategy("strategy1", Strategy1)

    with pytest.raises(StrategyNotFoundError) as exc_info:
        factory.create_strategy("wrong_name")

    assert "strategy1" in str(exc_info.value)


def test_create_strategy_without_config(factory: RAGFactory) -> None:
    """Test creating strategy without config uses defaults."""
    factory.register_strategy("test", TestStrategy)

    strategy = factory.create_strategy("test")
    # Strategy should be created but not initialized
    assert isinstance(strategy, TestStrategy)


# TC2.4: Configuration Validation Tests


def test_invalid_config_raises_error(factory: RAGFactory) -> None:
    """Test invalid configuration raises error."""
    factory.register_strategy("test", TestStrategy)

    with pytest.raises(ValueError, match="must be non-negative"):
        factory.create_strategy("test", {"chunk_size": -1})


def test_strategy_initialization_error_propagates(
    factory: RAGFactory,
) -> None:
    """Test errors during strategy initialization propagate correctly."""
    factory.register_strategy("broken", BrokenStrategy)

    with pytest.raises(RuntimeError, match="Initialization failed"):
        factory.create_strategy("broken", {"chunk_size": 512})


# TC2.5: Decorator Tests


def test_register_decorator_auto_registers() -> None:
    """Test @register_rag_strategy decorator auto-registers strategy."""
    # Clear registry
    RAGFactory._registry = {}

    @register_rag_strategy("decorated_strategy")
    class DecoratedStrategy(IRAGStrategy):
        """Decorated strategy."""

        def initialize(self, config: StrategyConfig) -> None:
            """Initialize."""
            pass

        def prepare_data(
            self, documents: List[Dict[str, Any]]
        ) -> PreparedData:
            """Prepare data."""
            return PreparedData(chunks=[], embeddings=[], index_metadata={})

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    factory = RAGFactory()
    assert "decorated_strategy" in factory.list_strategies()


def test_decorator_returns_class_unchanged() -> None:
    """Test decorator doesn't modify the class."""
    # Clear registry
    RAGFactory._registry = {}

    @register_rag_strategy("test")
    class TestStrategyWithAttr(IRAGStrategy):
        """Test strategy with attribute."""

        test_attr = "value"

        def initialize(self, config: StrategyConfig) -> None:
            """Initialize."""
            pass

        def prepare_data(
            self, documents: List[Dict[str, Any]]
        ) -> PreparedData:
            """Prepare data."""
            return PreparedData(chunks=[], embeddings=[], index_metadata={})

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Retrieve."""
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            """Async retrieve."""
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            """Process query."""
            return ""

    assert TestStrategyWithAttr.test_attr == "value"


# TC2.6: File-based Configuration Tests


def test_create_from_yaml_config(tmp_path: Path, factory: RAGFactory) -> None:
    """Test creating strategy from YAML config file."""
    factory.register_strategy("test", TestStrategy)

    # Create config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
strategy_name: test
chunk_size: 512
top_k: 5
"""
    )

    strategy = factory.create_from_config(str(config_file))
    assert strategy.config.chunk_size == 512
    assert strategy.config.top_k == 5


def test_create_from_json_config(tmp_path: Path, factory: RAGFactory) -> None:
    """Test creating strategy from JSON config file."""
    factory.register_strategy("test", TestStrategy)

    config_file = tmp_path / "config.json"
    config_file.write_text('{"strategy_name": "test", "chunk_size": 256}')

    strategy = factory.create_strategy_from_config(str(config_file))
    assert strategy.config.chunk_size == 256


def test_create_from_nonexistent_config_file(
    factory: RAGFactory,
) -> None:
    """Test creating from nonexistent config file raises error."""
    with pytest.raises(FileNotFoundError):
        factory.create_from_config("/nonexistent/config.yaml")


def test_create_from_invalid_yaml(
    tmp_path: Path, factory: RAGFactory
) -> None:
    """Test creating from invalid YAML raises error."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: yaml: content:")

    with pytest.raises(yaml.YAMLError):
        factory.create_from_config(str(config_file))


def test_create_from_config_missing_strategy_name(
    tmp_path: Path, factory: RAGFactory
) -> None:
    """Test config without strategy_name raises error."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("chunk_size: 512\ntop_k: 5\n")

    with pytest.raises(ConfigurationError, match="strategy_name"):
        factory.create_from_config(str(config_file))


# Dependency Injection Tests


def test_set_dependency(factory: RAGFactory) -> None:
    """Test setting a dependency."""

    class MockService:
        """Mock service for testing."""

        pass

    service = MockService()
    factory.set_dependency("embedding_service", service)

    assert "embedding_service" in factory._dependencies
    assert factory._dependencies["embedding_service"] is service


def test_get_dependency(factory: RAGFactory) -> None:
    """Test getting a dependency."""

    class MockService:
        """Mock service for testing."""

        pass

    service = MockService()
    factory.set_dependency("embedding_service", service)

    retrieved = factory.get_dependency("embedding_service")
    assert retrieved is service


def test_get_nonexistent_dependency(factory: RAGFactory) -> None:
    """Test getting nonexistent dependency returns None."""
    assert factory.get_dependency("nonexistent") is None


# Thread Safety Tests (basic)


def test_registry_is_class_level(factory: RAGFactory) -> None:
    """Test that registry is shared across factory instances."""
    factory1 = RAGFactory()
    factory2 = RAGFactory()

    factory1.register_strategy("shared", TestStrategy)

    assert "shared" in factory2.list_strategies()


# Additional Coverage Tests


def test_create_from_unsupported_file_format(
    tmp_path: Path, factory: RAGFactory
) -> None:
    """Test creating from unsupported file format raises error."""
    config_file = tmp_path / "config.txt"
    config_file.write_text("some content")

    with pytest.raises(ConfigurationError, match="Unsupported config file"):
        factory.create_from_config(str(config_file))


def test_clear_registry(factory: RAGFactory) -> None:
    """Test clearing the registry."""
    factory.register_strategy("strategy1", TestStrategy)
    factory.register_strategy("strategy2", TestStrategy)

    assert len(factory.list_strategies()) == 2

    factory.clear_registry()

    assert len(factory.list_strategies()) == 0


def test_clear_dependencies(factory: RAGFactory) -> None:
    """Test clearing dependencies."""

    class MockService:
        """Mock service."""

        pass

    factory.set_dependency("service1", MockService())
    factory.set_dependency("service2", MockService())

    assert factory.get_dependency("service1") is not None
    assert factory.get_dependency("service2") is not None

    factory.clear_dependencies()

    assert factory.get_dependency("service1") is None
    assert factory.get_dependency("service2") is None
