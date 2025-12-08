"""Unit tests for RAGFactory dependency injection (Story 11.5)."""

import pytest
from typing import Set, Dict, Any, List
from unittest.mock import Mock

from rag_factory.factory import RAGFactory
from rag_factory.strategies.base import IRAGStrategy, Chunk, PreparedData
from rag_factory.services.dependencies import StrategyDependencies, ServiceDependency
from rag_factory.services.interfaces import (
    ILLMService,
    IEmbeddingService,
    IGraphService,
    IDatabaseService,
    IRerankingService,
)


# Test Strategy Classes

class DITestStrategy(IRAGStrategy):
    """Test strategy that requires LLM and Embedding services."""

    def requires_services(self) -> Set[ServiceDependency]:
        return {ServiceDependency.LLM, ServiceDependency.EMBEDDING}

    def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
        return PreparedData(chunks=[], embeddings=[], index_metadata={})

    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        return []

    async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
        return []

    def process_query(self, query: str, context: List[Chunk]) -> str:
        return "test answer"


class DatabaseOnlyStrategy(IRAGStrategy):
    """Test strategy that only requires database service."""

    def requires_services(self) -> Set[ServiceDependency]:
        return {ServiceDependency.DATABASE}

    def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
        return PreparedData(chunks=[], embeddings=[], index_metadata={})

    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        return []

    async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
        return []

    def process_query(self, query: str, context: List[Chunk]) -> str:
        return "database answer"


class NoServiceStrategy(IRAGStrategy):
    """Test strategy that requires no services."""

    def requires_services(self) -> Set[ServiceDependency]:
        return set()

    def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
        return PreparedData(chunks=[], embeddings=[], index_metadata={})

    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        return []

    async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
        return []

    def process_query(self, query: str, context: List[Chunk]) -> str:
        return "no service answer"


# Fixtures

@pytest.fixture
def mock_llm():
    """Create a mock LLM service."""
    return Mock(spec=ILLMService)


@pytest.fixture
def mock_embedding():
    """Create a mock embedding service."""
    return Mock(spec=IEmbeddingService)


@pytest.fixture
def mock_database():
    """Create a mock database service."""
    return Mock(spec=IDatabaseService)


@pytest.fixture
def mock_graph():
    """Create a mock graph service."""
    return Mock(spec=IGraphService)


@pytest.fixture
def mock_reranker():
    """Create a mock reranker service."""
    return Mock(spec=IRerankingService)


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean the factory registry before each test."""
    RAGFactory._registry = {}
    RAGFactory._dependencies = {}
    yield
    RAGFactory._registry = {}
    RAGFactory._dependencies = {}


# TC11.5.1: Factory Initialization with Services

def test_factory_initialization_with_no_services():
    """Test factory can be initialized with no services."""
    factory = RAGFactory()
    
    assert factory.dependencies is not None
    assert factory.dependencies.llm_service is None
    assert factory.dependencies.embedding_service is None
    assert factory.dependencies.graph_service is None
    assert factory.dependencies.database_service is None
    assert factory.dependencies.reranker_service is None


def test_factory_initialization_with_llm_service(mock_llm):
    """Test factory initialization with LLM service."""
    factory = RAGFactory(llm_service=mock_llm)
    
    assert factory.dependencies.llm_service is mock_llm
    assert factory.dependencies.embedding_service is None


def test_factory_initialization_with_all_services(
    mock_llm, mock_embedding, mock_database, mock_graph, mock_reranker
):
    """Test factory initialization with all services."""
    factory = RAGFactory(
        llm_service=mock_llm,
        embedding_service=mock_embedding,
        database_service=mock_database,
        graph_service=mock_graph,
        reranker_service=mock_reranker,
    )
    
    assert factory.dependencies.llm_service is mock_llm
    assert factory.dependencies.embedding_service is mock_embedding
    assert factory.dependencies.database_service is mock_database
    assert factory.dependencies.graph_service is mock_graph
    assert factory.dependencies.reranker_service is mock_reranker


def test_factory_initialization_with_partial_services(mock_llm, mock_database):
    """Test factory initialization with partial services."""
    factory = RAGFactory(
        llm_service=mock_llm,
        database_service=mock_database,
    )
    
    assert factory.dependencies.llm_service is mock_llm
    assert factory.dependencies.database_service is mock_database
    assert factory.dependencies.embedding_service is None
    assert factory.dependencies.graph_service is None
    assert factory.dependencies.reranker_service is None


# TC11.5.2: Strategy Creation with Dependencies

def test_create_strategy_with_required_services(mock_llm, mock_embedding):
    """Test creating strategy with all required services."""
    factory = RAGFactory(
        llm_service=mock_llm,
        embedding_service=mock_embedding,
    )
    factory.register_strategy("di_test", DITestStrategy)
    
    config = {"chunk_size": 512}
    strategy = factory.create_strategy("di_test", config)
    
    assert isinstance(strategy, DITestStrategy)
    assert strategy.config == config
    assert strategy.deps.llm_service is mock_llm
    assert strategy.deps.embedding_service is mock_embedding


def test_create_strategy_without_required_services():
    """Test creating strategy without required services raises error."""
    factory = RAGFactory()  # No services
    factory.register_strategy("di_test", DITestStrategy)
    
    with pytest.raises(ValueError) as exc_info:
        factory.create_strategy("di_test", {})
    
    error_msg = str(exc_info.value)
    assert "Failed to create strategy 'di_test'" in error_msg
    assert "DITestStrategy requires services" in error_msg


def test_create_strategy_with_partial_services(mock_llm):
    """Test creating strategy with only partial required services fails."""
    factory = RAGFactory(llm_service=mock_llm)  # Missing embedding
    factory.register_strategy("di_test", DITestStrategy)
    
    with pytest.raises(ValueError) as exc_info:
        factory.create_strategy("di_test", {})
    
    error_msg = str(exc_info.value)
    assert "EMBEDDING" in error_msg


def test_create_strategy_with_no_service_requirements():
    """Test creating strategy that requires no services."""
    factory = RAGFactory()  # No services
    factory.register_strategy("no_service", NoServiceStrategy)
    
    strategy = factory.create_strategy("no_service", {})
    
    assert isinstance(strategy, NoServiceStrategy)


def test_create_database_only_strategy(mock_database):
    """Test creating strategy that only requires database."""
    factory = RAGFactory(database_service=mock_database)
    factory.register_strategy("db_only", DatabaseOnlyStrategy)
    
    strategy = factory.create_strategy("db_only", {})
    
    assert isinstance(strategy, DatabaseOnlyStrategy)
    assert strategy.deps.database_service is mock_database


def test_create_strategy_with_extra_services(
    mock_llm, mock_embedding, mock_database
):
    """Test creating strategy with extra services it doesn't need."""
    factory = RAGFactory(
        llm_service=mock_llm,
        embedding_service=mock_embedding,
        database_service=mock_database,  # Extra service
    )
    factory.register_strategy("di_test", DITestStrategy)
    
    strategy = factory.create_strategy("di_test", {})
    
    assert isinstance(strategy, DITestStrategy)
    assert strategy.deps.database_service is mock_database  # Extra service available


# TC11.5.3: Dependency Overrides

def test_create_strategy_with_override_dependencies(mock_llm, mock_embedding):
    """Test creating strategy with override dependencies."""
    # Factory has one set of services
    factory_llm = Mock(spec=ILLMService)
    factory_embedding = Mock(spec=IEmbeddingService)
    factory = RAGFactory(
        llm_service=factory_llm,
        embedding_service=factory_embedding,
    )
    factory.register_strategy("di_test", DITestStrategy)
    
    # Override with different services
    override_deps = StrategyDependencies(
        llm_service=mock_llm,
        embedding_service=mock_embedding,
    )
    
    strategy = factory.create_strategy("di_test", {}, override_deps=override_deps)
    
    # Strategy should use override services, not factory services
    assert strategy.deps.llm_service is mock_llm
    assert strategy.deps.embedding_service is mock_embedding
    assert strategy.deps.llm_service is not factory_llm
    assert strategy.deps.embedding_service is not factory_embedding


def test_override_dependencies_can_add_missing_services(mock_llm, mock_embedding):
    """Test override dependencies can provide missing services."""
    factory = RAGFactory()  # No services
    factory.register_strategy("di_test", DITestStrategy)
    
    # Provide services via override
    override_deps = StrategyDependencies(
        llm_service=mock_llm,
        embedding_service=mock_embedding,
    )
    
    strategy = factory.create_strategy("di_test", {}, override_deps=override_deps)
    
    assert isinstance(strategy, DITestStrategy)
    assert strategy.deps.llm_service is mock_llm
    assert strategy.deps.embedding_service is mock_embedding


def test_override_dependencies_still_validates_requirements():
    """Test override dependencies are still validated."""
    factory = RAGFactory()
    factory.register_strategy("di_test", DITestStrategy)
    
    # Override with incomplete services
    override_deps = StrategyDependencies(
        llm_service=Mock(spec=ILLMService)
        # Missing embedding
    )
    
    with pytest.raises(ValueError) as exc_info:
        factory.create_strategy("di_test", {}, override_deps=override_deps)
    
    assert "EMBEDDING" in str(exc_info.value)


# TC11.5.4: Error Handling

def test_error_message_includes_strategy_name():
    """Test error message includes strategy name for context."""
    factory = RAGFactory()
    factory.register_strategy("di_test", DITestStrategy)
    
    with pytest.raises(ValueError) as exc_info:
        factory.create_strategy("di_test", {})
    
    error_msg = str(exc_info.value)
    assert "di_test" in error_msg
    assert "DITestStrategy" in error_msg


def test_error_message_lists_missing_services():
    """Test error message clearly lists missing services."""
    factory = RAGFactory()
    factory.register_strategy("di_test", DITestStrategy)
    
    with pytest.raises(ValueError) as exc_info:
        factory.create_strategy("di_test", {})
    
    error_msg = str(exc_info.value)
    assert "LLM" in error_msg
    assert "EMBEDDING" in error_msg


# TC11.5.5: Integration with Existing Factory Features

def test_create_strategy_without_config(mock_llm, mock_embedding):
    """Test creating strategy without config uses empty dict."""
    factory = RAGFactory(
        llm_service=mock_llm,
        embedding_service=mock_embedding,
    )
    factory.register_strategy("di_test", DITestStrategy)
    
    strategy = factory.create_strategy("di_test")
    
    assert isinstance(strategy, DITestStrategy)
    assert strategy.config == {}


def test_create_strategy_with_config(mock_llm, mock_embedding):
    """Test creating strategy with config passes it correctly."""
    factory = RAGFactory(
        llm_service=mock_llm,
        embedding_service=mock_embedding,
    )
    factory.register_strategy("di_test", DITestStrategy)
    
    config = {"chunk_size": 1024, "top_k": 10, "custom_param": "value"}
    strategy = factory.create_strategy("di_test", config)
    
    assert strategy.config == config
    assert strategy.config["chunk_size"] == 1024
    assert strategy.config["custom_param"] == "value"


def test_create_from_config_with_dependencies(tmp_path, mock_llm, mock_embedding):
    """Test creating strategy from config file with dependencies."""
    factory = RAGFactory(
        llm_service=mock_llm,
        embedding_service=mock_embedding,
    )
    factory.register_strategy("di_test", DITestStrategy)
    
    # Create config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
strategy_name: di_test
chunk_size: 512
top_k: 5
"""
    )
    
    strategy = factory.create_from_config(str(config_file))
    
    assert isinstance(strategy, DITestStrategy)
    assert strategy.config["chunk_size"] == 512
    assert strategy.deps.llm_service is mock_llm


def test_multiple_strategies_with_same_factory(
    mock_llm, mock_embedding, mock_database
):
    """Test creating multiple different strategies with same factory."""
    factory = RAGFactory(
        llm_service=mock_llm,
        embedding_service=mock_embedding,
        database_service=mock_database,
    )
    
    factory.register_strategy("di_test", DITestStrategy)
    factory.register_strategy("db_only", DatabaseOnlyStrategy)
    factory.register_strategy("no_service", NoServiceStrategy)
    
    # Create all three strategies
    strategy1 = factory.create_strategy("di_test", {})
    strategy2 = factory.create_strategy("db_only", {})
    strategy3 = factory.create_strategy("no_service", {})
    
    assert isinstance(strategy1, DITestStrategy)
    assert isinstance(strategy2, DatabaseOnlyStrategy)
    assert isinstance(strategy3, NoServiceStrategy)
    
    # Each should have correct dependencies
    assert strategy1.deps.llm_service is mock_llm
    assert strategy2.deps.database_service is mock_database
    assert strategy3.deps.llm_service is mock_llm  # Has access but doesn't need


# TC11.5.6: Backward Compatibility

def test_class_level_dependencies_still_exist():
    """Test that class-level _dependencies dict still exists for backward compat."""
    factory = RAGFactory()
    
    assert hasattr(RAGFactory, "_dependencies")
    assert isinstance(RAGFactory._dependencies, dict)


def test_set_and_get_dependency_still_work():
    """Test legacy set_dependency and get_dependency methods still work."""
    factory = RAGFactory()
    
    mock_service = Mock()
    factory.set_dependency("test_service", mock_service)
    
    retrieved = factory.get_dependency("test_service")
    assert retrieved is mock_service


def test_legacy_create_strategy_method_exists():
    """Test that create_strategy_legacy method exists for backward compatibility."""
    factory = RAGFactory()
    
    assert hasattr(factory, "create_strategy_legacy")
    assert callable(factory.create_strategy_legacy)
