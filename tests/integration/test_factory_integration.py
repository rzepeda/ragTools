"""
Integration tests for RAGFactory.

This module contains integration tests that verify the complete workflows
and interactions of the RAGFactory with strategies, configuration files,
and multiple components working together.
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest

from rag_factory.factory import RAGFactory, register_rag_strategy
from rag_factory.strategies.base import (
    IRAGStrategy,
    StrategyConfig,
    Chunk,
    PreparedData,
)


# Test Strategy Implementations


class DummyStrategy(IRAGStrategy):
    """Complete dummy strategy for integration testing."""

    def initialize(self, config: StrategyConfig) -> None:
        """Initialize the strategy with config."""
        self.config = config
        self.initialized = True

    def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
        """Prepare documents for retrieval."""
        chunks = [
            Chunk(
                text=doc.get("text", ""),
                metadata={"doc_id": doc.get("id", "unknown")},
                score=1.0,
                source_id=doc.get("id", "unknown"),
                chunk_id=f"chunk_{i}",
            )
            for i, doc in enumerate(documents)
        ]
        embeddings = [[0.1, 0.2, 0.3] for _ in chunks]
        return PreparedData(
            chunks=chunks,
            embeddings=embeddings,
            index_metadata={"prepared": len(documents)},
        )

    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        """Retrieve relevant chunks."""
        return [
            Chunk(
                text=f"Result {i}",
                metadata={},
                score=0.9 - (i * 0.1),
                source_id=f"doc_{i}",
                chunk_id=f"chunk_{i}",
            )
            for i in range(top_k)
        ]

    async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
        """Async retrieve relevant chunks."""
        return self.retrieve(query, top_k)

    def process_query(self, query: str, context: List[Chunk]) -> str:
        """Process query with context."""
        context_text = " ".join([c.text for c in context])
        return f"Answer to '{query}' using context: {context_text}"


class StrategyA(IRAGStrategy):
    """Strategy A implementation."""

    strategy_type = "A"

    def initialize(self, config: StrategyConfig) -> None:
        """Initialize."""
        self.config = config

    def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
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
        return f"Strategy A answer to: {query}"


class StrategyB(IRAGStrategy):
    """Strategy B implementation."""

    strategy_type = "B"

    def initialize(self, config: StrategyConfig) -> None:
        """Initialize."""
        self.config = config

    def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
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
        return f"Strategy B answer to: {query}"


class StrategyWithDeps(IRAGStrategy):
    """Strategy that accepts dependencies."""

    def __init__(self, embedding_service: Any = None) -> None:
        """Initialize with optional embedding service."""
        self.embedding_service = embedding_service

    def initialize(self, config: StrategyConfig) -> None:
        """Initialize."""
        self.config = config

    def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
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
        return "answer"


class WorkingStrategy(IRAGStrategy):
    """Strategy that works correctly."""

    def initialize(self, config: StrategyConfig) -> None:
        """Initialize."""
        self.config = config

    def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
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
        return "working"


class BrokenStrategy(IRAGStrategy):
    """Strategy that fails during initialization."""

    def initialize(self, config: StrategyConfig) -> None:
        """Initialize and raise error."""
        raise RuntimeError("Initialization failed")

    def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
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


# Fixtures


@pytest.fixture
def factory() -> RAGFactory:
    """Create a fresh factory for each test."""
    RAGFactory._registry = {}
    RAGFactory._dependencies = {}
    return RAGFactory()


# IS2.1: End-to-End Strategy Creation


@pytest.mark.integration
def test_register_create_use_strategy(factory: RAGFactory) -> None:
    """Test complete workflow: register -> create -> use."""
    # Register
    factory.register_strategy("dummy", DummyStrategy)

    # Create with config
    config = {"chunk_size": 512, "top_k": 3}
    strategy = factory.create_strategy("dummy", config)

    # Verify initialization
    assert strategy.initialized
    assert strategy.config.chunk_size == 512
    assert strategy.config.top_k == 3

    # Use - prepare data
    documents = [
        {"text": "Document 1 content", "id": "doc1"},
        {"text": "Document 2 content", "id": "doc2"},
    ]
    result = strategy.prepare_data(documents)
    assert result.index_metadata["prepared"] == 2
    assert len(result.chunks) == 2

    # Use - retrieve
    chunks = strategy.retrieve("test query", 3)
    assert len(chunks) == 3
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].score > chunks[1].score  # Verify descending scores

    # Use - process query
    answer = strategy.process_query("What is this?", chunks[:2])
    assert "What is this?" in answer
    assert isinstance(answer, str)


# IS2.2: Multiple Strategies Integration


@pytest.mark.integration
def test_create_multiple_different_strategies(factory: RAGFactory) -> None:
    """Test creating multiple different strategy types."""
    factory.register_strategy("strategy_a", StrategyA)
    factory.register_strategy("strategy_b", StrategyB)

    strategy_a = factory.create_strategy("strategy_a")
    strategy_b = factory.create_strategy("strategy_b")

    assert strategy_a.strategy_type == "A"
    assert strategy_b.strategy_type == "B"
    assert type(strategy_a) != type(strategy_b)

    # Both strategies can be used independently
    answer_a = strategy_a.process_query("test", [])
    answer_b = strategy_b.process_query("test", [])

    assert "Strategy A" in answer_a
    assert "Strategy B" in answer_b


@pytest.mark.integration
def test_create_multiple_instances_of_same_strategy(
    factory: RAGFactory,
) -> None:
    """Test creating multiple instances of the same strategy."""
    factory.register_strategy("dummy", DummyStrategy)

    config1 = {"chunk_size": 256, "top_k": 3}
    config2 = {"chunk_size": 1024, "top_k": 10}

    strategy1 = factory.create_strategy("dummy", config1)
    strategy2 = factory.create_strategy("dummy", config2)

    # Different instances with different configs
    assert strategy1 is not strategy2
    assert strategy1.config.chunk_size == 256
    assert strategy2.config.chunk_size == 1024


# IS2.3: Dependency Injection Integration


@pytest.mark.integration
def test_dependency_injection(factory: RAGFactory) -> None:
    """Test injecting dependencies into strategies."""

    class EmbeddingService:
        """Mock embedding service."""

        def embed(self, text: str) -> List[float]:
            """Generate mock embeddings."""
            return [0.1, 0.2, 0.3]

    embedding_service = EmbeddingService()
    factory.set_dependency("embedding_service", embedding_service)

    # Verify dependency is stored
    retrieved_service = factory.get_dependency("embedding_service")
    assert retrieved_service is embedding_service

    # Test embedding service works
    embedding = retrieved_service.embed("test text")
    assert len(embedding) == 3


# IS2.4: Configuration File Integration


@pytest.mark.integration
def test_config_file_with_yaml(tmp_path: Path, factory: RAGFactory) -> None:
    """Test loading configuration from YAML file."""
    factory.register_strategy("dummy", DummyStrategy)

    # Create YAML config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
strategy_name: dummy
chunk_size: 2048
chunk_overlap: 100
top_k: 15
metadata:
  author: test
  version: 1.0
"""
    )

    # Create strategy from config file
    strategy = factory.create_from_config(str(config_file))

    assert isinstance(strategy, DummyStrategy)
    assert strategy.config.chunk_size == 2048
    assert strategy.config.chunk_overlap == 100
    assert strategy.config.top_k == 15
    assert strategy.config.metadata["author"] == "test"
    assert strategy.config.metadata["version"] == 1.0


@pytest.mark.integration
def test_config_file_with_json(tmp_path: Path, factory: RAGFactory) -> None:
    """Test loading configuration from JSON file."""
    factory.register_strategy("dummy", DummyStrategy)

    # Create JSON config file
    config_file = tmp_path / "config.json"
    config_file.write_text(
        """{
    "strategy_name": "dummy",
    "chunk_size": 768,
    "top_k": 7
}"""
    )

    # Create strategy from config file
    strategy = factory.create_from_config(str(config_file))

    assert isinstance(strategy, DummyStrategy)
    assert strategy.config.chunk_size == 768
    assert strategy.config.top_k == 7


# IS2.5: Error Recovery Integration


@pytest.mark.integration
def test_factory_error_recovery(factory: RAGFactory) -> None:
    """Test factory handles errors gracefully and maintains state."""
    factory.register_strategy("working", WorkingStrategy)
    factory.register_strategy("broken", BrokenStrategy)

    # Try to create broken strategy
    with pytest.raises(RuntimeError, match="Initialization failed"):
        factory.create_strategy("broken", {"chunk_size": 512})

    # Factory should still work for other strategies
    strategy = factory.create_strategy("working", {"chunk_size": 512})
    assert isinstance(strategy, WorkingStrategy)

    # Registry should still contain both strategies
    assert "working" in factory.list_strategies()
    assert "broken" in factory.list_strategies()


@pytest.mark.integration
def test_factory_state_after_failed_creation(factory: RAGFactory) -> None:
    """Test factory state remains consistent after failed strategy creation."""
    factory.register_strategy("dummy", DummyStrategy)

    # Try to create with invalid config
    with pytest.raises(ValueError):
        factory.create_strategy("dummy", {"chunk_size": -100})

    # Factory should still work normally
    strategy = factory.create_strategy("dummy", {"chunk_size": 512})
    assert isinstance(strategy, DummyStrategy)
    assert strategy.config.chunk_size == 512


# Additional Integration Tests


@pytest.mark.integration
def test_decorator_integration(factory: RAGFactory) -> None:
    """Test decorator-based registration in integration context."""
    # Clear registry
    RAGFactory._registry = {}

    @register_rag_strategy("decorated_strategy")
    class DecoratedStrategy(IRAGStrategy):
        """Strategy registered via decorator."""

        def initialize(self, config: StrategyConfig) -> None:
            """Initialize."""
            self.config = config

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
            return "decorated answer"

    # Create new factory instance
    new_factory = RAGFactory()

    # Strategy should be available
    assert "decorated_strategy" in new_factory.list_strategies()

    # Can create and use the strategy
    strategy = new_factory.create_strategy("decorated_strategy")
    assert isinstance(strategy, DecoratedStrategy)


@pytest.mark.integration
def test_full_rag_workflow(factory: RAGFactory) -> None:
    """Test complete RAG workflow from registration to answer generation."""
    factory.register_strategy("dummy", DummyStrategy)

    # Create strategy
    config = {"chunk_size": 512, "chunk_overlap": 50, "top_k": 5}
    strategy = factory.create_strategy("dummy", config)

    # Prepare documents
    documents = [
        {"text": "RAG stands for Retrieval-Augmented Generation", "id": "doc1"},
        {"text": "RAG combines retrieval and generation", "id": "doc2"},
        {"text": "RAG improves LLM accuracy", "id": "doc3"},
    ]
    prepared = strategy.prepare_data(documents)

    assert len(prepared.chunks) == 3
    assert len(prepared.embeddings) == 3

    # Retrieve relevant chunks
    query = "What is RAG?"
    chunks = strategy.retrieve(query, top_k=2)

    assert len(chunks) == 2
    assert all(c.score > 0 for c in chunks)

    # Generate answer
    answer = strategy.process_query(query, chunks)

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert query in answer


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_retrieve_integration(factory: RAGFactory) -> None:
    """Test async retrieve in integration context."""
    factory.register_strategy("dummy", DummyStrategy)

    strategy = factory.create_strategy("dummy", {"top_k": 5})
    strategy.initialize(StrategyConfig(top_k=5))

    # Test async retrieve
    chunks = await strategy.aretrieve("async query", top_k=3)

    assert len(chunks) == 3
    assert all(isinstance(c, Chunk) for c in chunks)
