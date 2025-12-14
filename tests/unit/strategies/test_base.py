"""Unit tests for RAG Strategy Base Interface."""
import pytest
import inspect
import asyncio
from dataclasses import asdict
from typing import List, Dict, Any

from rag_factory.strategies.base import (
    IRAGStrategy,
    Chunk,
    StrategyConfig,
    PreparedData,
    QueryResult,
)


# TC1.1: Interface Definition Tests
def test_interface_is_abstract():
    """Test that IRAGStrategy cannot be instantiated directly."""
    with pytest.raises(TypeError):
        IRAGStrategy()


def test_interface_requires_all_abstract_methods():
    """Test that concrete class must implement all abstract methods."""
    class IncompleteStrategy(IRAGStrategy):
        def initialize(self, config):
            pass

    with pytest.raises(TypeError):
        IncompleteStrategy()


# TC1.2: Configuration Dataclass Tests
def test_strategy_config_defaults():
    """Test StrategyConfig has correct default values."""
    config = StrategyConfig()
    assert config.chunk_size == 512
    assert config.chunk_overlap == 50
    assert config.top_k == 5
    assert isinstance(config.metadata, dict)


def test_strategy_config_custom_values():
    """Test StrategyConfig accepts custom values."""
    config = StrategyConfig(
        chunk_size=1024,
        top_k=10,
        strategy_name="test_strategy"
    )
    assert config.chunk_size == 1024
    assert config.top_k == 10
    assert config.strategy_name == "test_strategy"


def test_strategy_config_validation():
    """Test StrategyConfig validates parameter ranges."""
    # Should raise error for invalid chunk_size
    with pytest.raises(ValueError):
        StrategyConfig(chunk_size=-1)

    # Should raise error for invalid chunk_overlap
    with pytest.raises(ValueError):
        StrategyConfig(chunk_overlap=-1)

    # Should raise error for invalid top_k
    with pytest.raises(ValueError):
        StrategyConfig(top_k=0)


# TC1.3: Chunk Dataclass Tests
def test_chunk_creation():
    """Test Chunk can be created with all fields."""
    chunk = Chunk(
        text="Sample text",
        metadata={"key": "value"},
        score=0.95,
        source_id="doc_123",
        chunk_id="chunk_456"
    )
    assert chunk.text == "Sample text"
    assert chunk.score == 0.95
    assert chunk.metadata == {"key": "value"}
    assert chunk.source_id == "doc_123"
    assert chunk.chunk_id == "chunk_456"


def test_chunk_serialization():
    """Test Chunk can be serialized to dict."""
    chunk = Chunk(
        text="Sample",
        metadata={},
        score=0.8,
        source_id="doc_1",
        chunk_id="chunk_1"
    )
    chunk_dict = asdict(chunk)
    assert isinstance(chunk_dict, dict)
    assert "text" in chunk_dict
    assert "metadata" in chunk_dict
    assert "score" in chunk_dict
    assert "source_id" in chunk_dict
    assert "chunk_id" in chunk_dict


# TC1.4: PreparedData Dataclass Tests
def test_prepared_data_creation():
    """Test PreparedData can be created with all fields."""
    chunks = [
        Chunk(
            text="chunk1",
            metadata={},
            score=0.0,
            source_id="doc_1",
            chunk_id="chunk_1"
        )
    ]
    prepared = PreparedData(
        chunks=chunks,
        embeddings=[[0.1, 0.2, 0.3]],
        index_metadata={"index_type": "faiss"}
    )
    assert len(prepared.chunks) == 1
    assert len(prepared.embeddings) == 1
    assert prepared.index_metadata["index_type"] == "faiss"


# TC1.5: QueryResult Dataclass Tests
def test_query_result_creation():
    """Test QueryResult can be created with all fields."""
    chunks_used = [
        Chunk(
            text="relevant chunk",
            metadata={},
            score=0.9,
            source_id="doc_1",
            chunk_id="chunk_1"
        )
    ]
    result = QueryResult(
        answer="This is the answer",
        chunks_used=chunks_used,
        metadata={"execution_time": 0.5},
        strategy_info={"strategy_name": "test"}
    )
    assert result.answer == "This is the answer"
    assert len(result.chunks_used) == 1
    assert result.metadata["execution_time"] == 0.5
    assert result.strategy_info["strategy_name"] == "test"


# TC1.6: Concrete Implementation Tests
def test_minimal_concrete_implementation():
    """Test a minimal concrete implementation works."""
    from rag_factory.services.dependencies import StrategyDependencies
    
    class MinimalStrategy(IRAGStrategy):
        def requires_services(self):
            from rag_factory.services.dependencies import ServiceDependency
            return set()

        def initialize(self, config: StrategyConfig) -> None:
            self.config = config

        def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
            return PreparedData(chunks=[], embeddings=[], index_metadata={})

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            return ""

    # Instantiate with required arguments
    config = {}
    dependencies = StrategyDependencies()
    strategy = MinimalStrategy(config, dependencies)
    assert isinstance(strategy, IRAGStrategy)


def test_concrete_implementation_initialize():
    """Test concrete implementation can be initialized."""
    from rag_factory.services.dependencies import StrategyDependencies
    
    class TestStrategy(IRAGStrategy):
        def requires_services(self):
            from rag_factory.services.dependencies import ServiceDependency
            return set()

        def initialize(self, config: StrategyConfig):
            self.config = config

        def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
            return PreparedData(chunks=[], embeddings=[], index_metadata={})

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            return ""

    # Instantiate with required arguments
    config_dict = {}
    dependencies = StrategyDependencies()
    strategy = TestStrategy(config_dict, dependencies)
    
    # Test initialize method
    config = StrategyConfig(strategy_name="test")
    strategy.initialize(config)
    assert strategy.config.strategy_name == "test"


# TC1.7: Type Hint Validation Tests
def test_type_hints_present():
    """Test that all methods have proper type hints."""
    sig = inspect.signature(IRAGStrategy.retrieve)
    assert sig.return_annotation == List[Chunk]


def test_async_method_signature():
    """Test async method is properly defined."""
    assert inspect.iscoroutinefunction(IRAGStrategy.aretrieve)



def test_requires_services_type_hints():
    """Test requires_services method has proper type hints."""
    from rag_factory.services.dependencies import ServiceDependency
    from typing import Set
    sig = inspect.signature(IRAGStrategy.requires_services)
    # Should return a Set of ServiceDependency
    assert sig.return_annotation == Set[ServiceDependency]


def test_prepare_data_type_hints():
    """Test prepare_data method has proper type hints."""
    sig = inspect.signature(IRAGStrategy.prepare_data)
    assert sig.return_annotation == PreparedData


def test_process_query_type_hints():
    """Test process_query method has proper type hints."""
    sig = inspect.signature(IRAGStrategy.process_query)
    assert sig.return_annotation == str
