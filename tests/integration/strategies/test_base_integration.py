"""Integration tests for RAG Strategy Base Interface."""
import pytest
import asyncio
from typing import List, Dict, Any

from rag_factory.strategies.base import (
    IRAGStrategy,
    Chunk,
    StrategyConfig,
    PreparedData,
)


# IS1.1: Strategy Lifecycle Test
@pytest.mark.integration
def test_strategy_full_lifecycle():
    """Test complete lifecycle: initialize -> prepare -> retrieve."""
    from rag_factory.services.dependencies import StrategyDependencies, ServiceDependency
    from typing import Set
    
    class DummyStrategy(IRAGStrategy):
        def requires_services(self) -> Set[ServiceDependency]:
            return set()  # No services required for this dummy
        
        def __init__(self, config: Dict[str, Any], dependencies: StrategyDependencies):
            super().__init__(config, dependencies)
            self.data_prepared = False # Still track for this specific test's logic

        def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
            self.data_prepared = True
            chunks = [
                Chunk(
                    text=doc.get("text", ""),
                    metadata=doc.get("metadata", {}),
                    score=0.0,
                    source_id=f"doc_{i}",
                    chunk_id=f"chunk_{i}"
                )
                for i, doc in enumerate(documents)
            ]
            return PreparedData(
                chunks=chunks,
                embeddings=[],
                index_metadata={"status": "prepared"}
            )

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            if not self.data_prepared:
                raise RuntimeError("Data not prepared")
            return [
                Chunk(
                    text=f"Result {i}",
                    metadata={},
                    score=0.9,
                    source_id=f"doc_{i}",
                    chunk_id=f"chunk_{i}"
                )
                for i in range(top_k)
            ]

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            return self.retrieve(query, top_k)

        def process_query(self, query: str, context: List[Chunk]) -> str:
            return f"Processed: {query}"

    # Initialize with new pattern
    config = {"top_k": 3}
    dependencies = StrategyDependencies()
    strategy = DummyStrategy(config=config, dependencies=dependencies)

    # Prepare data
    result = strategy.prepare_data([{"text": "doc1"}])
    assert result.index_metadata["status"] == "prepared"
    assert len(result.chunks) == 1

    # Retrieve
    chunks = strategy.retrieve("test query", top_k=3)
    assert len(chunks) == 3
    assert all(isinstance(c, Chunk) for c in chunks)


# IS1.2: Multiple Strategy Implementation Test
@pytest.mark.integration
def test_multiple_strategies_implement_interface():
    """Test that multiple different strategies can implement the interface."""
    from rag_factory.services.dependencies import StrategyDependencies, ServiceDependency
    from typing import Set
    
    class StrategyA(IRAGStrategy):
        def requires_services(self) -> Set[ServiceDependency]:
            return set()

        def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
            return PreparedData(chunks=[], embeddings=[], index_metadata={"type": "A"})

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            return "Strategy A"

    class StrategyB(IRAGStrategy):
        def requires_services(self) -> Set[ServiceDependency]:
            return set()

        def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
            return PreparedData(chunks=[], embeddings=[], index_metadata={"type": "B"})

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            return []

        def process_query(self, query: str, context: List[Chunk]) -> str:
            return "Strategy B"

    dependencies = StrategyDependencies()
    strategies = [StrategyA(config={}, dependencies=dependencies), StrategyB(config={}, dependencies=dependencies)]
    for strategy in strategies:
        assert isinstance(strategy, IRAGStrategy)

    # Test they work differently
    result_a = strategies[0].prepare_data([])
    result_b = strategies[1].prepare_data([])

    assert result_a.index_metadata["type"] == "A"
    assert result_b.index_metadata["type"] == "B"


# IS1.3: Async Method Integration Test
@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_retrieve_works():
    """Test async retrieve method works correctly."""
    from rag_factory.services.dependencies import StrategyDependencies, ServiceDependency
    from typing import Set
    
    class AsyncStrategy(IRAGStrategy):
        def requires_services(self) -> Set[ServiceDependency]:
            return set()

        def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
            return PreparedData(chunks=[], embeddings=[], index_metadata={})

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            # Simulate async operation
            await asyncio.sleep(0.1)
            return [
                Chunk("text", {}, 0.9, "doc_1", "chunk_1")
            ]

        def process_query(self, query: str, context: List[Chunk]) -> str:
            return ""

    dependencies = StrategyDependencies()
    strategy = AsyncStrategy(config={}, dependencies=dependencies)
    results = await strategy.aretrieve("query", 5)
    assert len(results) == 1
    assert isinstance(results[0], Chunk)


# IS1.4: Error Handling Test
@pytest.mark.integration
def test_strategy_error_handling():
    """Test that strategies properly handle errors."""
    from rag_factory.services.dependencies import StrategyDependencies, ServiceDependency
    from typing import Set
    
    class ErrorStrategy(IRAGStrategy):
        def requires_services(self) -> Set[ServiceDependency]:
            return set()

        def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
            if not documents:
                raise ValueError("No documents provided")
            return PreparedData(chunks=[], embeddings=[], index_metadata={})

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            if not query:
                raise ValueError("Query cannot be empty")
            return []

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            return self.retrieve(query, top_k)

        def process_query(self, query: str, context: List[Chunk]) -> str:
            return ""

    dependencies = StrategyDependencies()
    strategy = ErrorStrategy(config={}, dependencies=dependencies)

    # Test error handling
    with pytest.raises(ValueError, match="No documents provided"):
        strategy.prepare_data([])

    with pytest.raises(ValueError, match="Query cannot be empty"):
        strategy.retrieve("", 5)


# IS1.5: Configuration Override Test
@pytest.mark.integration
def test_strategy_configuration_override():
    """Test that strategies can override configuration at runtime."""
    from rag_factory.services.dependencies import StrategyDependencies, ServiceDependency
    from typing import Set
    
    class ConfigurableStrategy(IRAGStrategy):
        def requires_services(self) -> Set[ServiceDependency]:
            return set()

        def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
            return PreparedData(chunks=[], embeddings=[], index_metadata={})

        def retrieve(self, query: str, top_k: int) -> List[Chunk]:
            # Use parameter top_k instead of config.top_k
            return [
                Chunk(f"text_{i}", {}, 0.9, f"doc_{i}", f"chunk_{i}")
                for i in range(top_k)
            ]

        async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
            return self.retrieve(query, top_k)

        def process_query(self, query: str, context: List[Chunk]) -> str:
            return ""

    dependencies = StrategyDependencies()
    config = {"top_k": 3}
    strategy = ConfigurableStrategy(config=config, dependencies=dependencies)

    # Override top_k at retrieval time
    results = strategy.retrieve("query", top_k=10)
    assert len(results) == 10  # Should use parameter, not config
