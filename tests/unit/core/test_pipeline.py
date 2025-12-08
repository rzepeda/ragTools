import pytest
from unittest.mock import Mock, AsyncMock
from typing import Set, Dict, Any

from rag_factory.core.pipeline import IndexingPipeline, RetrievalPipeline
from rag_factory.core.indexing_interface import IIndexingStrategy, IndexingContext
from rag_factory.core.retrieval_interface import IRetrievalStrategy, RetrievalContext
from rag_factory.core.capabilities import IndexCapability, IndexingResult
from rag_factory.services.dependencies import StrategyDependencies, ServiceDependency

# Helper to create strategy classes with specific names
def create_mock_strategy_class(name: str):
    class MockStrategy(IIndexingStrategy):
        def __init__(self, produces: Set[IndexCapability], metadata: Dict[str, Any]):
            self._produces = produces
            self._metadata = metadata
            self.process_called = False
            deps = Mock(spec=StrategyDependencies)
            # Bypass validation for test
            self.config = {}
            self.deps = deps
            
        def produces(self) -> Set[IndexCapability]:
            return self._produces

        def requires_services(self) -> Set[ServiceDependency]:
            return set()

        async def process(self, documents, context) -> IndexingResult:
            self.process_called = True
            return IndexingResult(
                capabilities=self._produces,
                metadata=self._metadata,
                document_count=len(documents),
                chunk_count=len(documents) * 2
            )
    
    MockStrategy.__name__ = name
    return MockStrategy

@pytest.fixture
def mock_context():
    return Mock(spec=IndexingContext)

@pytest.mark.asyncio
async def test_initialization(mock_context):
    Strategy1 = create_mock_strategy_class("Strategy1")
    s1 = Strategy1({IndexCapability.VECTORS}, {})
    
    pipeline = IndexingPipeline([s1], mock_context)
    assert pipeline.strategies == [s1]
    assert pipeline.context == mock_context

@pytest.mark.asyncio
async def test_get_capabilities_before_execution(mock_context):
    Strategy1 = create_mock_strategy_class("Strategy1")
    Strategy2 = create_mock_strategy_class("Strategy2")
    
    s1 = Strategy1({IndexCapability.VECTORS}, {})
    s2 = Strategy2({IndexCapability.KEYWORDS}, {})
    
    pipeline = IndexingPipeline([s1, s2], mock_context)
    caps = pipeline.get_capabilities()
    assert caps == {IndexCapability.VECTORS, IndexCapability.KEYWORDS}

@pytest.mark.asyncio
async def test_index_sequential_execution(mock_context):
    Strategy1 = create_mock_strategy_class("Strategy1")
    Strategy2 = create_mock_strategy_class("Strategy2")
    
    s1 = Strategy1({IndexCapability.VECTORS}, {"s1": "v1"})
    s2 = Strategy2({IndexCapability.KEYWORDS}, {"s2": "v2"})
    
    pipeline = IndexingPipeline([s1, s2], mock_context)
    documents = [{"text": "doc1"}]
    
    result = await pipeline.index(documents)
    
    # Verify execution order (implicit by result aggregation, but we can check flags)
    assert s1.process_called
    assert s2.process_called
    
    # Verify result aggregation
    assert result.capabilities == {IndexCapability.VECTORS, IndexCapability.KEYWORDS}
    assert result.metadata["Strategy1"] == {"s1": "v1"}
    assert result.metadata["Strategy2"] == {"s2": "v2"}
    
    assert result.document_count == 1
    assert result.chunk_count == 2 # 1 * 2

@pytest.mark.asyncio
async def test_get_capabilities_after_execution(mock_context):
    Strategy1 = create_mock_strategy_class("Strategy1")
    s1 = Strategy1({IndexCapability.VECTORS}, {})
    
    pipeline = IndexingPipeline([s1], mock_context)
    documents = [{"text": "doc"}]
    await pipeline.index(documents)
    
    assert pipeline._last_result is not None
    assert pipeline.get_capabilities() == {IndexCapability.VECTORS}

@pytest.mark.asyncio
async def test_error_handling(mock_context):
    Strategy1 = create_mock_strategy_class("Strategy1")
    s1 = Strategy1({IndexCapability.VECTORS}, {})
    
    # Mock process to raise error
    s1.process = AsyncMock(side_effect=ValueError("Strategy failed"))
    
    pipeline = IndexingPipeline([s1], mock_context)
    documents = [{"text": "doc"}]
    
    with pytest.raises(ValueError, match="Strategy failed"):
        await pipeline.index(documents)


# Helper to create retrieval strategy classes
def create_mock_retrieval_strategy_class(name: str):
    class MockRetrievalStrategy(IRetrievalStrategy):
        def __init__(self, requires: Set[IndexCapability], requires_services: Set[ServiceDependency]):
            self._requires = requires
            self._requires_services = requires_services
            self.retrieve_called = False
            self.last_query = None
            deps = Mock(spec=StrategyDependencies)
            # Bypass validation for test
            self.config = {}
            self.deps = deps
            
        def requires(self) -> Set[IndexCapability]:
            return self._requires

        def requires_services(self) -> Set[ServiceDependency]:
            return self._requires_services

        async def retrieve(self, query, context, top_k=10):
            self.retrieve_called = True
            self.last_query = query
            return [f"result_from_{name}"]
    
    MockRetrievalStrategy.__name__ = name
    return MockRetrievalStrategy

@pytest.fixture
def mock_retrieval_context():
    return Mock(spec=RetrievalContext)

@pytest.mark.asyncio
async def test_retrieval_initialization(mock_retrieval_context):
    Strategy1 = create_mock_retrieval_strategy_class("Strategy1")
    s1 = Strategy1({IndexCapability.VECTORS}, {ServiceDependency.DATABASE})
    
    pipeline = RetrievalPipeline([s1], mock_retrieval_context)
    assert pipeline.strategies == [s1]
    assert pipeline.context == mock_retrieval_context

@pytest.mark.asyncio
async def test_retrieval_requirements(mock_retrieval_context):
    Strategy1 = create_mock_retrieval_strategy_class("Strategy1")
    Strategy2 = create_mock_retrieval_strategy_class("Strategy2")
    
    s1 = Strategy1({IndexCapability.VECTORS}, {ServiceDependency.DATABASE})
    s2 = Strategy2({IndexCapability.KEYWORDS}, {ServiceDependency.RERANKER})
    
    pipeline = RetrievalPipeline([s1, s2], mock_retrieval_context)
    
    assert pipeline.get_requirements() == {IndexCapability.VECTORS, IndexCapability.KEYWORDS}
    assert pipeline.get_service_requirements() == {ServiceDependency.DATABASE, ServiceDependency.RERANKER}

@pytest.mark.asyncio
async def test_retrieval_sequential_execution(mock_retrieval_context):
    Strategy1 = create_mock_retrieval_strategy_class("Strategy1")
    Strategy2 = create_mock_retrieval_strategy_class("Strategy2")
    
    s1 = Strategy1(set(), set())
    s2 = Strategy2(set(), set())
    
    pipeline = RetrievalPipeline([s1, s2], mock_retrieval_context)
    
    results = await pipeline.retrieve("test query", top_k=5)
    
    assert s1.retrieve_called
    assert s2.retrieve_called
    assert s1.last_query == "test query"
    assert s2.last_query == "test query"
    # The pipeline returns the result of the last strategy
    assert results == ["result_from_Strategy2"]

@pytest.mark.asyncio
async def test_retrieval_error_handling(mock_retrieval_context):
    Strategy1 = create_mock_retrieval_strategy_class("Strategy1")
    s1 = Strategy1(set(), set())
    
    # Mock retrieve to raise error
    s1.retrieve = AsyncMock(side_effect=ValueError("Retrieval failed"))
    
    pipeline = RetrievalPipeline([s1], mock_retrieval_context)
    
    with pytest.raises(ValueError, match="Retrieval failed"):
        await pipeline.retrieve("query")
