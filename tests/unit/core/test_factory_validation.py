"""
Unit tests for RAGFactory validation logic.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch
from typing import Set, List, Dict, Any

# We don't import RAGFactory here to avoid triggering imports that need numpy

@pytest.fixture
def mock_dependencies():
    """Mock dependencies before importing RAGFactory."""
    with patch.dict(sys.modules):
        # Mock numpy
        sys.modules["numpy"] = MagicMock()
        
        # Mock services that cause issues
        sys.modules["rag_factory.services"] = MagicMock()
        sys.modules["rag_factory.services.onnx"] = MagicMock()
        sys.modules["rag_factory.services.onnx.embedding"] = MagicMock()
        sys.modules["rag_factory.services.embedding"] = MagicMock()
        sys.modules["rag_factory.services.embedding.providers"] = MagicMock()
        sys.modules["rag_factory.services.embedding.providers.onnx_local"] = MagicMock()
        sys.modules["rag_factory.services.embedding.service"] = MagicMock()
        sys.modules["rag_factory.services.api"] = MagicMock()
        sys.modules["rag_factory.services.database"] = MagicMock()
        sys.modules["rag_factory.services.local"] = MagicMock()
        
        # We need to ensure interfaces and dependencies are available
        # We can let them be imported normally if they don't depend on numpy
        # But if they do, we might need to mock them or ensure their deps are mocked
        
        # Import inside the patch context
        from rag_factory.factory import RAGFactory
        from rag_factory.core.capabilities import IndexCapability, ValidationResult
        from rag_factory.core.pipeline import IndexingPipeline, RetrievalPipeline
        from rag_factory.core.indexing_interface import IIndexingStrategy
        from rag_factory.core.retrieval_interface import IRetrievalStrategy
        from rag_factory.services.dependencies import ServiceDependency
        
        yield {
            "RAGFactory": RAGFactory,
            "IndexCapability": IndexCapability,
            "ValidationResult": ValidationResult,
            "IndexingPipeline": IndexingPipeline,
            "RetrievalPipeline": RetrievalPipeline,
            "IIndexingStrategy": IIndexingStrategy,
            "IRetrievalStrategy": IRetrievalStrategy,
            "ServiceDependency": ServiceDependency
        }

# Mock Strategies need to be defined dynamically or using the imported classes
# We'll define them inside the test or helper function using the classes from fixture

@pytest.fixture
def factory(mock_dependencies):
    RAGFactory = mock_dependencies["RAGFactory"]
    return RAGFactory()

@pytest.fixture
def indexing_pipeline(mock_dependencies):
    IndexCapability = mock_dependencies["IndexCapability"]
    IndexingPipeline = mock_dependencies["IndexingPipeline"]
    IIndexingStrategy = mock_dependencies["IIndexingStrategy"]
    ServiceDependency = mock_dependencies["ServiceDependency"]
    
    class MockIndexingStrategy(IIndexingStrategy):
        def __init__(self, produces_caps: Set, requires_svcs: Set = None):
            self._produces = produces_caps
            self._requires_svcs = requires_svcs or set()
            
        def produces(self) -> Set:
            return self._produces
            
        def requires_services(self) -> Set:
            return self._requires_svcs
            
        async def process(self, documents: List[Dict[str, Any]], context: Any) -> Any:
            pass

    strategy = MockIndexingStrategy({IndexCapability.VECTORS})
    pipeline = MagicMock(spec=IndexingPipeline)
    pipeline.strategies = [strategy]
    pipeline.get_capabilities.return_value = {IndexCapability.VECTORS}
    return pipeline

@pytest.fixture
def retrieval_pipeline(mock_dependencies):
    IndexCapability = mock_dependencies["IndexCapability"]
    RetrievalPipeline = mock_dependencies["RetrievalPipeline"]
    IRetrievalStrategy = mock_dependencies["IRetrievalStrategy"]
    ServiceDependency = mock_dependencies["ServiceDependency"]
    
    class MockRetrievalStrategy(IRetrievalStrategy):
        def __init__(self, requires_caps: Set, requires_svcs: Set = None):
            self._requires = requires_caps
            self._requires_svcs = requires_svcs or set()
            
        def requires(self) -> Set:
            return self._requires
            
        def requires_services(self) -> Set:
            return self._requires_svcs
            
        async def retrieve(self, query: str, context: Any, top_k: int) -> List[Any]:
            return []

    strategy = MockRetrievalStrategy({IndexCapability.VECTORS})
    pipeline = MagicMock(spec=RetrievalPipeline)
    pipeline.strategies = [strategy]
    pipeline.get_requirements.return_value = {IndexCapability.VECTORS}
    pipeline.get_service_requirements.return_value = set()
    return pipeline

def test_validate_compatibility_success(factory, indexing_pipeline, retrieval_pipeline, mock_dependencies):
    """Test validation passes when capabilities match requirements."""
    result = factory.validate_compatibility(indexing_pipeline, retrieval_pipeline)
    assert result.is_valid
    assert not result.missing_capabilities

def test_validate_compatibility_failure(factory, indexing_pipeline, retrieval_pipeline, mock_dependencies):
    """Test validation fails when capabilities are missing."""
    IndexCapability = mock_dependencies["IndexCapability"]
    
    # Retrieval requires KEYWORDS but indexing only provides VECTORS
    retrieval_pipeline.get_requirements.return_value = {IndexCapability.VECTORS, IndexCapability.KEYWORDS}
    
    result = factory.validate_compatibility(indexing_pipeline, retrieval_pipeline)
    assert not result.is_valid
    assert IndexCapability.KEYWORDS in result.missing_capabilities
    assert "Missing capabilities" in result.message

def test_validate_pipeline_success(factory, indexing_pipeline, retrieval_pipeline):
    """Test full pipeline validation passes."""
    result = factory.validate_pipeline(indexing_pipeline, retrieval_pipeline)
    assert result.is_valid

def test_validate_pipeline_missing_service(factory, indexing_pipeline, retrieval_pipeline, mock_dependencies):
    """Test validation fails when service is missing."""
    ServiceDependency = mock_dependencies["ServiceDependency"]
    
    # Retrieval requires EMBEDDING service
    retrieval_pipeline.get_service_requirements.return_value = {ServiceDependency.EMBEDDING}
    
    # Factory has no services
    result = factory.validate_pipeline(indexing_pipeline, retrieval_pipeline)
    
    assert not result.is_valid
    assert ServiceDependency.EMBEDDING in result.missing_services
    assert "Missing services" in result.message

def test_auto_select_retrieval(factory, mock_dependencies):
    """Test auto-selection of compatible strategies."""
    IndexCapability = mock_dependencies["IndexCapability"]
    IRetrievalStrategy = mock_dependencies["IRetrievalStrategy"]
    IndexingPipeline = mock_dependencies["IndexingPipeline"]
    
    class MockRetrievalStrategy(IRetrievalStrategy):
        def __init__(self, requires_caps: Set, requires_svcs: Set = None):
            self._requires = requires_caps
            self._requires_svcs = requires_svcs or set()
            
        def requires(self) -> Set:
            return self._requires
            
        def requires_services(self) -> Set:
            return self._requires_svcs
            
        async def retrieve(self, query: str, context: Any, top_k: int) -> List[Any]:
            return []

    # Register mock strategies
    class VectorRetriever(MockRetrievalStrategy):
        def __init__(self, config=None, dependencies=None):
            super().__init__({IndexCapability.VECTORS})
            
    class KeywordRetriever(MockRetrievalStrategy):
        def __init__(self, config=None, dependencies=None):
            super().__init__({IndexCapability.KEYWORDS})
            
    factory.register_strategy("vector_retriever", VectorRetriever)
    factory.register_strategy("keyword_retriever", KeywordRetriever)
    
    # Mock indexing pipeline providing VECTORS
    indexing_pipeline = MagicMock(spec=IndexingPipeline)
    indexing_pipeline.get_capabilities.return_value = {IndexCapability.VECTORS}
    
    # Should select vector_retriever but not keyword_retriever
    selected = factory.auto_select_retrieval(indexing_pipeline)
    assert "vector_retriever" in selected
    assert "keyword_retriever" not in selected

def test_auto_select_retrieval_preference(factory, mock_dependencies):
    """Test auto-selection respects preferences."""
    IndexCapability = mock_dependencies["IndexCapability"]
    IRetrievalStrategy = mock_dependencies["IRetrievalStrategy"]
    IndexingPipeline = mock_dependencies["IndexingPipeline"]
    
    class MockRetrievalStrategy(IRetrievalStrategy):
        def __init__(self, requires_caps: Set, requires_svcs: Set = None):
            self._requires = requires_caps
            self._requires_svcs = requires_svcs or set()
            
        def requires(self) -> Set:
            return self._requires
            
        def requires_services(self) -> Set:
            return self._requires_svcs
            
        async def retrieve(self, query: str, context: Any, top_k: int) -> List[Any]:
            return []

    class VectorRetriever(MockRetrievalStrategy):
        def __init__(self, config=None, dependencies=None):
            super().__init__({IndexCapability.VECTORS})
            
    factory.register_strategy("vector_retriever_1", VectorRetriever)
    factory.register_strategy("vector_retriever_2", VectorRetriever)
    
    indexing_pipeline = MagicMock(spec=IndexingPipeline)
    indexing_pipeline.get_capabilities.return_value = {IndexCapability.VECTORS}
    
    # Prefer 2
    selected = factory.auto_select_retrieval(indexing_pipeline, preferred_strategies=["vector_retriever_2"])
    
    # Should list 2 first
    assert selected[0] == "vector_retriever_2"
    assert "vector_retriever_1" in selected

def test_consistency_checking_called(factory, indexing_pipeline, retrieval_pipeline):
    """Test that consistency checker is invoked."""
    with patch.object(factory.consistency_checker, 'check_indexing_strategy') as mock_check_idx:
        with patch.object(factory.consistency_checker, 'check_retrieval_strategy') as mock_check_ret:
            factory.validate_compatibility(indexing_pipeline, retrieval_pipeline)
            
            assert mock_check_idx.called
            assert mock_check_ret.called
