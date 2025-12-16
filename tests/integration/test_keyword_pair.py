"""
Integration tests for keyword-pair strategy configuration.
Tests BM25 keyword-based search without embeddings.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from rag_factory.config.strategy_pair_manager import StrategyPairManager
from rag_factory.registry.service_registry import ServiceRegistry
from rag_factory.core.indexing_interface import IIndexingStrategy, IndexingContext
from rag_factory.core.retrieval_interface import IRetrievalStrategy, RetrievalContext
from rag_factory.core.capabilities import IndexingResult, IndexCapability

# Import strategies
import rag_factory.strategies.indexing.keyword_indexing


@pytest.fixture
def mock_registry():
    """Create mock service registry for keyword strategy."""
    try:
        registry = ServiceRegistry()
    except Exception:
        with patch('rag_factory.registry.service_registry.ServiceRegistry._load_config'):
            registry = ServiceRegistry()
            registry.config = {'services': {}}
    
    # Mock Database Service (no embedding service needed for keyword search)
    db_service = Mock()
    db_service.get_chunks_for_documents = AsyncMock(return_value=[
        {'id': 'chunk1', 'text': 'keyword search content', 'metadata': {}}
    ])
    db_service.store_chunks = AsyncMock()
    db_service.store_keyword_index = AsyncMock()
    db_service.search_keyword = AsyncMock(return_value=[
        {'id': 'chunk1', 'text': 'keyword search content', 'score': 0.85, 'metadata': {}}
    ])
    db_service.search_chunks = AsyncMock(return_value=[
        {'id': 'chunk1', 'text': 'keyword search content', 'score': 0.85, 'metadata': {}}
    ])
    db_service.get_context = Mock(return_value=db_service)
    
    # Support MigrationValidator
    mock_engine = Mock()
    mock_connection = Mock()
    mock_connection.__enter__ = Mock(return_value=mock_connection)
    mock_connection.__exit__ = Mock(return_value=None)
    mock_engine.connect.return_value = mock_connection
    db_service.get_engine.return_value = mock_engine
    
    # Inject into registry
    registry._instances["db_main"] = db_service
    registry._instances["main-postgres"] = db_service
    
    return registry


@pytest.mark.asyncio
async def test_keyword_pair_loading(mock_registry):
    """Test loading and basic functionality of keyword-pair."""
    with patch('rag_factory.config.strategy_pair_manager.MigrationValidator') as MockValidator:
        mock_validator_instance = MockValidator.return_value
        mock_validator_instance.validate.return_value = (True, [])
        
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "strategies"
        
        manager = StrategyPairManager(
            service_registry=mock_registry,
            config_dir=str(config_dir)
        )
        
        # Load pair
        indexing, retrieval = manager.load_pair("keyword-pair")
        
        assert isinstance(indexing, IIndexingStrategy)
        assert isinstance(retrieval, IRetrievalStrategy)
        
        # Verify dependencies (no embedding service needed)
        assert indexing.deps.database_service is not None
        assert retrieval.deps.database_service is not None
        
        # Test Indexing
        docs = [{'id': 'doc1', 'text': 'Sample keyword search text with important terms'}]
        context = IndexingContext(
            database_service=indexing.deps.database_service,
            config={}
        )
        result = await indexing.process(docs, context)
        
        assert isinstance(result, IndexingResult)
        # Keyword indexing produces KEYWORDS capability
        assert IndexCapability.KEYWORDS in result.capabilities
        
        # Verify DB interaction
        indexing.deps.database_service.store_keyword_index.assert_called()
        
        # Test Retrieval
        retrieval_context = RetrievalContext(
            database_service=retrieval.deps.database_service,
            config={}
        )
        chunks = await retrieval.retrieve("keyword query", retrieval_context)
        
        assert len(chunks) >= 1
        assert chunks[0].text == "keyword search content"
