"""
Integration tests for contextual-retrieval-pair strategy configuration.
Tests LLM-enriched chunks with contextual information.
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
import rag_factory.strategies.contextual.strategy
import rag_factory.strategies.retrieval.semantic_retriever


@pytest.fixture
def mock_registry():
    """Create mock service registry for contextual retrieval strategy."""
    try:
        registry = ServiceRegistry()
    except Exception:
        with patch('rag_factory.registry.service_registry.ServiceRegistry._load_config'):
            registry = ServiceRegistry()
            registry.config = {'services': {}}
    
    # Mock Embedding Service
    embedding_service = Mock()
    embedding_service.embed = AsyncMock(return_value=[0.1] * 384)
    embedding_service.embed_batch = AsyncMock(return_value=[[0.1] * 384])
    embedding_service.get_dimension.return_value = 384
    
    # Mock LLM Service - returns contextual information
    llm_service = Mock()
    llm_service.generate = AsyncMock(return_value="This chunk discusses contextual information about the topic.")
    llm_service.agenerate = AsyncMock(return_value="This chunk discusses contextual information about the topic.")
    
    # Mock Database Service
    db_service = Mock()
    db_service.get_chunks_for_documents = AsyncMock(return_value=[
        {'id': 'chunk1', 'text': 'contextual content', 'metadata': {'context': 'enriched'}}
    ])
    db_service.store_chunks = AsyncMock()
    db_service.search_chunks = AsyncMock(return_value=[
        {'id': 'chunk1', 'text': 'contextual content', 'score': 0.92, 'metadata': {'context': 'enriched'}}
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
    registry._instances["embedding_local"] = embedding_service
    registry._instances["llm_local"] = llm_service
    registry._instances["db_main"] = db_service
    registry._instances["local-onnx-minilm"] = embedding_service
    registry._instances["local-llama"] = llm_service
    registry._instances["main-postgres"] = db_service
    
    return registry


@pytest.mark.asyncio
async def test_contextual_retrieval_pair_loading(mock_registry):
    """Test loading and basic functionality of contextual-retrieval-pair."""
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
        indexing, retrieval = manager.load_pair("contextual-retrieval-pair")
        
        assert isinstance(indexing, IIndexingStrategy)
        assert isinstance(retrieval, IRetrievalStrategy)
        
        # Verify dependencies
        assert indexing.deps.embedding_service is not None
        assert indexing.deps.llm_service is not None
        assert indexing.deps.database_service is not None
        assert retrieval.deps.embedding_service is not None
        assert retrieval.deps.database_service is not None
        
        # Test Indexing
        docs = [{'id': 'doc1', 'text': 'Sample contextual text that will be enriched with LLM-generated context'}]
        context = IndexingContext(
            database_service=indexing.deps.database_service,
            config={}
        )
        result = await indexing.process(docs, context)
        
        assert isinstance(result, IndexingResult)
        assert IndexCapability.VECTORS in result.capabilities
        
        # Test Retrieval
        retrieval_context = RetrievalContext(
            database_service=retrieval.deps.database_service,
            config={}
        )
        chunks = await retrieval.retrieve("contextual query", retrieval_context)
        
        assert len(chunks) >= 1
        assert chunks[0].text == "contextual content"
