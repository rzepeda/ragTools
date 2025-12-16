"""
Integration tests for agentic-rag-pair strategy configuration.
Tests agent-based tool selection with dynamic retrieval.
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
import rag_factory.strategies.indexing.vector_embedding
import rag_factory.strategies.agentic.strategy


@pytest.fixture
def mock_registry():
    """Create mock service registry for agentic RAG strategy."""
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
    
    # Mock LLM Service for agent reasoning
    llm_service = Mock()
    llm_service.generate = AsyncMock(return_value="Tool: semantic_search\nQuery: refined query")
    llm_service.agenerate = AsyncMock(return_value="Tool: semantic_search\nQuery: refined query")
    
    # Mock Database Service
    db_service = Mock()
    db_service.get_chunks_for_documents = AsyncMock(return_value=[
        {'id': 'chunk1', 'text': 'agentic content', 'metadata': {}}
    ])
    db_service.store_chunks = AsyncMock()
    db_service.search_chunks = AsyncMock(return_value=[
        {'id': 'chunk1', 'text': 'agentic content', 'score': 0.91, 'metadata': {}}
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
async def test_agentic_rag_pair_loading(mock_registry):
    """Test loading and basic functionality of agentic-rag-pair."""
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
        indexing, retrieval = manager.load_pair("agentic-rag-pair")
        
        assert isinstance(indexing, IIndexingStrategy)
        # AgenticRAGStrategy is a full RAG strategy
        assert retrieval is not None
        
        # Verify dependencies
        assert indexing.deps.embedding_service is not None
        assert indexing.deps.database_service is not None
        
        # Test Indexing
        docs = [{'id': 'doc1', 'text': 'Sample agentic RAG text'}]
        context = IndexingContext(
            database_service=indexing.deps.database_service,
            config={}
        )
        result = await indexing.process(docs, context)
        
        assert isinstance(result, IndexingResult)
        assert IndexCapability.VECTORS in result.capabilities
