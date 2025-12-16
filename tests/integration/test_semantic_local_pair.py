import pytest
import os
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from rag_factory.config.strategy_pair_manager import StrategyPairManager
from rag_factory.registry.service_registry import ServiceRegistry
from rag_factory.core.indexing_interface import IIndexingStrategy, IndexingContext
from rag_factory.core.retrieval_interface import IRetrievalStrategy, RetrievalContext
from rag_factory.core.capabilities import IndexingResult, IndexCapability
from rag_factory.services.dependencies import ServiceDependency

# Import strategies to ensure registration
import rag_factory.strategies.indexing.vector_embedding
import rag_factory.strategies.retrieval.semantic_retriever

@pytest.fixture
def mock_registry():
    # We use a dummy file path or existing one.
    # To avoid loading real config which might have validation errors or env var issues in test env,
    # we can mock load_config or config_path.
    # But for now, let's just instantiate.
    try:
        registry = ServiceRegistry()
    except Exception:
        # If loading fails (e.g. env vars), we can patch _load_config or use empty config
        with patch('rag_factory.registry.service_registry.ServiceRegistry._load_config'):
             registry = ServiceRegistry()
             registry.config = {'services': {}} 

    # Mock Embedding Service
    embedding_service = Mock()
    embedding_service.embed = AsyncMock(return_value=[0.1] * 384)
    embedding_service.embed_batch = AsyncMock(return_value=[[0.1] * 384])
    embedding_service.get_dimension.return_value = 384
    
    # Mock Database Service
    db_service = Mock()
    db_service.get_chunks_for_documents = AsyncMock(return_value=[
        {'id': 'chunk1', 'text': 'content', 'metadata': {}}
    ])
    db_service.store_chunks = AsyncMock()
    db_service.search_chunks = AsyncMock(return_value=[
        {'id': 'chunk1', 'text': 'content', 'score': 0.9, 'metadata': {}}
    ])
    # Support get_context
    db_service.get_context = Mock(return_value=db_service)
    
    # Support get_engine().connect() as context manager for MigrationValidator
    mock_engine = Mock()
    mock_connection = Mock()
    mock_connection.__enter__ = Mock(return_value=mock_connection)
    mock_connection.__exit__ = Mock(return_value=None)
    mock_engine.connect.return_value = mock_connection
    db_service.get_engine.return_value = mock_engine
    
    # Inject into _instances
    # Note: services.yaml uses 'embedding_local' key.
    registry._instances["embedding_local"] = embedding_service
    registry._instances["db_main"] = db_service
    
    # Also inject 'db1' because StrategyPairManager might look for it (though I patched it to check db_main too)
    # And inject 'local-onnx-minilm' just in case code resolves by name
    registry._instances["local-onnx-minilm"] = embedding_service
    registry._instances["main-postgres"] = db_service
    registry._instances["db1"] = db_service
    
    return registry

@pytest.mark.asyncio
async def test_semantic_local_pair_loading(mock_registry):
    # Patch MigrationValidator to avoid DB interaction
    with patch('rag_factory.config.strategy_pair_manager.MigrationValidator') as MockValidator:
        # Configure mock validator instance
        mock_validator_instance = MockValidator.return_value
        mock_validator_instance.validate.return_value = (True, [])
        
        # Setup paths
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "strategies"
        
        # Initialize manager
        manager = StrategyPairManager(
            service_registry=mock_registry,
            config_dir=str(config_dir)
        )
        
        # Check that validator was initialized (since we found a DB service)
        # Note: StrategyPairManager inits validator in __init__
        assert manager.migration_validator is not None
        
        # Inject into _instances
        # Note: services.yaml uses 'embedding_local' key.
        # So "embedding_local" is the KEY in services.yaml.
        # The registry usually keys by Service Name (metadata name) OR by the identifier used in registration.
        # `ServiceRegistry.get(name)` retrieves by the name used in `register(name, service)`.
        
        # IF the system loads services.yaml, how does it register them?
        # It likely registers them using the key (e.g. 'embedding_local') or the 'name' field ('local-onnx-minilm')?
        # Let's check `ServiceRegistry` or `ServiceLoading`.
        # I assume it registers by the KEY 'embedding_local' because that's unique in the file.
        # So I should register as 'embedding_local'.
        
        # Now load pair
        indexing, retrieval = manager.load_pair("semantic-local-pair")
        
        assert isinstance(indexing, IIndexingStrategy)
        assert isinstance(retrieval, IRetrievalStrategy)
        
        # Verify dependencies
        assert indexing.deps.embedding_service is not None
        assert indexing.deps.database_service is not None
        assert retrieval.deps.embedding_service is not None
        assert retrieval.deps.database_service is not None
        
        # Test Indexing Process
        docs = [{'id': 'doc1', 'text': 'Sample text'}]
        context = IndexingContext(
            database_service=indexing.deps.database_service,
            config={}
        )
        result = await indexing.process(docs, context)
        
        assert isinstance(result, IndexingResult)
        assert IndexCapability.VECTORS in result.capabilities
        
        # Verify DB interaction
        indexing.deps.database_service.store_chunks.assert_called()
        
        # Test Retrieval
        retrieval_context = RetrievalContext(
            database_service=retrieval.deps.database_service,
            config={}
        )
        chunks = await retrieval.retrieve("query", retrieval_context)
        
        assert len(chunks) == 1
        assert chunks[0].text == "content"
        
        # Verify embedding call
        retrieval.deps.embedding_service.embed.assert_called_with("query")

