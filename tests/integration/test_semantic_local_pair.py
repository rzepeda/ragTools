import pytest
from unittest.mock import patch
from pathlib import Path

from rag_factory.config.strategy_pair_manager import StrategyPairManager
from rag_factory.core.indexing_interface import IIndexingStrategy, IndexingContext
from rag_factory.core.retrieval_interface import IRetrievalStrategy, RetrievalContext
from rag_factory.core.capabilities import IndexingResult, IndexCapability

# Import strategies to ensure registration
import rag_factory.strategies.indexing.vector_embedding
import rag_factory.strategies.retrieval.semantic_retriever

# Note: Using centralized mock_registry_with_services fixture from conftest.py
# No need to define mock_registry here anymore!

@pytest.mark.asyncio
async def test_semantic_local_pair_loading(mock_registry_with_services):
    """Test loading and execution of semantic-local-pair strategy.
    
    Uses centralized mock_registry_with_services fixture which provides:
    - Mock embedding service (384 dimensions)
    - Mock database service with CRUD operations
    - Mock migration validator
    
    This test verifies:
    1. Strategy pair loads correctly
    2. Indexing strategy processes documents
    3. Retrieval strategy retrieves chunks
    4. Dependencies are properly injected
    """
    # Patch MigrationValidator to avoid DB interaction
    with patch('rag_factory.config.strategy_pair_manager.MigrationValidator') as MockValidator:
        # Configure mock validator instance
        mock_validator_instance = MockValidator.return_value
        mock_validator_instance.validate.return_value = (True, [])
        
        # Setup paths
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "strategies"
        
        # Initialize manager with centralized mock registry
        manager = StrategyPairManager(
            service_registry=mock_registry_with_services,
            config_dir=str(config_dir)
        )
        
        # Check that validator was initialized
        assert manager.migration_validator is not None
        
        # Load strategy pair
        indexing, retrieval = manager.load_pair("semantic-local-pair")
        
        assert isinstance(indexing, IIndexingStrategy)
        assert isinstance(retrieval, IRetrievalStrategy)
        
        # Verify dependencies are injected
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
        assert chunks[0].text == "mock content"  # Updated to match centralized mock
        
        # Verify embedding call
        retrieval.deps.embedding_service.embed.assert_called_with("query")
