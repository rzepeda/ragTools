"""
Integration tests for fine-tuned-embeddings-pair strategy configuration.
Tests custom fine-tuned embedding models.
"""
import pytest
from unittest.mock import patch
from pathlib import Path

from rag_factory.config.strategy_pair_manager import StrategyPairManager
from rag_factory.core.indexing_interface import IIndexingStrategy, IndexingContext
from rag_factory.core.retrieval_interface import IRetrievalStrategy, RetrievalContext
from rag_factory.core.capabilities import IndexingResult, IndexCapability

# Import strategies
import rag_factory.strategies.indexing.vector_embedding
import rag_factory.strategies.retrieval.semantic_retriever

# Note: Using centralized mock_registry_with_services fixture from conftest.py

@pytest.mark.asyncio
async def test_fine_tuned_embeddings_pair_loading(mock_registry_with_services):
    """Test loading and basic functionality of fine-tuned-embeddings-pair."""
    with patch('rag_factory.config.strategy_pair_manager.MigrationValidator') as MockValidator:
        mock_validator_instance = MockValidator.return_value
        mock_validator_instance.validate.return_value = (True, [])
        
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "strategies"
        
        manager = StrategyPairManager(
            service_registry=mock_registry_with_services,
            config_dir=str(config_dir)
        )
        
        # Load pair
        indexing, retrieval = manager.load_pair("fine-tuned-embeddings-pair")
        
        assert isinstance(indexing, IIndexingStrategy)
        assert isinstance(retrieval, IRetrievalStrategy)
        
        # Verify dependencies
        assert indexing.deps.embedding_service is not None
        assert indexing.deps.database_service is not None
        assert retrieval.deps.embedding_service is not None
        assert retrieval.deps.database_service is not None
        
        # Verify custom embedding dimensions
        assert indexing.deps.embedding_service.get_dimension() == 384  # Updated to match centralized mock
        
        # Test Indexing
        docs = [{'id': 'doc1', 'text': 'Sample fine-tuned embedding text'}]
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
        chunks = await retrieval.retrieve("query", retrieval_context)
        
        assert len(chunks) == 1
        assert chunks[0].text == "mock content"  # Updated to match centralized mock
