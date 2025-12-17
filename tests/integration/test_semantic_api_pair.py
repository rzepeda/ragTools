"""
Integration tests for semantic-api-pair strategy configuration.
Tests OpenAI/Cohere API-based embeddings with semantic retrieval.
"""
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

@pytest.mark.asyncio
async def test_semantic_api_pair_loading(mock_registry_with_services):
    """Test loading and basic functionality of semantic-api-pair."""
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
        indexing, retrieval = manager.load_pair("semantic-api-pair")
        
        assert isinstance(indexing, IIndexingStrategy)
        assert isinstance(retrieval, IRetrievalStrategy)
        
        # Verify dependencies
        assert indexing.deps.embedding_service is not None
        assert indexing.deps.database_service is not None
        assert retrieval.deps.embedding_service is not None
        assert retrieval.deps.database_service is not None
        
        # Test Indexing
        docs = [{'id': 'doc1', 'text': 'Sample API text'}]
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
        chunks = await retrieval.retrieve("API query", retrieval_context)
        
        assert len(chunks) == 1
        assert chunks[0].text == "API-based content"
        
        # Verify embedding call
        retrieval.deps.embedding_service.embed.assert_called_with("API query")
