"""
Integration tests for keyword-pair strategy configuration.
Tests BM25 keyword-based search without embeddings.
"""
import pytest
from unittest.mock import patch
from pathlib import Path

from rag_factory.config.strategy_pair_manager import StrategyPairManager
from rag_factory.core.indexing_interface import IIndexingStrategy, IndexingContext
from rag_factory.core.retrieval_interface import IRetrievalStrategy, RetrievalContext
from rag_factory.core.capabilities import IndexingResult, IndexCapability

# Import strategies
import rag_factory.strategies.indexing.keyword_indexing

# Note: Using centralized mock_registry_with_services fixture from conftest.py

@pytest.mark.asyncio
async def test_keyword_pair_loading(mock_registry_with_services):
    """Test loading and basic functionality of keyword-pair."""
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
        chunks = await retrieval.retrieve("query", retrieval_context)
        
        assert len(chunks) == 1
        assert chunks[0].text == "mock content"  # Updated to match centralized mock
