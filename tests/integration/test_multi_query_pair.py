"""
Integration tests for multi-query-pair strategy configuration.
Tests LLM-based query expansion with multiple query variants.
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
import rag_factory.strategies.multi_query.strategy

# Note: Using centralized mock_registry_with_llm_services fixture from conftest.py

@pytest.mark.asyncio
async def test_multi_query_pair_loading(mock_registry_with_llm_services):
    """Test loading and basic functionality of multi-query-pair."""
    with patch('rag_factory.config.strategy_pair_manager.MigrationValidator') as MockValidator:
        mock_validator_instance = MockValidator.return_value
        mock_validator_instance.validate.return_value = (True, [])
        
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "strategies"
        
        manager = StrategyPairManager(
            service_registry=mock_registry_with_llm_services,
            config_dir=str(config_dir)
        )
        
        # Load pair
        indexing, retrieval = manager.load_pair("multi-query-pair")
        
        assert isinstance(indexing, IIndexingStrategy)
        assert isinstance(retrieval, IRetrievalStrategy)
        
        # Verify dependencies
        assert indexing.deps.embedding_service is not None
        assert indexing.deps.database_service is not None
        assert retrieval.deps.embedding_service is not None
        assert retrieval.deps.llm_service is not None
        # MultiQueryRAGStrategy uses database_service
        assert hasattr(retrieval.deps, 'database_service')
        assert retrieval.deps.database_service is not None
        
        # Test Indexing
        docs = [{'id': 'doc1', 'text': 'Sample multi-query text'}]
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
        chunks = await retrieval.retrieve("complex query", retrieval_context)
        
        assert len(chunks) >= 1
        # MultiQueryRetriever returns Chunk objects
        assert chunks[0].text == "multi-query content"
