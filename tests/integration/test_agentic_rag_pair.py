"""
Integration tests for agentic-rag-pair strategy configuration.
Tests agent-based tool selection with dynamic retrieval.
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
import rag_factory.strategies.agentic.strategy

# Note: Using centralized mock_registry_with_llm_services fixture from conftest.py

@pytest.mark.asyncio
async def test_agentic_rag_pair_loading(mock_registry_with_llm_services):
    """Test loading and basic functionality of agentic-rag-pair."""
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
