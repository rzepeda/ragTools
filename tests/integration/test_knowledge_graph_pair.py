"""
Integration tests for knowledge-graph-pair strategy configuration.
Tests graph-based retrieval with entity relationships.
"""
import pytest
from unittest.mock import patch
from pathlib import Path

from rag_factory.config.strategy_pair_manager import StrategyPairManager
from rag_factory.core.indexing_interface import IIndexingStrategy, IndexingContext
from rag_factory.core.retrieval_interface import IRetrievalStrategy, RetrievalContext
from rag_factory.core.capabilities import IndexingResult, IndexCapability

# Import strategies
import rag_factory.strategies.knowledge_graph.strategy

# Note: Using centralized mock_registry_with_graph_services fixture from conftest.py
# This fixture includes: embedding, LLM, database, and Neo4j services


@pytest.mark.asyncio
async def test_knowledge_graph_pair_loading(mock_registry_with_graph_services):
    """Test loading and basic functionality of knowledge-graph-pair.
    
    Uses centralized mock_registry_with_graph_services fixture which provides:
    - Mock embedding service (384 dimensions)
    - Mock LLM service for entity extraction
    - Mock database service
    - Mock Neo4j graph database service
    - Mock migration validator
    
    This test verifies:
    1. Strategy pair loads correctly with all required services
    2. Indexing strategy can process documents with graph extraction
    3. Retrieval strategy can query the knowledge graph
    4. All dependencies are properly injected
    """
    with patch('rag_factory.config.strategy_pair_manager.MigrationValidator') as MockValidator:
        mock_validator_instance = MockValidator.return_value
        mock_validator_instance.validate.return_value = (True, [])
        
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "strategies"
        
        manager = StrategyPairManager(
            service_registry=mock_registry_with_graph_services,
            config_dir=str(config_dir)
        )
        
        # Load pair
        indexing, retrieval = manager.load_pair("knowledge-graph-pair")
        
        assert isinstance(indexing, IIndexingStrategy)
        assert isinstance(retrieval, IRetrievalStrategy)
        
        # Verify dependencies
        assert indexing.deps.embedding_service is not None
        assert indexing.deps.database_service is not None
        assert indexing.deps.llm_service is not None
        assert retrieval.deps.embedding_service is not None
        assert retrieval.deps.database_service is not None
        
        # Test Indexing
        docs = [{'id': 'doc1', 'text': 'John works at Google in Mountain View'}]
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
