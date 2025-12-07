"""Integration tests for Knowledge Graph RAG Strategy."""

import pytest
from unittest.mock import Mock

from rag_factory.strategies.knowledge_graph.strategy import KnowledgeGraphRAGStrategy
from rag_factory.strategies.knowledge_graph.config import KnowledgeGraphConfig
from rag_factory.services.llm.base import LLMResponse


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    store = Mock()
    store.index_chunk = Mock()
    store.search = Mock(return_value=[
        {
            "chunk_id": "ml_doc_chunk_0",
            "text": "Python is a popular programming language for Machine Learning.",
            "score": 0.95
        },
        {
            "chunk_id": "ml_doc_chunk_1",
            "text": "Machine Learning is a subset of Artificial Intelligence.",
            "score": 0.85
        }
    ])
    return store


@pytest.fixture
def mock_llm():
    """Mock LLM service for testing."""
    llm = Mock()
    
    # Mock entity extraction response
    entity_response = LLMResponse(
        content='''[
  {"name": "Python", "type": "concept", "description": "Programming language", "confidence": 0.95},
  {"name": "Machine Learning", "type": "concept", "description": "AI technique", "confidence": 0.90},
  {"name": "Artificial Intelligence", "type": "concept", "description": "AI field", "confidence": 0.92}
]''',
        model="test-model",
        provider="test-provider",
        prompt_tokens=100,
        completion_tokens=80,
        total_tokens=180,
        cost=0.002,
        latency=0.6,
        metadata={}
    )
    
    # Mock relationship extraction response
    relationship_response = LLMResponse(
        content='''[
  {"source": "Python", "target": "Machine Learning", "type": "related_to", "description": "Used for ML", "strength": 0.9, "confidence": 0.85},
  {"source": "Machine Learning", "target": "Artificial Intelligence", "type": "is_part_of", "description": "Subset of AI", "strength": 0.95, "confidence": 0.90}
]''',
        model="test-model",
        provider="test-provider",
        prompt_tokens=120,
        completion_tokens=90,
        total_tokens=210,
        cost=0.003,
        latency=0.7,
        metadata={}
    )
    
    # Alternate between entity and relationship responses
    import itertools
    responses = itertools.cycle([entity_response, relationship_response])
    llm.complete = Mock(side_effect=responses)
    
    return llm


@pytest.fixture
def config():
    """Configuration for testing."""
    return KnowledgeGraphConfig(
        graph_backend="memory",
        vector_weight=0.6,
        graph_weight=0.4,
        max_graph_hops=2
    )


@pytest.mark.integration
def test_knowledge_graph_workflow(mock_vector_store, mock_llm, config):
    """Test complete knowledge graph RAG workflow."""
    strategy = KnowledgeGraphRAGStrategy(
        vector_store_service=mock_vector_store,
        llm_service=mock_llm,
        config=config
    )
    
    # Index document
    document = """Python is a popular programming language for Machine Learning.

Machine Learning is a subset of Artificial Intelligence.

Artificial Intelligence enables computers to learn and make decisions."""
    
    result = strategy.index_document(document, "ml_doc")
    
    # Verify indexing results
    assert result["document_id"] == "ml_doc"
    assert result["total_chunks"] == 3
    assert result["total_entities"] > 0
    
    # Verify entities were extracted
    graph_stats = strategy.graph_store.get_stats()
    assert graph_stats["num_entities"] > 0
    
    # Verify vector store was called
    assert mock_vector_store.index_chunk.called


@pytest.mark.integration
def test_hybrid_retrieval(mock_vector_store, mock_llm, config):
    """Test hybrid retrieval combining vector and graph."""
    strategy = KnowledgeGraphRAGStrategy(
        vector_store_service=mock_vector_store,
        llm_service=mock_llm,
        config=config
    )
    
    # Index document
    document = """Python is a popular programming language for Machine Learning.

Machine Learning is a subset of Artificial Intelligence."""
    
    strategy.index_document(document, "ml_doc")
    
    # Retrieve with hybrid search
    results = strategy.retrieve("What is Machine Learning?", top_k=3)
    
    assert len(results) > 0
    assert all("related_entities" in r for r in results)
    assert all("score" in r for r in results)  # Combined score is returned as 'score'
    assert all("vector_score" in r for r in results)
    assert all("graph_score" in r for r in results)


@pytest.mark.integration
def test_relationship_queries(mock_vector_store, mock_llm, config):
    """Test relationship-based queries."""
    strategy = KnowledgeGraphRAGStrategy(
        vector_store_service=mock_vector_store,
        llm_service=mock_llm,
        config=config
    )
    
    document = """Climate change causes rising temperatures.

Rising temperatures lead to glacier melting.

Glacier melting results in sea level rise."""
    
    strategy.index_document(document, "climate_doc")
    
    # Query about causal relationships
    results = strategy.retrieve("What causes sea level rise?", top_k=3)
    
    assert len(results) > 0
    # Should leverage graph relationships to connect concepts


@pytest.mark.integration
def test_graph_statistics(mock_vector_store, mock_llm, config):
    """Test graph statistics tracking."""
    strategy = KnowledgeGraphRAGStrategy(
        vector_store_service=mock_vector_store,
        llm_service=mock_llm,
        config=config
    )
    
    document = "Python is used for Machine Learning. Machine Learning is part of AI."
    
    result = strategy.index_document(document, "doc1")
    
    # Check graph stats
    stats = result["graph_stats"]
    assert "num_entities" in stats
    assert "num_relationships" in stats
    assert "density" in stats
    assert stats["num_entities"] > 0
