"""Unit tests for hybrid retriever."""

import pytest
from unittest.mock import Mock

from rag_factory.strategies.knowledge_graph.hybrid_retriever import HybridRetriever
from rag_factory.strategies.knowledge_graph.models import (
    Entity,
    EntityType,
    GraphTraversalResult
)
from rag_factory.strategies.knowledge_graph.config import KnowledgeGraphConfig


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    store = Mock()
    store.search.return_value = [
        {"chunk_id": "c1", "text": "Text about Python", "score": 0.95},
        {"chunk_id": "c2", "text": "Text about ML", "score": 0.85}
    ]
    return store


@pytest.fixture
def mock_graph_store():
    """Mock graph store for testing."""
    store = Mock()
    store.entities = {
        "e1": Entity(
            id="e1",
            name="Python",
            type=EntityType.CONCEPT,
            confidence=1.0,
            source_chunks=["c1"]
        ),
        "e2": Entity(
            id="e2",
            name="ML",
            type=EntityType.CONCEPT,
            confidence=1.0,
            source_chunks=["c2"]
        )
    }
    store.traverse.return_value = GraphTraversalResult(
        entities=list(store.entities.values()),
        relationships=[],
        paths=[["e1", "e2"]],
        scores={"e1": 0.8, "e2": 0.7}
    )
    return store


@pytest.fixture
def config():
    """Configuration for testing."""
    return KnowledgeGraphConfig(
        vector_weight=0.6,
        graph_weight=0.4,
        max_graph_hops=2
    )


@pytest.fixture
def hybrid_retriever(mock_vector_store, mock_graph_store, config):
    """Hybrid retriever instance for testing."""
    return HybridRetriever(mock_vector_store, mock_graph_store, config)


def test_hybrid_retrieval(hybrid_retriever, mock_vector_store, mock_graph_store):
    """Test hybrid retrieval combines vector and graph."""
    results = hybrid_retriever.retrieve("test query", top_k=2)
    
    assert len(results) > 0
    assert all(hasattr(r, "combined_score") for r in results)
    assert all(hasattr(r, "vector_score") for r in results)
    assert all(hasattr(r, "graph_score") for r in results)
    
    # Verify vector store was called
    mock_vector_store.search.assert_called_once()


def test_score_combination(hybrid_retriever):
    """Test that scores are combined correctly."""
    results = hybrid_retriever.retrieve("test query", top_k=2)
    
    for result in results:
        # Combined score should be weighted average
        expected = (
            hybrid_retriever.vector_weight * result.vector_score +
            hybrid_retriever.graph_weight * result.graph_score
        )
        assert abs(result.combined_score - expected) < 0.001


def test_empty_vector_results(hybrid_retriever, mock_vector_store):
    """Test handling of empty vector results."""
    mock_vector_store.search.return_value = []
    
    results = hybrid_retriever.retrieve("test query", top_k=2)
    
    assert len(results) == 0


def test_entities_in_chunks(hybrid_retriever, mock_graph_store):
    """Test finding entities in chunks."""
    entities = hybrid_retriever._get_entities_in_chunks(["c1", "c2"])
    
    assert len(entities) == 2
    assert any(e.name == "Python" for e in entities)
    assert any(e.name == "ML" for e in entities)


def test_graph_expansion(hybrid_retriever, mock_graph_store):
    """Test graph expansion from vector results."""
    results = hybrid_retriever.retrieve("test query", top_k=2)
    
    # Should have called graph traversal
    mock_graph_store.traverse.assert_called_once()
    
    # Results should have related entities
    assert all(len(r.related_entities) > 0 for r in results)


def test_top_k_limiting(hybrid_retriever):
    """Test that results are limited to top_k."""
    results = hybrid_retriever.retrieve("test query", top_k=1)
    
    assert len(results) == 1


def test_score_ordering(hybrid_retriever):
    """Test that results are ordered by combined score."""
    results = hybrid_retriever.retrieve("test query", top_k=2)
    
    # Results should be in descending order
    for i in range(len(results) - 1):
        assert results[i].combined_score >= results[i + 1].combined_score
