"""
Integration tests for late chunking strategy.
"""

import os
import pytest

from rag_factory.strategies.late_chunking.strategy import LateChunkingRAGStrategy
from rag_factory.strategies.late_chunking.models import EmbeddingChunkingMethod

# Get embedding model from environment or use ONNX-compatible default
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "Xenova/all-MiniLM-L6-v2")


class MockVectorStore:
    """Mock vector store for testing."""

    def __init__(self):
        self.chunks = []

    async def store_chunks(self, chunks):
        """Store chunks."""
        for chunk in chunks:
            self.chunks.append({
                "chunk_id": chunk.get("chunk_id", chunk.get("id")),
                "text": chunk.get("text", chunk.get("content", "")),
                "embedding": chunk.get("embedding"),
                "metadata": chunk.get("metadata", {})
            })

    def search(self, query, top_k=5, **kwargs):
        """Return mock search results."""
        # Return first top_k chunks
        results = []
        for chunk in self.chunks[:top_k]:
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": 0.9,
                "metadata": chunk["metadata"]
            })
        return results


class MockDependencies:
    """Mock dependencies for strategy."""
    def __init__(self, database_service):
        self.database_service = database_service


@pytest.fixture
def test_vector_store():
    """Create mock vector store."""
    return MockVectorStore()


@pytest.fixture
def test_dependencies(test_vector_store):
    """Create mock dependencies with vector store."""
    return MockDependencies(test_vector_store)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_late_chunking_workflow(test_dependencies, test_vector_store):
    """Test complete late chunking workflow."""
    config = {
        "model_name": EMBEDDING_MODEL,
        "chunking_method": EmbeddingChunkingMethod.SEMANTIC_BOUNDARY.value,
        "target_chunk_size": 128,
        "compute_coherence_scores": True,

    }

    strategy = LateChunkingRAGStrategy(config, test_dependencies)

    # Index document
    document = """
    Machine learning is a subset of artificial intelligence that enables systems to learn from data.
    Deep learning is a type of machine learning that uses neural networks with many layers.
    Neural networks are inspired by biological neurons in the human brain.
    The training process involves adjusting network weights to minimize error.
    """

    await strategy.index_document(document, "ml_doc")

    # Verify chunks were indexed
    assert len(test_vector_store.chunks) > 0

    # Verify chunk metadata
    for chunk in test_vector_store.chunks:
        assert "document_id" in chunk["metadata"]
        assert chunk["metadata"]["document_id"] == "ml_doc"
        assert "chunking_method" in chunk["metadata"]
        assert chunk["metadata"]["chunking_method"] == "late_chunking"
        assert "coherence_score" in chunk["metadata"]

    # Retrieve
    results = strategy.retrieve("What is machine learning?", top_k=3)

    assert len(results) > 0
    assert all("strategy" in r for r in results)
    assert all(r["strategy"] == "late_chunking" for r in results)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fixed_size_chunking_integration(test_dependencies, test_vector_store):
    """Test integration with fixed-size chunking."""
    config = {
        "model_name": EMBEDDING_MODEL,
        "chunking_method": "fixed_size",
        "target_chunk_size": 50,
        "chunk_overlap_tokens": 10,
        "compute_coherence_scores": False,

    }

    strategy = LateChunkingRAGStrategy(config, test_dependencies)

    document = "This is a test document. " * 20

    await strategy.index_document(document, "test_doc")

    assert len(test_vector_store.chunks) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_adaptive_chunking_integration(test_dependencies, test_vector_store):
    """Test integration with adaptive chunking."""
    config = {
        "model_name": EMBEDDING_MODEL,
        "chunking_method": "adaptive",
        "min_chunk_size": 20,
        "max_chunk_size": 100,

    }

    strategy = LateChunkingRAGStrategy(config, test_dependencies)

    document = """
    Python is a versatile programming language. It is widely used in data science and machine learning.
    Libraries like NumPy and Pandas make data manipulation easy. TensorFlow and PyTorch are popular ML frameworks.
    """

    await strategy.index_document(document, "python_doc")

    assert len(test_vector_store.chunks) > 0

    # Verify chunks respect size constraints
    for chunk in test_vector_store.chunks:
        token_count = chunk["metadata"]["token_count"]
        assert token_count <= config["max_chunk_size"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_documents(test_dependencies, test_vector_store):
    """Test indexing multiple documents."""
    config = {
        "model_name": EMBEDDING_MODEL,
        "chunking_method": "semantic_boundary",

    }

    strategy = LateChunkingRAGStrategy(config, test_dependencies)

    documents = [
        ("First document about AI and machine learning.", "doc1"),
        ("Second document about neural networks and deep learning.", "doc2"),
        ("Third document about data science and analytics.", "doc3")
    ]

    for text, doc_id in documents:
        await strategy.index_document(text, doc_id)

    # Should have chunks from all documents
    assert len(test_vector_store.chunks) >= 3

    # Verify document IDs
    doc_ids = set(chunk["metadata"]["document_id"] for chunk in test_vector_store.chunks)
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids
    assert "doc3" in doc_ids


@pytest.mark.integration
def test_strategy_properties(test_dependencies):
    """Test strategy name and description."""
    config = {
        "model_name": EMBEDDING_MODEL,

    }

    strategy = LateChunkingRAGStrategy(config, test_dependencies)

    assert strategy.name == "late_chunking"
    assert "embed" in strategy.description.lower()
    assert "context" in strategy.description.lower()


@pytest.mark.integration
def test_coherence_scores_computed(test_dependencies, test_vector_store):
    """Test that coherence scores are computed when enabled."""
    config = {
        "model_name": EMBEDDING_MODEL,
        "chunking_method": "semantic_boundary",
        "compute_coherence_scores": True,

    }

    strategy = LateChunkingRAGStrategy(config, test_dependencies)

    document = "Test document with multiple sentences. Each sentence adds more context."

    strategy.index_document(document, "coherence_test")

    # All chunks should have coherence scores
    for chunk in test_vector_store.chunks:
        assert chunk["metadata"]["coherence_score"] is not None
        assert 0.0 <= chunk["metadata"]["coherence_score"] <= 1.0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_short_document(test_dependencies, test_vector_store):
    """Test handling of very short documents."""
    config = {
        "model_name": EMBEDDING_MODEL,

    }

    strategy = LateChunkingRAGStrategy(config, test_dependencies)

    short_doc = "Short."

    await strategy.index_document(short_doc, "short_doc")

    # Should create at least one chunk
    assert len(test_vector_store.chunks) >= 1


@pytest.mark.integration
def test_chunk_embeddings_valid(test_dependencies, test_vector_store):
    """Test that chunk embeddings are valid vectors."""
    config = {
        "model_name": EMBEDDING_MODEL,

    }

    strategy = LateChunkingRAGStrategy(config, test_dependencies)

    document = "Test document for embedding validation."

    strategy.index_document(document, "embed_test")

    for chunk in test_vector_store.chunks:
        embedding = chunk["embedding"]
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, (int, float)) for x in embedding)


@pytest.mark.integration
def test_embedding_quality(test_dependencies, test_vector_store):
    """Test that ONNX embeddings are of good quality."""
    import numpy as np
    
    config = {
        "model_name": EMBEDDING_MODEL,

    }

    strategy = LateChunkingRAGStrategy(config, test_dependencies)

    # Test with sample text
    text = "The quick brown fox jumps over the lazy dog."
    strategy.index_document(text, "quality_test")

    # Get embeddings
    for chunk in test_vector_store.chunks:
        embedding = np.array(chunk["embedding"])
        
        # Check embeddings are not all zeros
        assert np.any(embedding != 0), "Embeddings should not be all zeros"
        
        # Check embeddings are in reasonable range
        assert np.all(np.abs(embedding) < 100), "Embeddings should be in reasonable range"
        
        # Check norm is reasonable (not exploding or vanishing)
        norm = np.linalg.norm(embedding)
        assert norm > 0.1, "Embedding norm should not be too small"
        assert norm < 1000, "Embedding norm should not be too large"

