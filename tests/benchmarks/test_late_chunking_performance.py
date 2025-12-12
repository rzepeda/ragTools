"""
Performance benchmarks for late chunking strategy.
"""

import os
import pytest
import time

from rag_factory.strategies.late_chunking.strategy import LateChunkingRAGStrategy

# Get embedding model from environment or use ONNX-compatible default
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "Xenova/all-mpnet-base-v2")


class MockVectorStore:
    """Mock vector store for benchmarking."""

    def __init__(self):
        self.chunks = []

    def index_chunk(self, chunk_id, text, embedding, metadata):
        """Store chunk."""
        self.chunks.append({
            "chunk_id": chunk_id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata
        })

    def search(self, query, top_k=5, **kwargs):
        """Return mock search results."""
        return []


@pytest.fixture
def benchmark_vector_store():
    """Create mock vector store for benchmarks."""
    return MockVectorStore()


@pytest.mark.benchmark
def test_document_embedding_speed(benchmark_vector_store):
    """Benchmark document embedding speed."""
    config = {
        "model_name": EMBEDDING_MODEL,
        "device": "cpu"
    }

    strategy = LateChunkingRAGStrategy(benchmark_vector_store, config)

    # Generate document (~2K tokens)
    document = ". ".join([f"Sentence {i} with some content" for i in range(200)])

    start = time.time()
    doc_emb = strategy.document_embedder.embed_document(document, "perf_test")
    duration = time.time() - start

    print(f"\nDocument embedding: {doc_emb.token_count} tokens in {duration:.3f}s")
    assert duration < 5.0, f"Too slow: {duration:.2f}s (expected <5s for CPU)"


@pytest.mark.benchmark
def test_embedding_chunking_speed(benchmark_vector_store):
    """Benchmark embedding chunking speed."""
    config = {
        "model_name": EMBEDDING_MODEL,
        "chunking_method": "fixed_size",
        "device": "cpu"
    }

    strategy = LateChunkingRAGStrategy(benchmark_vector_store, config)

    document = ". ".join([f"Sentence {i}" for i in range(100)])

    # Embed document
    doc_emb = strategy.document_embedder.embed_document(document, "perf_test")

    # Time chunking
    start = time.time()
    chunks = strategy.embedding_chunker.chunk_embeddings(doc_emb)
    duration = time.time() - start

    print(f"\nEmbedding chunking: {len(chunks)} chunks in {duration:.3f}s")
    assert duration < 1.0, f"Too slow: {duration:.2f}s (expected <1s)"


@pytest.mark.benchmark
def test_semantic_boundary_speed(benchmark_vector_store):
    """Benchmark semantic boundary detection speed."""
    config = {
        "model_name": EMBEDDING_MODEL,
        "chunking_method": "semantic_boundary",
        "similarity_threshold": 0.7,
        "device": "cpu"
    }

    strategy = LateChunkingRAGStrategy(benchmark_vector_store, config)

    document = ". ".join([f"Sentence {i} with content" for i in range(100)])

    doc_emb = strategy.document_embedder.embed_document(document, "perf_test")

    start = time.time()
    chunks = strategy.embedding_chunker.chunk_embeddings(doc_emb)
    duration = time.time() - start

    print(f"\nSemantic boundary chunking: {len(chunks)} chunks in {duration:.3f}s")
    assert duration < 1.0, f"Too slow: {duration:.2f}s"


@pytest.mark.benchmark
def test_end_to_end_latency(benchmark_vector_store):
    """Benchmark end-to-end late chunking latency."""
    config = {
        "model_name": EMBEDDING_MODEL,
        "compute_coherence_scores": False,  # Disable for performance test
        "device": "cpu"
    }

    strategy = LateChunkingRAGStrategy(benchmark_vector_store, config)

    document = "Test document. " * 100

    start = time.time()
    strategy.index_document(document, "latency_test")
    duration = time.time() - start

    print(f"\nEnd-to-end late chunking: {duration:.3f}s")
    # This includes embedding + chunking + indexing
    assert duration < 10.0, f"Too slow: {duration:.2f}s"


@pytest.mark.benchmark
def test_coherence_analysis_overhead(benchmark_vector_store):
    """Benchmark coherence analysis overhead."""
    config_without = {
        "model_name": EMBEDDING_MODEL,
        "compute_coherence_scores": False,
        "device": "cpu"
    }

    config_with = {
        "model_name": EMBEDDING_MODEL,
        "compute_coherence_scores": True,
        "device": "cpu"
    }

    document = "Test sentence. " * 50

    # Without coherence
    strategy_without = LateChunkingRAGStrategy(MockVectorStore(), config_without)
    start = time.time()
    strategy_without.index_document(document, "test1")
    time_without = time.time() - start

    # With coherence
    strategy_with = LateChunkingRAGStrategy(MockVectorStore(), config_with)
    start = time.time()
    strategy_with.index_document(document, "test2")
    time_with = time.time() - start

    overhead = time_with - time_without
    print(f"\nCoherence analysis overhead: {overhead:.3f}s")
    
    # Overhead should be minimal
    assert overhead < 1.0, f"Coherence overhead too high: {overhead:.2f}s"


@pytest.mark.benchmark
def test_batch_processing_speed(benchmark_vector_store):
    """Benchmark batch processing of documents."""
    config = {
        "model_name": EMBEDDING_MODEL,
        "device": "cpu"
    }

    strategy = LateChunkingRAGStrategy(benchmark_vector_store, config)

    documents = [
        {"text": f"Document {i} with some content.", "document_id": f"doc_{i}"}
        for i in range(10)
    ]

    start = time.time()
    doc_embeddings = strategy.document_embedder.embed_documents_batch(documents)
    duration = time.time() - start

    print(f"\nBatch processing: {len(documents)} documents in {duration:.3f}s")
    print(f"Average: {duration/len(documents):.3f}s per document")

    assert len(doc_embeddings) == len(documents)


@pytest.mark.benchmark
def test_adaptive_chunking_speed(benchmark_vector_store):
    """Benchmark adaptive chunking speed."""
    config = {
        "model_name": EMBEDDING_MODEL,
        "chunking_method": "adaptive",
        "min_chunk_size": 20,
        "max_chunk_size": 100,
        "device": "cpu"
    }

    strategy = LateChunkingRAGStrategy(benchmark_vector_store, config)

    document = ". ".join([f"Sentence {i}" for i in range(100)])

    doc_emb = strategy.document_embedder.embed_document(document, "perf_test")

    start = time.time()
    chunks = strategy.embedding_chunker.chunk_embeddings(doc_emb)
    duration = time.time() - start

    print(f"\nAdaptive chunking: {len(chunks)} chunks in {duration:.3f}s")
    assert duration < 1.0, f"Too slow: {duration:.2f}s"


@pytest.mark.benchmark
def test_memory_efficiency(benchmark_vector_store):
    """Test memory usage for large documents."""
    import sys

    config = {
        "model_name": EMBEDDING_MODEL,
        "device": "cpu"
    }

    strategy = LateChunkingRAGStrategy(benchmark_vector_store, config)

    # Create a moderately large document
    document = ". ".join([f"Sentence {i} with content" for i in range(500)])

    doc_emb = strategy.document_embedder.embed_document(document, "memory_test")

    # Estimate memory usage
    embedding_size = sys.getsizeof(doc_emb.full_embedding)
    token_embeddings_size = sum(sys.getsizeof(t.embedding) for t in doc_emb.token_embeddings)
    total_size = embedding_size + token_embeddings_size

    print(f"\nMemory usage: {total_size / 1024 / 1024:.2f} MB for {doc_emb.token_count} tokens")
    
    # Should be reasonable (less than 100MB for this test)
    assert total_size < 100 * 1024 * 1024, "Memory usage too high"
