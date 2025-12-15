"""Integration tests for embedding service.

These tests require actual API keys and models to be available.
They can be skipped if the required dependencies are not installed.
"""

import pytest
import os
from rag_factory.services.embedding import EmbeddingService, EmbeddingServiceConfig

# Get embedding model from environment or use ONNX-compatible default
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "Xenova/all-MiniLM-L6-v2")


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_API_KEY").startswith("sk-"),
    reason="Valid OPENAI_API_KEY environment variable not set (must start with sk-)"
)
def test_openai_full_workflow():
    """Test complete embedding workflow with real OpenAI API."""
    config = EmbeddingServiceConfig(
        provider="openai",
        model="text-embedding-3-small",
        provider_config={
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        enable_cache=True,
        enable_rate_limiting=True,
        rate_limit_config={
            "requests_per_second": 2
        }
    )

    service = EmbeddingService(config)

    # Test single embedding
    result1 = service.embed(["Hello, world!"])
    assert len(result1.embeddings) == 1
    assert len(result1.embeddings[0]) == 1536
    assert result1.cached[0] is False

    # Test cache hit
    result2 = service.embed(["Hello, world!"])
    assert result2.cached[0] is True

    # Test batch embedding
    texts = [f"Test text {i}" for i in range(5)]
    result3 = service.embed(texts)
    assert len(result3.embeddings) == 5

    # Check stats
    stats = service.get_stats()
    assert stats["total_requests"] == 3
    assert stats["cache_hits"] >= 1


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("COHERE_API_KEY"),
    reason="COHERE_API_KEY environment variable not set"
)
def test_cohere_full_workflow():
    """Test complete embedding workflow with real Cohere API."""
    config = EmbeddingServiceConfig(
        provider="cohere",
        model="embed-english-v3.0",
        provider_config={
            "api_key": os.getenv("COHERE_API_KEY")
        },
        enable_cache=True
    )

    service = EmbeddingService(config)

    # Test single embedding
    result1 = service.embed(["Hello, world!"])
    assert len(result1.embeddings) == 1
    assert len(result1.embeddings[0]) == 1024
    assert result1.provider == "cohere"


@pytest.mark.integration
def test_local_embedding_provider():
    """Test ONNX local embedding provider."""
    try:
        config = EmbeddingServiceConfig(
            provider="onnx-local",
            model=EMBEDDING_MODEL,
            enable_cache=True
        )

        service = EmbeddingService(config)

        texts = ["This is a test", "Another test"]
        result = service.embed(texts)

        assert len(result.embeddings) == 2
        assert result.provider == "onnx-local"
        assert result.cost == 0.0  # Local models have no cost
        assert len(result.embeddings[0]) == result.dimensions  # Use actual model dimensions

    except ImportError:
        pytest.skip("ONNX dependencies not installed (optimum[onnxruntime], transformers)")


@pytest.mark.integration
def test_large_batch_processing():
    """Test processing large batch of texts with ONNX local model."""
    try:
        config = EmbeddingServiceConfig(
            provider="onnx-local",
            model=EMBEDDING_MODEL
        )

        service = EmbeddingService(config)

        # Generate 100 texts
        texts = [f"Document {i} with some content" for i in range(100)]

        import time
        start = time.time()
        result = service.embed(texts, use_cache=False)
        duration = time.time() - start

        assert len(result.embeddings) == 100

        # Should process at reasonable speed
        throughput = 100 / duration
        print(f"ONNX Throughput: {throughput:.0f} texts/sec")

    except ImportError:
        pytest.skip("ONNX dependencies not installed (optimum[onnxruntime], transformers)")


@pytest.mark.integration
def test_concurrent_embedding_requests():
    """Test concurrent requests to ONNX embedding service."""
    try:
        import concurrent.futures

        config = EmbeddingServiceConfig(
            provider="onnx-local",
            model=EMBEDDING_MODEL,
            enable_cache=True
        )

        service = EmbeddingService(config)

        def embed_text(text_id):
            texts = [f"Concurrent text {text_id}"]
            result = service.embed(texts)
            return result.embeddings[0]

        # Run 20 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(embed_text, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All requests should succeed
        assert len(results) == 20
        assert all(len(emb) > 0 for emb in results)

    except ImportError:
        pytest.skip("ONNX dependencies not installed (optimum[onnxruntime], transformers)")


@pytest.mark.integration
def test_cache_persistence_across_batches():
    """Test that cache works correctly across multiple batches with ONNX."""
    try:
        config = EmbeddingServiceConfig(
            provider="onnx-local",
            model=EMBEDDING_MODEL,
            enable_cache=True
        )

        service = EmbeddingService(config)

        # First batch
        texts1 = ["Text A", "Text B", "Text C"]
        result1 = service.embed(texts1)
        assert all(not cached for cached in result1.cached)

        # Second batch with some overlap
        texts2 = ["Text B", "Text C", "Text D"]
        result2 = service.embed(texts2)

        # First two should be cached
        assert result2.cached[0] is True  # Text B
        assert result2.cached[1] is True  # Text C
        assert result2.cached[2] is False  # Text D (new)

        # Verify embeddings match
        assert result1.embeddings[1] == result2.embeddings[0]  # Text B
        assert result1.embeddings[2] == result2.embeddings[1]  # Text C

    except ImportError:
        pytest.skip("ONNX dependencies not installed (optimum[onnxruntime], transformers)")


@pytest.mark.integration
def test_error_handling():
    """Test error handling with invalid inputs using ONNX provider."""
    try:
        config = EmbeddingServiceConfig(
            provider="onnx-local",
            model=EMBEDDING_MODEL
        )

        service = EmbeddingService(config)

        # Empty list should raise error
        with pytest.raises(ValueError, match="cannot be empty"):
            service.embed([])

    except ImportError:
        pytest.skip("ONNX dependencies not installed (optimum[onnxruntime], transformers)")


@pytest.mark.integration
def test_onnx_provider_compatibility():
    """Test that ONNX provider produces consistent results."""
    try:
        config = EmbeddingServiceConfig(
            provider="onnx-local",
            model=EMBEDDING_MODEL,
            enable_cache=False
        )

        service = EmbeddingService(config)

        # Test text
        text = "Machine learning is transforming technology"

        # Generate embedding twice
        result1 = service.embed([text])
        result2 = service.embed([text])

        # Should produce identical embeddings
        assert len(result1.embeddings) == 1
        assert len(result2.embeddings) == 1
        assert result1.embeddings[0] == result2.embeddings[0]

        # Verify metadata
        assert result1.provider == "onnx-local"
        assert result1.dimensions in [384, 768]  # Support both MiniLM (384) and mpnet (768)
        assert result1.cost == 0.0

    except ImportError:
        pytest.skip("ONNX dependencies not installed (optimum[onnxruntime], transformers)")
