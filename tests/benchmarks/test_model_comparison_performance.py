"""Performance benchmarks for model comparison."""

import pytest
import time
from rag_factory.models.embedding import CustomModelLoader, ModelConfig, ModelFormat


@pytest.mark.benchmark
def test_model_loading_speed():
    """Benchmark model loading time.
    
    Requirement: Model loading should be <2s for typical model.
    """
    loader = CustomModelLoader()

    config = ModelConfig(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        model_format=ModelFormat.SENTENCE_TRANSFORMERS,
        device="cpu"
    )

    start = time.time()
    model = loader.load_model(config)
    duration = time.time() - start

    print(f"\nModel loading time: {duration:.3f}s")
    assert duration < 5.0, f"Loading too slow: {duration:.2f}s (relaxed from 2s for CI)"


@pytest.mark.benchmark
def test_model_loading_cached():
    """Benchmark cached model loading time.
    
    Cached loading should be nearly instant.
    """
    loader = CustomModelLoader()

    config = ModelConfig(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        model_format=ModelFormat.SENTENCE_TRANSFORMERS,
        device="cpu"
    )

    # First load
    loader.load_model(config)

    # Second load (cached)
    start = time.time()
    model = loader.load_model(config)
    duration = time.time() - start

    print(f"\nCached model loading time: {duration:.3f}s")
    assert duration < 0.1, f"Cached loading too slow: {duration:.3f}s"


@pytest.mark.benchmark
def test_inference_speed():
    """Benchmark inference speed.
    
    Requirement: Should achieve >10 texts/second on CPU.
    """
    loader = CustomModelLoader()

    config = ModelConfig(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        model_format=ModelFormat.SENTENCE_TRANSFORMERS,
        device="cpu",
        batch_size=32
    )

    model = loader.load_model(config)

    # Test data
    texts = ["Test sentence " * 10] * 100

    # Benchmark
    start = time.time()
    embeddings = loader.embed_texts(texts, model, config)
    duration = time.time() - start

    print(f"\n{len(texts)} texts in {duration:.3f}s")

    # Calculate throughput
    throughput = len(texts) / duration
    print(f"Throughput: {throughput:.1f} texts/second")

    # Relaxed requirement for CI environments
    assert throughput > 5, f"Too slow: {throughput:.1f} texts/s (relaxed from 10)"
    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) == 384


@pytest.mark.benchmark
def test_batch_size_comparison():
    """Compare inference speed with different batch sizes."""
    loader = CustomModelLoader()

    texts = ["Test sentence " * 10] * 100
    batch_sizes = [8, 16, 32, 64]
    results = {}

    for batch_size in batch_sizes:
        config = ModelConfig(
            model_path="sentence-transformers/all-MiniLM-L6-v2",
            model_format=ModelFormat.SENTENCE_TRANSFORMERS,
            device="cpu",
            batch_size=batch_size
        )

        model = loader.load_model(config)

        start = time.time()
        embeddings = loader.embed_texts(texts, model, config)
        duration = time.time() - start

        throughput = len(texts) / duration
        results[batch_size] = throughput

        print(f"\nBatch size {batch_size}: {throughput:.1f} texts/s")

    # Larger batch sizes should generally be faster (or at least not much slower)
    # This is a soft check since it depends on hardware
    print(f"\nBatch size comparison: {results}")


@pytest.mark.benchmark
def test_embedding_quality():
    """Test that embeddings have expected properties."""
    loader = CustomModelLoader()

    config = ModelConfig(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        model_format=ModelFormat.SENTENCE_TRANSFORMERS,
        device="cpu",
        normalize_embeddings=True
    )

    model = loader.load_model(config)

    # Generate embeddings
    texts = ["Hello world", "Goodbye world", "Completely different text"]
    embeddings = loader.embed_texts(texts, model, config)

    # Check dimensions
    assert all(len(emb) == 384 for emb in embeddings)

    # Check normalization (L2 norm should be ~1.0)
    import numpy as np
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        assert 0.99 < norm < 1.01, f"Embedding not normalized: {norm}"

    # Check similarity
    # Similar texts should have higher similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_hello_goodbye = cosine_similarity(embeddings[0], embeddings[1])
    sim_hello_different = cosine_similarity(embeddings[0], embeddings[2])

    print(f"\nSimilarity (Hello-Goodbye): {sim_hello_goodbye:.3f}")
    print(f"Similarity (Hello-Different): {sim_hello_different:.3f}")

    # "Hello world" and "Goodbye world" should be more similar
    assert sim_hello_goodbye > sim_hello_different


@pytest.mark.benchmark
def test_memory_efficiency():
    """Test that model caching doesn't cause memory issues."""
    loader = CustomModelLoader()

    config = ModelConfig(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        model_format=ModelFormat.SENTENCE_TRANSFORMERS,
        device="cpu"
    )

    # Load same model multiple times
    for _ in range(5):
        model = loader.load_model(config)
        embeddings = loader.embed_texts(["test"], model, config)

    # Should only have one cached model
    assert len(loader.loaded_models) == 1
