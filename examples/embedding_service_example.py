"""Example usage of the Embedding Service.

This example demonstrates how to use the embedding service with different
providers and configurations.
"""

import os
from rag_factory.services.embedding import EmbeddingService, EmbeddingServiceConfig


def example_openai():
    """Example using OpenAI embeddings."""
    print("\n" + "="*60)
    print("OpenAI Embedding Example")
    print("="*60)

    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Skipping OpenAI example.")
        return

    # Configure service
    config = EmbeddingServiceConfig(
        provider="openai",
        model="text-embedding-3-small",
        enable_cache=True,
        enable_rate_limiting=True,
        rate_limit_config={
            "requests_per_second": 2
        }
    )

    # Initialize service
    service = EmbeddingService(config)

    # Generate embeddings
    texts = [
        "Machine learning is transforming technology",
        "Natural language processing enables human-computer interaction",
        "Machine learning is transforming technology"  # Duplicate to test cache
    ]

    print(f"\nEmbedding {len(texts)} texts...")
    result = service.embed(texts)

    print(f"\n✅ Success!")
    print(f"   Generated: {len(result.embeddings)} embeddings")
    print(f"   Dimensions: {result.dimensions}")
    print(f"   Model: {result.model}")
    print(f"   Token count: {result.token_count}")
    print(f"   Cost: ${result.cost:.6f}")
    print(f"   Cached: {result.cached}")

    # Check statistics
    stats = service.get_stats()
    print(f"\n📊 Statistics:")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache misses: {stats['cache_misses']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.2%}")


def example_cohere():
    """Example using Cohere embeddings."""
    print("\n" + "="*60)
    print("Cohere Embedding Example")
    print("="*60)

    # Check if API key is available
    if not os.getenv("COHERE_API_KEY"):
        print("⚠️  COHERE_API_KEY not set. Skipping Cohere example.")
        return

    # Configure service
    config = EmbeddingServiceConfig(
        provider="cohere",
        model="embed-english-v3.0",
        enable_cache=True
    )

    # Initialize service
    service = EmbeddingService(config)

    # Generate embeddings
    texts = [
        "Artificial intelligence is advancing rapidly",
        "Deep learning powers modern AI systems"
    ]

    print(f"\nEmbedding {len(texts)} texts...")
    result = service.embed(texts)

    print(f"\n✅ Success!")
    print(f"   Generated: {len(result.embeddings)} embeddings")
    print(f"   Dimensions: {result.dimensions}")
    print(f"   Model: {result.model}")
    print(f"   Cost: ${result.cost:.6f}")


def example_local():
    """Example using local sentence-transformers."""
    print("\n" + "="*60)
    print("Local Model Embedding Example")
    print("="*60)

    try:
        # Configure service
        config = EmbeddingServiceConfig(
            provider="local",
            model="all-MiniLM-L6-v2",
            enable_cache=True
        )

        # Initialize service (will download model if needed)
        print("\n📥 Loading local model (may download first time)...")
        service = EmbeddingService(config)

        # Generate embeddings
        texts = [
            "Local embeddings require no API",
            "Models run on your own hardware",
            "No costs, full privacy",
            "Local embeddings require no API"  # Duplicate to test cache
        ]

        print(f"\nEmbedding {len(texts)} texts...")
        result = service.embed(texts)

        print(f"\n✅ Success!")
        print(f"   Generated: {len(result.embeddings)} embeddings")
        print(f"   Dimensions: {result.dimensions}")
        print(f"   Model: {result.model}")
        print(f"   Cost: ${result.cost:.6f} (free!)")
        print(f"   Cached: {result.cached}")

        # Check statistics
        stats = service.get_stats()
        print(f"\n📊 Statistics:")
        print(f"   Cache hit rate: {stats['cache_hit_rate']:.2%}")

    except ImportError as e:
        print(f"⚠️  Error: {e}")
        print("   Install with: pip install sentence-transformers torch")


def example_batch_processing():
    """Example of batch processing with local model."""
    print("\n" + "="*60)
    print("Batch Processing Example")
    print("="*60)

    try:
        # Configure service with smaller batch size for demo
        config = EmbeddingServiceConfig(
            provider="local",
            model="all-MiniLM-L6-v2",
            provider_config={
                "max_batch_size": 5
            },
            enable_cache=False  # Disable cache to see actual processing
        )

        print("\n📥 Loading local model...")
        service = EmbeddingService(config)

        # Generate many texts
        texts = [f"Document {i} with unique content" for i in range(15)]

        print(f"\nEmbedding {len(texts)} texts in batches...")
        print("(Batch size: 5, will create 3 batches)")

        import time
        start = time.time()
        result = service.embed(texts)
        duration = time.time() - start

        print(f"\n✅ Success!")
        print(f"   Generated: {len(result.embeddings)} embeddings")
        print(f"   Time taken: {duration:.2f}s")
        print(f"   Throughput: {len(texts)/duration:.1f} texts/sec")

    except ImportError as e:
        print(f"⚠️  Error: {e}")


def example_cache_effectiveness():
    """Example demonstrating cache effectiveness."""
    print("\n" + "="*60)
    print("Cache Effectiveness Example")
    print("="*60)

    try:
        config = EmbeddingServiceConfig(
            provider="local",
            model="all-MiniLM-L6-v2",
            enable_cache=True
        )

        print("\n📥 Loading local model...")
        service = EmbeddingService(config)

        texts = ["Hello world", "How are you?", "Hello world"]

        print("\n1️⃣  First embedding request (cache cold)...")
        import time
        start = time.time()
        result1 = service.embed(texts)
        time1 = time.time() - start

        print(f"   Time: {time1:.3f}s")
        print(f"   Cached: {result1.cached}")

        print("\n2️⃣  Second embedding request (cache warm)...")
        start = time.time()
        result2 = service.embed(texts)
        time2 = time.time() - start

        print(f"   Time: {time2:.3f}s")
        print(f"   Cached: {result2.cached}")

        print(f"\n⚡ Cache speedup: {time1/time2:.1f}x faster!")

        stats = service.get_stats()
        print(f"\n📊 Final statistics:")
        print(f"   Total requests: {stats['total_requests']}")
        print(f"   Cache hit rate: {stats['cache_hit_rate']:.2%}")

    except ImportError as e:
        print(f"⚠️  Error: {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RAG Factory - Embedding Service Examples")
    print("="*60)

    # Run examples
    example_local()
    example_batch_processing()
    example_cache_effectiveness()
    example_openai()
    example_cohere()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60 + "\n")
