"""
Example: Using ONNX-based Local Embedding Provider

This example demonstrates how to use the lightweight ONNX local provider
for embeddings. ONNX requires only ~200MB of dependencies compared to
~2.5GB for PyTorch-based sentence-transformers.

Installation:
    pip install optimum[onnxruntime]>=1.16.0 transformers>=4.36.0

Benefits:
    - 90% smaller dependencies (~200MB vs ~2.5GB)
    - Faster inference (ONNX optimized)
    - Same model compatibility
    - Zero API costs
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_factory.services.embedding import EmbeddingService, EmbeddingServiceConfig


def example_1_basic_usage():
    """Example 1: Basic ONNX local embeddings."""
    print("\n" + "=" * 60)
    print("Example 1: Basic ONNX Local Embeddings")
    print("=" * 60)

    # Configure ONNX local provider
    config = EmbeddingServiceConfig(
        provider="onnx-local",
        model="Xenova/all-MiniLM-L6-v2",
        provider_config={
            "model": "Xenova/all-MiniLM-L6-v2",
            "max_batch_size": 32,
        },
        enable_cache=True,
        enable_rate_limiting=False,  # No rate limiting for local models
    )

    service = EmbeddingService(config)

    # Generate embeddings
    texts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating",
    ]

    print(f"\nGenerating embeddings for {len(texts)} texts...")
    result = service.embed(texts)

    print(f"\n✓ Generated {len(result.embeddings)} embeddings")
    print(f"  Model: {result.model}")
    print(f"  Provider: {result.provider}")
    print(f"  Dimensions: {result.dimensions}")
    print(f"  Cost: ${result.cost:.4f} (always $0 for local)")
    print(f"  Token count: {result.token_count}")

    # Show first embedding (truncated)
    print(f"\nFirst embedding (first 10 values):")
    print(f"  {result.embeddings[0][:10]}")


def example_2_model_comparison():
    """Example 2: Compare different ONNX models."""
    print("\n" + "=" * 60)
    print("Example 2: Compare Different ONNX Models")
    print("=" * 60)

    models = [
        {
            "name": "MiniLM-L6 (Fast & Lightweight)",
            "model": "Xenova/all-MiniLM-L6-v2",
            "dims": 384,
            "size": "90MB",
        },
        {
            "name": "BGE Small (State-of-the-art)",
            "model": "BAAI/bge-small-en-v1.5",
            "dims": 384,
            "size": "133MB",
        },
        {
            "name": "MPNet Base (Higher Quality)",
            "model": "sentence-transformers/all-mpnet-base-v2",
            "dims": 768,
            "size": "420MB",
        },
    ]

    text = "What is the meaning of life?"

    print("\nComparing models on text:")
    print(f'  "{text}"\n')

    for model_info in models:
        try:
            print(f"Testing {model_info['name']}...")
            print(f"  Model: {model_info['model']}")
            print(f"  Dimensions: {model_info['dims']}")
            print(f"  Size: {model_info['size']}")

            config = EmbeddingServiceConfig(
                provider="onnx-local",
                model=model_info["model"],
                provider_config={"model": model_info["model"]},
                enable_cache=False,
            )

            service = EmbeddingService(config)
            result = service.embed([text])

            print(f"  ✓ Generated {result.dimensions}D embedding")
            print(f"  Embedding norm: {sum(x**2 for x in result.embeddings[0]) ** 0.5:.4f}")
            print()

        except ImportError:
            print(f"  ⚠ ONNX dependencies not installed")
            print(f"  Run: pip install optimum[onnxruntime] transformers\n")
            break
        except Exception as e:
            print(f"  ✗ Error: {e}\n")


def example_3_batch_processing():
    """Example 3: Efficient batch processing with ONNX."""
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing with ONNX")
    print("=" * 60)

    config = EmbeddingServiceConfig(
        provider="onnx-local",
        model="Xenova/all-MiniLM-L6-v2",
        provider_config={"max_batch_size": 32},
        enable_cache=True,
    )

    service = EmbeddingService(config)

    # Generate a larger batch
    texts = [
        f"This is document number {i} about machine learning."
        for i in range(50)
    ]

    print(f"\nProcessing {len(texts)} documents in batches...")
    result = service.embed(texts)

    print(f"\n✓ Processed {len(result.embeddings)} embeddings")
    print(f"  Total cost: ${result.cost:.4f}")
    print(f"  Cached: {sum(result.cached)} / {len(result.cached)}")

    # Try again (should use cache)
    print("\nProcessing same documents again (should use cache)...")
    result2 = service.embed(texts)

    print(f"  Cached: {sum(result2.cached)} / {len(result2.cached)}")

    # Get stats
    stats = service.get_stats()
    print(f"\nService statistics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Total cost: ${stats['total_cost']:.4f}")


def example_4_similarity_search():
    """Example 4: Semantic similarity search with ONNX embeddings."""
    print("\n" + "=" * 60)
    print("Example 4: Semantic Similarity Search")
    print("=" * 60)

    config = EmbeddingServiceConfig(
        provider="onnx-local",
        model="Xenova/all-MiniLM-L6-v2",
        enable_cache=False,
    )

    service = EmbeddingService(config)

    # Documents to search
    documents = [
        "Python is a programming language",
        "Dogs are loyal animals",
        "Machine learning is a subset of AI",
        "Cats are independent pets",
        "JavaScript is used for web development",
        "Deep learning uses neural networks",
    ]

    # Query
    query = "What is artificial intelligence?"

    print(f'\nQuery: "{query}"')
    print(f"\nSearching {len(documents)} documents...")

    # Embed all documents and query
    all_texts = documents + [query]
    result = service.embed(all_texts)

    # Split embeddings
    doc_embeddings = result.embeddings[:-1]
    query_embedding = result.embeddings[-1]

    # Calculate cosine similarity
    def cosine_similarity(a, b):
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x**2 for x in a) ** 0.5
        norm_b = sum(x**2 for x in b) ** 0.5
        return dot_product / (norm_a * norm_b)

    # Find most similar documents
    similarities = [
        (doc, cosine_similarity(query_embedding, emb))
        for doc, emb in zip(documents, doc_embeddings)
    ]

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 3 most similar documents:")
    for i, (doc, score) in enumerate(similarities[:3], 1):
        print(f"  {i}. [{score:.4f}] {doc}")


def example_5_onnx_vs_api():
    """Example 5: Compare ONNX local vs API providers."""
    print("\n" + "=" * 60)
    print("Example 5: ONNX Local vs API Providers")
    print("=" * 60)

    providers = [
        {
            "name": "ONNX Local (MiniLM)",
            "config": EmbeddingServiceConfig(
                provider="onnx-local",
                model="Xenova/all-MiniLM-L6-v2",
                enable_cache=False,
            ),
        },
    ]

    # Try to add API providers if keys available
    import os

    if os.getenv("OPENAI_API_KEY"):
        providers.append(
            {
                "name": "OpenAI (text-embedding-3-small)",
                "config": EmbeddingServiceConfig(
                    provider="openai",
                    model="text-embedding-3-small",
                    enable_cache=False,
                ),
            }
        )

    if os.getenv("COHERE_API_KEY"):
        providers.append(
            {
                "name": "Cohere (embed-english-v3.0)",
                "config": EmbeddingServiceConfig(
                    provider="cohere",
                    model="embed-english-v3.0",
                    enable_cache=False,
                ),
            }
        )

    text = "Machine learning is transforming technology."

    print(f'\nGenerating embeddings for: "{text}"\n')

    for provider_info in providers:
        try:
            print(f"{provider_info['name']}:")
            service = EmbeddingService(provider_info["config"])
            result = service.embed([text])

            print(f"  Dimensions: {result.dimensions}")
            print(f"  Cost: ${result.cost:.6f}")
            print(f"  Provider: {result.provider}")
            print()

        except ImportError as e:
            print(f"  ⚠ Dependencies not installed: {e}\n")
        except Exception as e:
            print(f"  ✗ Error: {e}\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ONNX Local Embedding Provider Examples")
    print("=" * 60)
    print("\nThese examples demonstrate the lightweight ONNX provider")
    print("which requires only ~200MB vs ~2.5GB for PyTorch.\n")

    try:
        example_1_basic_usage()
        example_2_model_comparison()
        example_3_batch_processing()
        example_4_similarity_search()
        example_5_onnx_vs_api()

    except ImportError as e:
        print("\n" + "=" * 60)
        print("⚠ ONNX Dependencies Not Installed")
        print("=" * 60)
        print("\nTo use ONNX local embeddings, install:")
        print("  pip install optimum[onnxruntime]>=1.16.0 transformers>=4.36.0")
        print("\nTotal size: ~200MB (vs ~2.5GB for sentence-transformers)")
        print(f"\nError: {e}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
