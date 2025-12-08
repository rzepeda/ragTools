#!/usr/bin/env python3
"""Performance benchmarks for ONNX embeddings.

This script benchmarks the ONNX embedding provider to ensure it meets
performance targets:
- Embedding speed: <100ms per document (CPU)
- Memory usage: <500MB for model + inference
- Batch processing: support up to 32 documents

Usage:
    python tests/performance/benchmark_onnx_embeddings.py
    python tests/performance/benchmark_onnx_embeddings.py --model BAAI/bge-small-en-v1.5
    python tests/performance/benchmark_onnx_embeddings.py --iterations 1000
"""

import argparse
import time
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory profiling disabled.")

from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider


def get_memory_usage():
    """Get current memory usage in MB."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    return 0


def benchmark_embedding_speed(provider, iterations=100):
    """Benchmark embedding speed for different text lengths.
    
    Args:
        provider: ONNX embedding provider
        iterations: Number of iterations for each test
        
    Returns:
        Dictionary with benchmark results
    """
    print("\n" + "=" * 70)
    print("EMBEDDING SPEED BENCHMARK")
    print("=" * 70)
    
    # Test documents of varying lengths
    test_cases = {
        "short": "Short document.",
        "medium": "This is a medium length document with several sentences. " * 10,
        "long": "This is a longer document with more content to process. " * 50,
    }
    
    results = {}
    
    for name, text in test_cases.items():
        print(f"\nBenchmarking {name} text ({len(text)} chars)...")
        
        # Warm up
        for _ in range(10):
            provider.get_embeddings([text])
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            provider.get_embeddings([text])
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms
        
        times = np.array(times)
        
        results[name] = {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
        }
        
        # Print results
        print(f"  Mean:  {results[name]['mean_ms']:.2f} ms ± {results[name]['std_ms']:.2f} ms")
        print(f"  P50:   {results[name]['p50_ms']:.2f} ms")
        print(f"  P95:   {results[name]['p95_ms']:.2f} ms")
        print(f"  P99:   {results[name]['p99_ms']:.2f} ms")
        print(f"  Range: {results[name]['min_ms']:.2f} - {results[name]['max_ms']:.2f} ms")
        
        # Check if meets target
        if results[name]['p95_ms'] < 100:
            print(f"  ✓ Meets <100ms target")
        else:
            print(f"  ⚠ Exceeds 100ms target")
    
    return results


def benchmark_batch_processing(provider, max_batch_size=32, iterations=50):
    """Benchmark batch processing efficiency.
    
    Args:
        provider: ONNX embedding provider
        max_batch_size: Maximum batch size to test
        iterations: Number of iterations for each batch size
        
    Returns:
        Dictionary with batch processing results
    """
    print("\n" + "=" * 70)
    print("BATCH PROCESSING BENCHMARK")
    print("=" * 70)
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    if max_batch_size < 32:
        batch_sizes = [b for b in batch_sizes if b <= max_batch_size]
    
    text = "This is a test document for batch processing benchmarks."
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size {batch_size}...")
        
        texts = [text] * batch_size
        
        # Warm up
        for _ in range(5):
            provider.get_embeddings(texts)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            provider.get_embeddings(texts)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms
        
        times = np.array(times)
        
        results[batch_size] = {
            "total_mean_ms": float(np.mean(times)),
            "total_std_ms": float(np.std(times)),
            "per_doc_ms": float(np.mean(times) / batch_size),
            "throughput": float(batch_size / (np.mean(times) / 1000)),  # docs/sec
        }
        
        # Print results
        print(f"  Total time:  {results[batch_size]['total_mean_ms']:.2f} ms ± {results[batch_size]['total_std_ms']:.2f} ms")
        print(f"  Per document: {results[batch_size]['per_doc_ms']:.2f} ms")
        print(f"  Throughput:   {results[batch_size]['throughput']:.1f} docs/sec")
    
    # Calculate efficiency
    print("\nBatch Processing Efficiency:")
    baseline = results[1]["per_doc_ms"]
    for batch_size in batch_sizes:
        speedup = baseline / results[batch_size]["per_doc_ms"]
        efficiency = (speedup / batch_size) * 100
        print(f"  Batch {batch_size:2d}: {speedup:.2f}x speedup, {efficiency:.1f}% efficiency")
    
    return results


def benchmark_memory_usage(provider):
    """Benchmark memory usage.
    
    Args:
        provider: ONNX embedding provider
        
    Returns:
        Dictionary with memory usage results
    """
    if not PSUTIL_AVAILABLE:
        print("\n⚠ Memory profiling skipped (psutil not available)")
        return {}
    
    print("\n" + "=" * 70)
    print("MEMORY USAGE BENCHMARK")
    print("=" * 70)
    
    # Baseline memory
    baseline_memory = get_memory_usage()
    print(f"\nBaseline memory: {baseline_memory:.2f} MB")
    
    # Memory after loading model (already loaded)
    model_memory = get_memory_usage()
    model_overhead = model_memory - baseline_memory
    print(f"Model memory:    {model_memory:.2f} MB (+{model_overhead:.2f} MB)")
    
    # Memory during inference
    text = "This is a test document. " * 50
    texts = [text] * 32  # Max batch size
    
    provider.get_embeddings(texts)
    
    inference_memory = get_memory_usage()
    inference_overhead = inference_memory - model_memory
    total_overhead = inference_memory - baseline_memory
    
    print(f"Inference memory: {inference_memory:.2f} MB (+{inference_overhead:.2f} MB)")
    print(f"Total overhead:   {total_overhead:.2f} MB")
    
    # Check if meets target
    if total_overhead < 500:
        print(f"\n✓ Meets <500MB target")
    else:
        print(f"\n⚠ Exceeds 500MB target")
    
    return {
        "baseline_mb": baseline_memory,
        "model_mb": model_memory,
        "inference_mb": inference_memory,
        "model_overhead_mb": model_overhead,
        "inference_overhead_mb": inference_overhead,
        "total_overhead_mb": total_overhead,
    }


def benchmark_model_load_time(model_name):
    """Benchmark model loading time.
    
    Args:
        model_name: Model to load
        
    Returns:
        Load time in seconds
    """
    print("\n" + "=" * 70)
    print("MODEL LOAD TIME BENCHMARK")
    print("=" * 70)
    
    print(f"\nLoading model: {model_name}")
    
    start = time.perf_counter()
    provider = ONNXLocalProvider({"model": model_name})
    load_time = time.perf_counter() - start
    
    print(f"Load time: {load_time:.2f} seconds")
    
    # Check if meets target
    if load_time < 5:
        print(f"✓ Meets <5s target")
    else:
        print(f"⚠ Exceeds 5s target")
    
    return load_time, provider


def print_summary(speed_results, batch_results, memory_results, load_time):
    """Print benchmark summary."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    print("\nSpeed (P95):")
    for name, results in speed_results.items():
        status = "✓" if results['p95_ms'] < 100 else "⚠"
        print(f"  {status} {name:8s}: {results['p95_ms']:6.2f} ms")
    
    print("\nBatch Processing:")
    for batch_size, results in batch_results.items():
        print(f"  Batch {batch_size:2d}: {results['throughput']:6.1f} docs/sec")
    
    if memory_results:
        print("\nMemory:")
        status = "✓" if memory_results['total_overhead_mb'] < 500 else "⚠"
        print(f"  {status} Total overhead: {memory_results['total_overhead_mb']:.2f} MB")
    
    print("\nModel Load:")
    status = "✓" if load_time < 5 else "⚠"
    print(f"  {status} Load time: {load_time:.2f} seconds")
    
    print("\n" + "=" * 70)


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark ONNX embeddings")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model to benchmark"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations for speed benchmarks"
    )
    parser.add_argument(
        "--batch-iterations",
        type=int,
        default=50,
        help="Number of iterations for batch benchmarks"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ONNX EMBEDDING PERFORMANCE BENCHMARKS")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Iterations: {args.iterations}")
    print("=" * 70)
    
    try:
        # Benchmark model load time
        load_time, provider = benchmark_model_load_time(args.model)
        
        # Benchmark embedding speed
        speed_results = benchmark_embedding_speed(provider, args.iterations)
        
        # Benchmark batch processing
        batch_results = benchmark_batch_processing(
            provider,
            max_batch_size=provider.get_max_batch_size(),
            iterations=args.batch_iterations
        )
        
        # Benchmark memory usage
        memory_results = benchmark_memory_usage(provider)
        
        # Print summary
        print_summary(speed_results, batch_results, memory_results, load_time)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
