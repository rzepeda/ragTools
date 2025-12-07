"""
Performance benchmarks for Contextual Retrieval Strategy.

Tests performance requirements including throughput and processing time.
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, AsyncMock

from rag_factory.strategies.contextual.context_generator import ContextGenerator
from rag_factory.strategies.contextual.batch_processor import BatchProcessor
from rag_factory.strategies.contextual.cost_tracker import CostTracker
from rag_factory.strategies.contextual.config import ContextualRetrievalConfig


@pytest.fixture
def mock_llm_service():
    """Mock LLM service with realistic delay."""
    service = Mock()
    
    async def mock_generate(prompt, temperature, max_tokens):
        # Simulate realistic LLM latency
        await asyncio.sleep(0.01)
        response = Mock()
        response.text = "This is a contextual description of the chunk content."
        return response
    
    service.agenerate = mock_generate
    return service


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_batch_processing_throughput(mock_llm_service):
    """Benchmark batch processing throughput."""
    config = ContextualRetrievalConfig(
        batch_size=20,
        enable_parallel_batches=True,
        max_concurrent_batches=5
    )
    
    generator = ContextGenerator(mock_llm_service, config)
    tracker = CostTracker(config)
    processor = BatchProcessor(generator, tracker, config)
    
    # Generate large chunk set
    chunks = [
        {"chunk_id": f"chunk_{i}", "text": f"Text content {i} " * 50, "metadata": {}}
        for i in range(200)
    ]
    
    start = time.time()
    processed = await processor.process_chunks(chunks)
    duration = time.time() - start
    
    chunks_per_minute = (len(processed) / duration) * 60
    
    print(f"\nBatch processing: {len(processed)} chunks in {duration:.2f}s")
    print(f"Throughput: {chunks_per_minute:.0f} chunks/minute")
    
    # Should meet >100 chunks/minute target
    assert chunks_per_minute >= 100, f"Throughput {chunks_per_minute:.0f} chunks/min (expected >=100)"


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_parallel_vs_sequential_performance(mock_llm_service):
    """Compare parallel vs sequential batch processing."""
    chunks = [
        {"chunk_id": f"chunk_{i}", "text": f"Text {i} " * 30, "metadata": {}}
        for i in range(50)
    ]
    
    # Test parallel
    config_parallel = ContextualRetrievalConfig(
        batch_size=10,
        enable_parallel_batches=True,
        max_concurrent_batches=5
    )
    
    generator_p = ContextGenerator(mock_llm_service, config_parallel)
    tracker_p = CostTracker(config_parallel)
    processor_p = BatchProcessor(generator_p, tracker_p, config_parallel)
    
    start_parallel = time.time()
    await processor_p.process_chunks(chunks)
    parallel_duration = time.time() - start_parallel
    
    # Test sequential
    config_sequential = ContextualRetrievalConfig(
        batch_size=10,
        enable_parallel_batches=False
    )
    
    generator_s = ContextGenerator(mock_llm_service, config_sequential)
    tracker_s = CostTracker(config_sequential)
    processor_s = BatchProcessor(generator_s, tracker_s, config_sequential)
    
    start_sequential = time.time()
    await processor_s.process_chunks(chunks)
    sequential_duration = time.time() - start_sequential
    
    print(f"\nParallel: {parallel_duration:.2f}s")
    print(f"Sequential: {sequential_duration:.2f}s")
    print(f"Speedup: {sequential_duration / parallel_duration:.2f}x")
    
    # Parallel should be faster
    assert parallel_duration < sequential_duration


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_large_document_processing_time(mock_llm_service):
    """Benchmark processing time for large documents."""
    config = ContextualRetrievalConfig(
        batch_size=50,
        enable_parallel_batches=True,
        max_concurrent_batches=10
    )
    
    generator = ContextGenerator(mock_llm_service, config)
    tracker = CostTracker(config)
    processor = BatchProcessor(generator, tracker, config)
    
    # 1000 chunks (large document)
    chunks = [
        {"chunk_id": f"chunk_{i}", "text": f"Content {i} " * 40, "metadata": {}}
        for i in range(1000)
    ]
    
    start = time.time()
    processed = await processor.process_chunks(chunks)
    duration = time.time() - start
    
    print(f"\nLarge document: {len(processed)} chunks in {duration:.2f}s")
    print(f"Average per chunk: {(duration / len(processed)) * 1000:.2f}ms")
    
    # Should meet <5min for 1000 chunks target
    assert duration < 300, f"Processing took {duration:.2f}s (expected <300s)"


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_batch_size_impact(mock_llm_service):
    """Test impact of different batch sizes on performance."""
    chunks = [
        {"chunk_id": f"chunk_{i}", "text": f"Text {i} " * 30, "metadata": {}}
        for i in range(100)
    ]
    
    batch_sizes = [5, 10, 20, 50]
    results = {}
    
    for batch_size in batch_sizes:
        config = ContextualRetrievalConfig(
            batch_size=batch_size,
            enable_parallel_batches=True,
            max_concurrent_batches=5
        )
        
        generator = ContextGenerator(mock_llm_service, config)
        tracker = CostTracker(config)
        processor = BatchProcessor(generator, tracker, config)
        
        start = time.time()
        await processor.process_chunks(chunks)
        duration = time.time() - start
        
        results[batch_size] = duration
        print(f"Batch size {batch_size}: {duration:.2f}s")
    
    # Larger batches should generally be faster (up to a point)
    assert results[20] <= results[5] * 1.5  # Allow some variance


@pytest.mark.benchmark
def test_cost_tracker_performance():
    """Benchmark cost tracker performance."""
    config = ContextualRetrievalConfig()
    tracker = CostTracker(config)
    
    # Record many chunks
    start = time.time()
    for i in range(10000):
        tracker.record_chunk_cost(f"chunk_{i}", 100, 50, 0.001)
    duration = time.time() - start
    
    print(f"\nRecorded 10000 chunks in {duration:.4f}s")
    print(f"Average per chunk: {(duration / 10000) * 1000000:.2f}μs")
    
    # Should be very fast
    assert duration < 1.0  # Less than 1 second for 10k chunks


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_context_generation_latency(mock_llm_service):
    """Benchmark individual context generation latency."""
    config = ContextualRetrievalConfig()
    generator = ContextGenerator(mock_llm_service, config)
    
    chunk = {
        "chunk_id": "chunk_1",
        "text": "This is a test chunk with some content " * 10,
        "metadata": {}
    }
    
    latencies = []
    
    for _ in range(10):
        start = time.time()
        await generator.generate_context(chunk)
        latency = time.time() - start
        latencies.append(latency)
    
    avg_latency = sum(latencies) / len(latencies)
    
    print(f"\nAverage context generation latency: {avg_latency * 1000:.2f}ms")
    print(f"Min: {min(latencies) * 1000:.2f}ms, Max: {max(latencies) * 1000:.2f}ms")
    
    # Should be reasonably fast
    assert avg_latency < 1.0  # Less than 1 second per chunk
