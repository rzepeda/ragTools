"""
Unit tests for BatchProcessor.

Tests batch processing functionality including parallel/sequential modes,
error handling, and cost tracking integration.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from rag_factory.strategies.contextual.batch_processor import BatchProcessor
from rag_factory.strategies.contextual.config import ContextualRetrievalConfig


@pytest.fixture
def mock_context_generator():
    """Mock context generator for testing."""
    generator = Mock()
    
    async def mock_generate(chunk, doc_context=None):
        return f"Context for {chunk['chunk_id']}"
    
    generator.generate_context = mock_generate
    generator._count_tokens = lambda text: len(text) // 4
    return generator


@pytest.fixture
def mock_cost_tracker():
    """Mock cost tracker for testing."""
    tracker = Mock()
    tracker.calculate_cost.return_value = 0.001
    tracker.record_chunk_cost = Mock()
    return tracker


@pytest.fixture
def config():
    """Default configuration for testing."""
    return ContextualRetrievalConfig(
        batch_size=5,
        enable_parallel_batches=True
    )


@pytest.fixture
def batch_processor(mock_context_generator, mock_cost_tracker, config):
    """Batch processor instance for testing."""
    return BatchProcessor(mock_context_generator, mock_cost_tracker, config)


@pytest.mark.asyncio
async def test_process_chunks_batching(batch_processor):
    """Test that chunks are processed in batches."""
    chunks = [
        {"chunk_id": f"chunk_{i}", "text": f"Text {i}"} for i in range(12)
    ]
    
    processed = await batch_processor.process_chunks(chunks)
    
    # All chunks should be processed
    assert len(processed) == 12
    assert all("context_description" in c for c in processed)


@pytest.mark.asyncio
async def test_parallel_batch_processing(batch_processor):
    """Test parallel batch processing."""
    chunks = [
        {"chunk_id": f"chunk_{i}", "text": f"Text {i}"} for i in range(20)
    ]
    
    processed = await batch_processor.process_chunks(chunks)
    
    # Should process all chunks
    assert len(processed) == 20


@pytest.mark.asyncio
async def test_cost_tracking_during_batch(batch_processor, mock_cost_tracker):
    """Test that costs are tracked during batch processing."""
    chunks = [
        {"chunk_id": f"chunk_{i}", "text": f"Text {i}"} for i in range(5)
    ]
    
    await batch_processor.process_chunks(chunks)
    
    # Should record cost for each chunk
    assert mock_cost_tracker.record_chunk_cost.call_count == 5


@pytest.mark.asyncio
async def test_error_handling_in_batch(mock_context_generator, mock_cost_tracker):
    """Test error handling during batch processing."""
    call_count = 0
    
    async def flaky_generate(chunk, doc_context=None):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise Exception("Generation failed")
        return f"Context for {chunk['chunk_id']}"
    
    mock_context_generator.generate_context = flaky_generate
    
    config = ContextualRetrievalConfig(fallback_to_no_context=True, batch_size=5)
    processor = BatchProcessor(mock_context_generator, mock_cost_tracker, config)
    
    chunks = [{"chunk_id": f"chunk_{i}", "text": f"Text {i}"} for i in range(5)]
    
    processed = await processor.process_chunks(chunks)
    
    # Should process all, some without context
    assert len(processed) == 5
    chunks_with_context = [c for c in processed if "context_description" in c]
    assert len(chunks_with_context) == 4  # One failed


@pytest.mark.asyncio
async def test_sequential_batch_processing(mock_context_generator, mock_cost_tracker):
    """Test sequential batch processing."""
    config = ContextualRetrievalConfig(
        batch_size=3,
        enable_parallel_batches=False
    )
    processor = BatchProcessor(mock_context_generator, mock_cost_tracker, config)
    
    chunks = [{"chunk_id": f"chunk_{i}", "text": f"Text {i}"} for i in range(9)]
    
    processed = await processor.process_chunks(chunks)
    
    assert len(processed) == 9
    assert all("context_description" in c for c in processed)


@pytest.mark.asyncio
async def test_batch_creation(batch_processor):
    """Test batch creation logic."""
    chunks = [{"chunk_id": f"chunk_{i}", "text": f"Text {i}"} for i in range(13)]
    
    batches = batch_processor._create_batches(chunks)
    
    # Should create 3 batches (5, 5, 3)
    assert len(batches) == 3
    assert len(batches[0]) == 5
    assert len(batches[1]) == 5
    assert len(batches[2]) == 3


@pytest.mark.asyncio
async def test_contextualized_text_format(batch_processor):
    """Test that contextualized text is properly formatted."""
    chunks = [{"chunk_id": "chunk_1", "text": "Original text"}]
    
    processed = await batch_processor.process_chunks(chunks)
    
    assert len(processed) == 1
    chunk = processed[0]
    
    # Should have contextualized text with prefix
    assert "contextualized_text" in chunk
    assert chunk["contextualized_text"].startswith("Context:")
    assert "Original text" in chunk["contextualized_text"]
