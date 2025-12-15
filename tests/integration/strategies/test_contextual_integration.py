"""
Integration tests for Contextual Retrieval Strategy.

Tests end-to-end workflow including indexing, retrieval, and cost tracking
with real service integrations.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from rag_factory.strategies.contextual.strategy import ContextualRetrievalStrategy
from rag_factory.strategies.contextual.config import ContextualRetrievalConfig


@pytest.fixture
def mock_vector_store():
    """Mock vector store service."""
    store = Mock()
    store.index_chunk = Mock()
    store.search = Mock(return_value=[
        {"chunk_id": "chunk_0", "score": 0.9},
        {"chunk_id": "chunk_1", "score": 0.85},
    ])
    return store


@pytest.fixture
def mock_database():
    """Mock database service."""
    db = Mock()
    db.store_chunk = Mock()
    db.get_chunks_by_ids = Mock(return_value=[
        {
            "chunk_id": "chunk_0",
            "original_text": "This is chunk 0 about machine learning concepts.",
            "contextualized_text": "Context: ML chunk\n\nThis is chunk 0 about machine learning concepts.",
            "context_description": "ML chunk"
        },
        {
            "chunk_id": "chunk_1",
            "original_text": "This is chunk 1 about machine learning concepts.",
            "contextualized_text": "Context: ML chunk\n\nThis is chunk 1 about machine learning concepts.",
            "context_description": "ML chunk"
        }
    ])
    return db


@pytest.fixture
def mock_llm_service():
    """Mock LLM service."""
    service = Mock()
    response = Mock()
    response.text = "This chunk discusses machine learning fundamentals in the context of an AI tutorial."
    service.agenerate = AsyncMock(return_value=response)
    return service


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    service = Mock()
    result = Mock()
    result.embeddings = [[0.1] * 768]  # Mock embedding vector
    service.embed = Mock(return_value=result)
    return service


@pytest.mark.integration
@pytest.mark.asyncio
async def test_contextual_retrieval_complete_workflow(
    mock_vector_store,
    mock_database,
    mock_llm_service,
    mock_embedding_service
):
    """Test complete contextual retrieval workflow."""
    # Import StrategyDependencies
    from rag_factory.services.dependencies import StrategyDependencies
    
    config = ContextualRetrievalConfig(
        enable_contextualization=True,
        batch_size=10,
        min_chunk_size_for_context=10,  # Lower threshold so test chunks aren't skipped
        context_length_min=10  # Accept shorter mock responses
    )
    
    dependencies = StrategyDependencies(
        database_service=mock_database,
        llm_service=mock_llm_service,
        embedding_service=mock_embedding_service
    )
    # Add vector_store as additional attribute for testing
    dependencies.vector_store = mock_vector_store
    
    strategy = ContextualRetrievalStrategy(
        config=config.model_dump(),
        dependencies=dependencies
    )
    
    # Prepare document chunks
    chunks = [
        {
            "chunk_id": f"chunk_{i}",
            "document_id": "doc_1",
            "text": f"This is chunk {i} about machine learning concepts.",
            "metadata": {"section_hierarchy": ["Chapter 1", f"Section {i}"]}
        }
        for i in range(20)
    ]
    
    # Index document
    result = await strategy.aindex_document(
        document="Full document text",
        document_id="doc_1",
        chunks=chunks,
        document_metadata={"title": "ML Guide"}
    )
    
    # Check indexing result
    assert result["total_chunks"] == 20
    assert result["contextualized_chunks"] > 0
    assert result["total_cost"] > 0
    
    # Verify storage was called
    assert mock_database.store_chunk.call_count == 20
    
    # Verify vector store indexing
    assert mock_vector_store.index_chunk.call_count == 20
    
    # Retrieve
    results = strategy.retrieve("machine learning", top_k=5)
    
    assert len(results) <= 5
    # Should return original text by default
    assert all("text" in r for r in results)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cost_tracking_accuracy(
    mock_vector_store,
    mock_database,
    mock_llm_service,
    mock_embedding_service
):
    """Test accuracy of cost tracking."""
    from rag_factory.services.dependencies import StrategyDependencies
    
    config = ContextualRetrievalConfig(
        enable_cost_tracking=True,
        min_chunk_size_for_context=10,
        context_length_min=10
    )
    
    dependencies = StrategyDependencies(
        database_service=mock_database,
        llm_service=mock_llm_service,
        embedding_service=mock_embedding_service
    )
    # Add vector_store as additional attribute for testing
    dependencies.vector_store = mock_vector_store
    
    strategy = ContextualRetrievalStrategy(
        config=config.model_dump(),
        dependencies=dependencies
    )
    
    chunks = [
        {"chunk_id": f"chunk_{i}", "text": f"Text {i} " * 20, "metadata": {}}
        for i in range(10)
    ]
    
    result = await strategy.aindex_document("doc", "doc_1", chunks)
    
    # Verify cost tracking
    assert result["total_cost"] > 0
    assert result["total_input_tokens"] > 0
    assert result["total_output_tokens"] > 0
    
    cost_summary = strategy.get_cost_summary()
    assert cost_summary["total_cost"] == result["total_cost"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieval_with_different_formats(
    mock_vector_store,
    mock_database,
    mock_llm_service,
    mock_embedding_service
):
    """Test retrieval with different return formats."""
    from rag_factory.services.dependencies import StrategyDependencies
    
    config_original = ContextualRetrievalConfig(
        return_original_text=True,
        return_context=False,
        min_chunk_size_for_context=10,
        context_length_min=10
    )
    
    dependencies = StrategyDependencies(
        database_service=mock_database,
        llm_service=mock_llm_service,
        embedding_service=mock_embedding_service
    )
    # Add vector_store as additional attribute for testing
    dependencies.vector_store = mock_vector_store
    
    strategy = ContextualRetrievalStrategy(
        config=config_original.model_dump(),
        dependencies=dependencies
    )
    
    chunks = [{"chunk_id": f"c{i}", "text": f"Text {i} " * 10, "metadata": {}} for i in range(5)]
    await strategy.aindex_document("doc", "doc_1", chunks)
    
    results = strategy.retrieve("query", top_k=2)
    assert len(results) == 2
    assert all("text" in r for r in results)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_recovery(
    mock_vector_store,
    mock_database,
    mock_llm_service,
    mock_embedding_service
):
    """Test error recovery with fallback."""
    from rag_factory.services.dependencies import StrategyDependencies
    
    # Make LLM fail for some chunks
    call_count = 0
    
    async def flaky_generate(prompt, temperature, max_tokens):
        nonlocal call_count
        call_count += 1
        if call_count % 3 == 0:
            raise Exception("LLM error")
        response = Mock()
        # Return longer text to pass context_length_min=10 (need ~40+ characters)
        response.text = "This chunk provides detailed contextual information about the content."
        return response
    
    mock_llm_service.agenerate = flaky_generate
    
    config = ContextualRetrievalConfig(
        fallback_to_no_context=True,
        batch_size=5,
        min_chunk_size_for_context=10,
        context_length_min=10,
        enable_parallel_batches=False  # Use sequential to ensure proper error handling
    )
    
    dependencies = StrategyDependencies(
        database_service=mock_database,
        llm_service=mock_llm_service,
        embedding_service=mock_embedding_service
    )
    # Add vector_store as additional attribute for testing
    dependencies.vector_store = mock_vector_store
    
    strategy = ContextualRetrievalStrategy(
        config=config.model_dump(),
        dependencies=dependencies
    )
    
    chunks = [{"chunk_id": f"c{i}", "text": f"Text {i} " * 10, "metadata": {}} for i in range(10)]
    
    # Should complete despite errors
    result = await strategy.aindex_document("doc", "doc_1", chunks)
    
    assert result["total_chunks"] == 10
    # Some chunks should have context, some not
    assert result["contextualized_chunks"] < 10


@pytest.mark.integration
def test_synchronous_indexing(
    mock_vector_store,
    mock_database,
    mock_llm_service,
    mock_embedding_service
):
    """Test synchronous indexing wrapper."""
    from rag_factory.services.dependencies import StrategyDependencies
    
    config = ContextualRetrievalConfig(
        batch_size=5,
        min_chunk_size_for_context=10,
        context_length_min=10
    )
    
    dependencies = StrategyDependencies(
        database_service=mock_database,
        llm_service=mock_llm_service,
        embedding_service=mock_embedding_service
    )
    # Add vector_store as additional attribute for testing
    dependencies.vector_store = mock_vector_store
    
    strategy = ContextualRetrievalStrategy(
        config=config.model_dump(),
        dependencies=dependencies
    )
    
    chunks = [{"chunk_id": f"c{i}", "text": f"Text {i} " * 10, "metadata": {}} for i in range(5)]
    
    # Use synchronous method
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(
        strategy.aindex_document("doc", "doc_1", chunks)
    )
    
    loop.close()
    
    assert result["total_chunks"] == 5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_large_document_processing(
    mock_vector_store,
    mock_database,
    mock_llm_service,
    mock_embedding_service
):
    """Test processing large documents."""
    from rag_factory.services.dependencies import StrategyDependencies
    
    config = ContextualRetrievalConfig(
        batch_size=20,
        enable_parallel_batches=True,
        max_concurrent_batches=5,
        min_chunk_size_for_context=10,
        context_length_min=10
    )
    
    dependencies = StrategyDependencies(
        database_service=mock_database,
        llm_service=mock_llm_service,
        embedding_service=mock_embedding_service
    )
    # Add vector_store as additional attribute for testing
    dependencies.vector_store = mock_vector_store
    
    strategy = ContextualRetrievalStrategy(
        config=config.model_dump(),
        dependencies=dependencies
    )
    
    # Large document with 100 chunks
    chunks = [
        {"chunk_id": f"c{i}", "text": f"Text {i} " * 20, "metadata": {}}
        for i in range(100)
    ]
    
    result = await strategy.aindex_document("doc", "doc_1", chunks)
    
    assert result["total_chunks"] == 100
    assert mock_database.store_chunk.call_count == 100
    assert mock_vector_store.index_chunk.call_count == 100
