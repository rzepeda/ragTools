"""Unit tests for InMemoryIndexing strategy."""

import pytest
from unittest.mock import Mock
from rag_factory.strategies.indexing.in_memory import InMemoryIndexing
from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import ServiceDependency, StrategyDependencies
from rag_factory.core.indexing_interface import IndexingContext


@pytest.fixture
def mock_database_service():
    """Create mock database service."""
    service = Mock()
    return service


@pytest.fixture
def strategy_dependencies():
    """Create strategy dependencies with no services."""
    return StrategyDependencies()


@pytest.fixture
def indexing_context(mock_database_service):
    """Create indexing context."""
    return IndexingContext(
        database_service=mock_database_service,
        config={}
    )


@pytest.fixture
def in_memory_indexing(strategy_dependencies):
    """Create InMemoryIndexing strategy."""
    config = {"chunk_size": 512}
    return InMemoryIndexing(config, strategy_dependencies)


@pytest.fixture(autouse=True)
def clear_storage():
    """Clear storage before and after each test."""
    InMemoryIndexing.clear_storage()
    yield
    InMemoryIndexing.clear_storage()


def test_capabilities(in_memory_indexing):
    """Test that strategy produces correct capabilities."""
    capabilities = in_memory_indexing.produces()
    assert IndexCapability.CHUNKS in capabilities
    assert IndexCapability.IN_MEMORY in capabilities
    assert len(capabilities) == 2


def test_dependencies(in_memory_indexing):
    """Test that strategy requires no services."""
    deps = in_memory_indexing.requires_services()
    assert len(deps) == 0
    assert deps == set()


@pytest.mark.asyncio
async def test_process_success(
    in_memory_indexing,
    indexing_context
):
    """Test successful document processing and chunking."""
    # Setup
    documents = [
        {"id": "doc1", "text": "This is a test document with some content."},
        {"id": "doc2", "text": "Another document for testing purposes."}
    ]
    
    # Execute
    result = await in_memory_indexing.process(documents, indexing_context)
    
    # Verify result
    assert result.document_count == 2
    assert result.chunk_count > 0
    assert IndexCapability.CHUNKS in result.capabilities
    assert IndexCapability.IN_MEMORY in result.capabilities
    assert result.metadata["storage_type"] == "in_memory"
    assert result.metadata["chunk_size"] == 512
    
    # Verify context metrics
    assert indexing_context.metrics["chunks_created"] > 0
    assert indexing_context.metrics["documents_processed"] == 2


@pytest.mark.asyncio
async def test_storage_operations(
    in_memory_indexing,
    indexing_context
):
    """Test storage operations (store, retrieve, clear)."""
    # Setup
    documents = [{"id": "doc1", "text": "Sample text for testing storage."}]
    
    # Execute
    result = await in_memory_indexing.process(documents, indexing_context)
    
    # Verify chunks are stored
    chunk = InMemoryIndexing.get_chunk("doc1_chunk_0")
    assert chunk is not None
    assert chunk["id"] == "doc1_chunk_0"
    assert chunk["document_id"] == "doc1"
    assert chunk["text"] == "Sample text for testing storage."
    assert chunk["index"] == 0
    
    # Verify all chunks can be retrieved
    all_chunks = InMemoryIndexing.get_all_chunks()
    assert len(all_chunks) == result.chunk_count
    
    # Verify clear works
    InMemoryIndexing.clear_storage()
    assert len(InMemoryIndexing.get_all_chunks()) == 0
    assert InMemoryIndexing.get_chunk("doc1_chunk_0") is None


@pytest.mark.asyncio
async def test_clear_storage(
    in_memory_indexing,
    indexing_context
):
    """Test that clear_storage removes all chunks."""
    # Setup
    documents = [
        {"id": "doc1", "text": "First document"},
        {"id": "doc2", "text": "Second document"}
    ]
    
    # Execute
    await in_memory_indexing.process(documents, indexing_context)
    
    # Verify chunks exist
    assert len(InMemoryIndexing.get_all_chunks()) > 0
    
    # Clear storage
    InMemoryIndexing.clear_storage()
    
    # Verify storage is empty
    assert len(InMemoryIndexing.get_all_chunks()) == 0


@pytest.mark.asyncio
async def test_get_chunk(
    in_memory_indexing,
    indexing_context
):
    """Test retrieving specific chunks by ID."""
    # Setup
    documents = [{"id": "test_doc", "text": "Test content"}]
    
    # Execute
    await in_memory_indexing.process(documents, indexing_context)
    
    # Verify chunk retrieval
    chunk = InMemoryIndexing.get_chunk("test_doc_chunk_0")
    assert chunk is not None
    assert chunk["id"] == "test_doc_chunk_0"
    assert chunk["document_id"] == "test_doc"
    
    # Verify non-existent chunk returns None
    assert InMemoryIndexing.get_chunk("nonexistent") is None


@pytest.mark.asyncio
async def test_multiple_documents(
    in_memory_indexing,
    indexing_context
):
    """Test processing multiple documents."""
    # Setup
    documents = [
        {"id": "doc1", "text": "First document content"},
        {"id": "doc2", "text": "Second document content"},
        {"id": "doc3", "text": "Third document content"}
    ]
    
    # Execute
    result = await in_memory_indexing.process(documents, indexing_context)
    
    # Verify
    assert result.document_count == 3
    assert result.chunk_count >= 3  # At least one chunk per document
    
    # Verify each document has chunks
    assert InMemoryIndexing.get_chunk("doc1_chunk_0") is not None
    assert InMemoryIndexing.get_chunk("doc2_chunk_0") is not None
    assert InMemoryIndexing.get_chunk("doc3_chunk_0") is not None


@pytest.mark.asyncio
async def test_empty_documents(
    in_memory_indexing,
    indexing_context
):
    """Test handling of empty documents."""
    # Setup
    documents = [
        {"id": "empty1", "text": ""},
        {"id": "empty2"}  # No text field
    ]
    
    # Execute
    result = await in_memory_indexing.process(documents, indexing_context)
    
    # Verify
    assert result.document_count == 2
    # Empty documents should not create chunks
    assert result.chunk_count == 0


@pytest.mark.asyncio
async def test_custom_chunk_size(
    strategy_dependencies,
    indexing_context
):
    """Test custom chunk size configuration."""
    # Setup with small chunk size
    config = {"chunk_size": 10}
    strategy = InMemoryIndexing(config, strategy_dependencies)
    
    documents = [{"id": "doc1", "text": "This is a longer text that should be split into multiple chunks"}]
    
    # Execute
    result = await strategy.process(documents, indexing_context)
    
    # Verify multiple chunks were created
    assert result.chunk_count > 1
    assert result.metadata["chunk_size"] == 10
    
    # Verify chunk sizes
    all_chunks = InMemoryIndexing.get_all_chunks()
    for chunk in all_chunks:
        assert len(chunk["text"]) <= 10


@pytest.mark.asyncio
async def test_chunk_metadata_preservation(
    in_memory_indexing,
    indexing_context
):
    """Test that document metadata is preserved in chunks."""
    # Setup
    documents = [{
        "id": "doc1",
        "text": "Sample text",
        "metadata": {"source": "test", "author": "tester"}
    }]
    
    # Execute
    await in_memory_indexing.process(documents, indexing_context)
    
    # Verify metadata is preserved
    chunk = InMemoryIndexing.get_chunk("doc1_chunk_0")
    assert chunk["metadata"]["source"] == "test"
    assert chunk["metadata"]["author"] == "tester"


@pytest.mark.asyncio
async def test_shared_storage_across_instances(
    strategy_dependencies,
    indexing_context
):
    """Test that storage is shared across strategy instances."""
    # Setup - create two separate instances
    strategy1 = InMemoryIndexing({"chunk_size": 512}, strategy_dependencies)
    strategy2 = InMemoryIndexing({"chunk_size": 512}, strategy_dependencies)
    
    documents1 = [{"id": "doc1", "text": "First instance"}]
    documents2 = [{"id": "doc2", "text": "Second instance"}]
    
    # Execute
    await strategy1.process(documents1, indexing_context)
    await strategy2.process(documents2, indexing_context)
    
    # Verify both instances can access all chunks
    all_chunks = InMemoryIndexing.get_all_chunks()
    assert len(all_chunks) == 2
    
    # Both chunks should be accessible from either instance
    assert InMemoryIndexing.get_chunk("doc1_chunk_0") is not None
    assert InMemoryIndexing.get_chunk("doc2_chunk_0") is not None


@pytest.mark.asyncio
async def test_document_without_id(
    in_memory_indexing,
    indexing_context
):
    """Test handling of documents without explicit IDs."""
    # Setup
    documents = [{"text": "Document without ID"}]
    
    # Execute
    result = await in_memory_indexing.process(documents, indexing_context)
    
    # Verify - should generate an ID
    assert result.chunk_count > 0
    all_chunks = InMemoryIndexing.get_all_chunks()
    assert len(all_chunks) > 0
    
    # Chunk should have a generated ID
    chunk = all_chunks[0]
    assert chunk["id"].startswith("doc_")
    assert chunk["id"].endswith("_chunk_0")
