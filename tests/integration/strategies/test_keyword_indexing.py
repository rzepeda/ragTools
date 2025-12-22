"""Unit tests for KeywordIndexing strategy."""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from rag_factory.strategies.indexing.keyword_indexing import KeywordIndexing
from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import ServiceDependency, StrategyDependencies
from rag_factory.core.indexing_interface import IndexingContext

@pytest.fixture
def mock_database_service():
    service = Mock()
    service.get_chunks_for_documents = AsyncMock()
    service.store_keyword_index = AsyncMock()
    return service

@pytest.fixture
def strategy_dependencies(mock_database_service):
    return StrategyDependencies(
        database_service=mock_database_service,
        embedding_service=Mock(), # Not used but required by StrategyDependencies
        llm_service=Mock(),
        graph_service=Mock(),
        reranker_service=Mock()
    )

@pytest.fixture
def indexing_context(mock_database_service):
    return IndexingContext(
        database_service=mock_database_service,
        config={}
    )

@pytest.fixture
def keyword_indexing(strategy_dependencies):
    config = {"max_keywords": 100, "ngram_range": (1, 1)}
    return KeywordIndexing(config, strategy_dependencies)

def test_capabilities(keyword_indexing):
    """Test that strategy produces correct capabilities."""
    capabilities = keyword_indexing.produces()
    assert IndexCapability.KEYWORDS in capabilities
    assert IndexCapability.DATABASE in capabilities
    assert len(capabilities) == 2

def test_dependencies(keyword_indexing):
    """Test that strategy requires correct services."""
    deps = keyword_indexing.requires_services()
    assert ServiceDependency.DATABASE in deps
    assert len(deps) == 1

@pytest.mark.asyncio
async def test_process_success(
    keyword_indexing,
    indexing_context,
    mock_database_service
):
    """Test successful keyword extraction and indexing."""
    # Setup
    documents = [{"id": "doc1", "text": "foo"}, {"id": "doc2", "text": "bar"}]
    chunks = [
        {"id": "chunk1", "text": "machine learning is great", "document_id": "doc1"},
        {"id": "chunk2", "text": "deep learning is awesome", "document_id": "doc2"}
    ]
    mock_database_service.get_chunks_for_documents.return_value = chunks
    
    # Execute
    result = await keyword_indexing.process(documents, indexing_context)
    
    # Verify
    assert result.document_count == 2
    assert result.chunk_count == 2
    assert result.metadata["total_keywords"] > 0
    
    # Verify database calls
    mock_database_service.get_chunks_for_documents.assert_called_once()
    mock_database_service.store_keyword_index.assert_called_once()
    
    # Verify index content (roughly)
    args, _ = mock_database_service.store_keyword_index.call_args
    inverted_index = args[0]
    assert "learning" in inverted_index
    assert len(inverted_index["learning"]) == 2  # In both chunks

@pytest.mark.asyncio
async def test_process_no_chunks(
    keyword_indexing,
    indexing_context,
    mock_database_service
):
    """Test that chunks are created when none are found."""
    # Setup
    documents = [{"id": "doc1", "text": "machine learning is great"}]
    mock_database_service.get_chunks_for_documents.return_value = []
    mock_database_service.store_chunks = AsyncMock()
    
    # Execute
    result = await keyword_indexing.process(documents, indexing_context)
    
    # Verify chunks were created and stored
    mock_database_service.store_chunks.assert_called_once()
    stored_chunks = mock_database_service.store_chunks.call_args[0][0]
    assert len(stored_chunks) > 0
    assert stored_chunks[0]['document_id'] == 'doc1'

@pytest.mark.asyncio
async def test_process_empty_text(
    keyword_indexing,
    indexing_context,
    mock_database_service
):
    """Test handling of empty text content."""
    # Setup
    documents = [{"id": "doc1"}]
    chunks = [{"id": "chunk1", "text": "   ", "document_id": "doc1"}]
    mock_database_service.get_chunks_for_documents.return_value = chunks
    
    # Execute
    result = await keyword_indexing.process(documents, indexing_context)
    
    # Verify
    assert "warning" in result.metadata
    mock_database_service.store_keyword_index.assert_not_called()

@pytest.mark.asyncio
async def test_process_stop_words_only(
    keyword_indexing,
    indexing_context,
    mock_database_service
):
    """Test handling of text containing only stop words."""
    # Setup
    documents = [{"id": "doc1"}]
    chunks = [{"id": "chunk1", "text": "the and of", "document_id": "doc1"}]
    mock_database_service.get_chunks_for_documents.return_value = chunks
    
    # Execute
    result = await keyword_indexing.process(documents, indexing_context)
    
    # Verify
    assert "warning" in result.metadata
    mock_database_service.store_keyword_index.assert_not_called()
