"""
Unit tests for ContextualStorageManager.

Tests dual storage functionality and retrieval format options.
"""

import pytest
from unittest.mock import Mock

from rag_factory.strategies.contextual.storage import ContextualStorageManager
from rag_factory.strategies.contextual.config import ContextualRetrievalConfig


@pytest.fixture
def mock_database():
    """Mock database service for testing."""
    db = Mock()
    db.store_chunk = Mock()
    db.get_chunks_by_ids = Mock(return_value=[])
    return db


@pytest.fixture
def config():
    """Default configuration for testing."""
    return ContextualRetrievalConfig(
        store_original=True,
        store_context=True,
        store_contextualized=True
    )


@pytest.fixture
def storage_manager(mock_database, config):
    """Storage manager instance for testing."""
    return ContextualStorageManager(mock_database, config)


def test_store_chunks(storage_manager, mock_database):
    """Test storing chunks with dual storage."""
    chunks = [
        {
            "chunk_id": "chunk_1",
            "document_id": "doc_1",
            "text": "Original text",
            "context_description": "Context for chunk 1",
            "contextualized_text": "Context: Context for chunk 1\n\nOriginal text",
            "context_generation_method": "llm_template",
            "context_token_count": 20,
            "context_cost": 0.001,
            "metadata": {"key": "value"}
        }
    ]
    
    storage_manager.store_chunks(chunks)
    
    # Should call store_chunk once
    assert mock_database.store_chunk.call_count == 1
    
    # Check stored data
    call_args = mock_database.store_chunk.call_args[0][0]
    assert call_args["chunk_id"] == "chunk_1"
    assert call_args["original_text"] == "Original text"
    assert call_args["context_description"] == "Context for chunk 1"
    assert "Context:" in call_args["contextualized_text"]


def test_store_without_context(storage_manager, mock_database):
    """Test storing chunks without context."""
    chunks = [
        {
            "chunk_id": "chunk_1",
            "document_id": "doc_1",
            "text": "Original text",
            "metadata": {}
        }
    ]
    
    storage_manager.store_chunks(chunks)
    
    call_args = mock_database.store_chunk.call_args[0][0]
    assert "context_description" not in call_args
    assert call_args["text"] == "Original text"


def test_retrieve_original_format(storage_manager, mock_database):
    """Test retrieving chunks in original format."""
    mock_database.get_chunks_by_ids.return_value = [
        {
            "chunk_id": "chunk_1",
            "original_text": "Original text",
            "contextualized_text": "Context: ...\n\nOriginal text",
            "context_description": "Context"
        }
    ]
    
    chunks = storage_manager.retrieve_chunks(["chunk_1"], return_format="original")
    
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Original text"


def test_retrieve_contextualized_format(storage_manager, mock_database):
    """Test retrieving chunks in contextualized format."""
    mock_database.get_chunks_by_ids.return_value = [
        {
            "chunk_id": "chunk_1",
            "original_text": "Original text",
            "contextualized_text": "Context: ...\n\nOriginal text",
            "context_description": "Context"
        }
    ]
    
    chunks = storage_manager.retrieve_chunks(["chunk_1"], return_format="contextualized")
    
    assert len(chunks) == 1
    assert "Context:" in chunks[0]["text"]


def test_retrieve_both_format(storage_manager, mock_database):
    """Test retrieving chunks with both formats."""
    mock_database.get_chunks_by_ids.return_value = [
        {
            "chunk_id": "chunk_1",
            "original_text": "Original text",
            "contextualized_text": "Context: ...\n\nOriginal text",
            "context_description": "Context"
        }
    ]
    
    chunks = storage_manager.retrieve_chunks(["chunk_1"], return_format="both")
    
    assert len(chunks) == 1
    assert "original_text" in chunks[0]
    assert "contextualized_text" in chunks[0]
    assert "context" in chunks[0]


def test_retrieve_context_only(storage_manager, mock_database):
    """Test retrieving only context."""
    mock_database.get_chunks_by_ids.return_value = [
        {
            "chunk_id": "chunk_1",
            "context_description": "Context for chunk"
        }
    ]
    
    chunks = storage_manager.retrieve_chunks(["chunk_1"], return_format="context")
    
    assert len(chunks) == 1
    assert chunks[0]["context"] == "Context for chunk"


def test_store_multiple_chunks(storage_manager, mock_database):
    """Test storing multiple chunks."""
    chunks = [
        {
            "chunk_id": f"chunk_{i}",
            "document_id": "doc_1",
            "text": f"Text {i}",
            "context_description": f"Context {i}",
            "contextualized_text": f"Context: Context {i}\n\nText {i}",
            "metadata": {}
        }
        for i in range(5)
    ]
    
    storage_manager.store_chunks(chunks)
    
    assert mock_database.store_chunk.call_count == 5


def test_config_controls_storage(mock_database):
    """Test that config controls what gets stored."""
    config = ContextualRetrievalConfig(
        store_original=False,
        store_context=False,
        store_contextualized=True
    )
    manager = ContextualStorageManager(mock_database, config)
    
    chunks = [
        {
            "chunk_id": "chunk_1",
            "document_id": "doc_1",
            "text": "Original",
            "context_description": "Context",
            "contextualized_text": "Context: Context\n\nOriginal",
            "metadata": {}
        }
    ]
    
    manager.store_chunks(chunks)
    
    call_args = mock_database.store_chunk.call_args[0][0]
    assert "original_text" not in call_args
    assert "context_description" not in call_args
    assert call_args["text"] == "Context: Context\n\nOriginal"
