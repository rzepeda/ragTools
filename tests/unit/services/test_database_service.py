"""Unit tests for database service."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys

# Mock asyncpg before importing service
sys.modules["asyncpg"] = MagicMock()

import json
from rag_factory.services.database.postgres import PostgresqlDatabaseService

@pytest.fixture
def mock_pool():
    """Create mock connection pool."""
    pool = AsyncMock()
    conn = AsyncMock()
    
    # Mock the async context manager for pool.acquire()
    class AsyncContextManagerMock:
        async def __aenter__(self):
            return conn
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None
    
    # Use Mock (not AsyncMock) for acquire since it's not an async function
    # It just returns an async context manager
    pool.acquire = Mock(side_effect=lambda: AsyncContextManagerMock())
    
    return pool, conn

@pytest.fixture
def service(mock_pool):
    """Create service with mock pool."""
    pool, _ = mock_pool
    
    service = PostgresqlDatabaseService(
        host="localhost",
        port=5432,
        database="test_db",
        user="test_user",
        password="test_password"
    )
    # Pre-set the pool to avoid connection attempts
    service._pool = pool
    return service


@pytest.mark.asyncio
async def test_service_initialization():
    """Test service initializes correctly."""
    service = PostgresqlDatabaseService(
        host="localhost",
        port=5432,
        database="test_db",
        user="test_user",
        password="test_password"
    )
    assert service.host == "localhost"
    assert service.port == 5432
    assert service.database == "test_db"
    assert service.user == "test_user"
    assert service.table_name == "chunks"

@pytest.mark.asyncio
async def test_ensure_table(service, mock_pool):
    """Test table creation."""
    pool, conn = mock_pool
    
    # Clear the pre-set pool so _get_pool() will create a new one
    service._pool = None
    
    # Patch asyncpg.create_pool to return our mock pool
    with patch("rag_factory.services.database.postgres.asyncpg.create_pool", new=AsyncMock(return_value=pool)):
        # Trigger pool creation which calls ensure_table
        await service._get_pool()
    
    # Check if create extension and table queries were executed
    assert conn.execute.call_count >= 3
    calls = [call[0][0] for call in conn.execute.call_args_list]
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in call for call in calls)
    assert any("CREATE TABLE IF NOT EXISTS chunks" in call for call in calls)
    assert any("CREATE INDEX IF NOT EXISTS chunks_embedding_idx" in call for call in calls)

@pytest.mark.asyncio
async def test_store_chunks(service, mock_pool):
    """Test storing chunks."""
    pool, conn = mock_pool
    
    chunks = [
        {
            "chunk_id": "1",
            "text": "Hello",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {"source": "doc1"}
        }
    ]
    
    await service.store_chunks(chunks)
    
    conn.execute.assert_called()
    call_args = conn.execute.call_args
    assert "INSERT INTO chunks" in call_args[0][0]
    assert "[0.1,0.2,0.3]" in call_args[0][0]  # Embedding is embedded in SQL string
    assert call_args[0][1] == "1"  # chunk_id
    assert call_args[0][2] == "Hello"  # text
    assert call_args[0][3] == '{"source": "doc1"}'  # metadata

@pytest.mark.asyncio
async def test_store_empty_chunks(service, mock_pool):
    """Test storing empty chunks list."""
    pool, conn = mock_pool
    await service.store_chunks([])
    conn.execute.assert_not_called()

@pytest.mark.asyncio
async def test_search_chunks(service, mock_pool):
    """Test searching chunks."""
    pool, conn = mock_pool
    
    # Mock search results
    conn.fetch.return_value = [
        {
            "chunk_id": "1",
            "text": "Hello",
            "metadata": '{"source": "doc1"}',
            "similarity": 0.95
        }
    ]
    
    results = await service.search_chunks([0.1, 0.2, 0.3], top_k=5)
    
    assert len(results) == 1
    assert results[0].chunk_id == "1"
    assert results[0].metadata["similarity"] == 0.95
    assert results[0].metadata["source"] == "doc1"
    
    conn.fetch.assert_called_once()
    call_args = conn.fetch.call_args
    assert "SELECT" in call_args[0][0]
    assert "[0.1,0.2,0.3]" in call_args[0][0]  # Embedding is embedded in SQL string
    assert call_args[0][1] == 5  # top_k is the only parameter

@pytest.mark.asyncio
async def test_get_chunk(service, mock_pool):
    """Test retrieving single chunk."""
    pool, conn = mock_pool
    
    conn.fetchrow.return_value = {
        "chunk_id": "1",
        "text": "Hello",
        "metadata": '{"source": "doc1"}'
    }
    
    chunk = await service.get_chunk("1")
    
    assert chunk["chunk_id"] == "1"
    assert chunk["text"] == "Hello"
    
    conn.fetchrow.assert_called_once()

@pytest.mark.asyncio
async def test_get_chunk_not_found(service, mock_pool):
    """Test retrieving non-existent chunk."""
    pool, conn = mock_pool
    conn.fetchrow.return_value = None
    
    with pytest.raises(ValueError, match="Chunk not found"):
        await service.get_chunk("999")

@pytest.mark.asyncio
async def test_get_chunks_for_documents(service, mock_pool):
    """Test retrieving chunks for documents."""
    pool, conn = mock_pool
    
    conn.fetch.return_value = [
        {
            "chunk_id": "1",
            "text": "Hello",
            "metadata": '{"document_id": "doc1"}'
        },
        {
            "chunk_id": "2",
            "text": "World",
            "metadata": '{"document_id": "doc1"}'
        }
    ]
    
    chunks = await service.get_chunks_for_documents(["doc1"])
    
    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "1"
    assert chunks[1]["chunk_id"] == "2"
    
    conn.fetch.assert_called_once()

@pytest.mark.asyncio
async def test_close(service, mock_pool):
    """Test closing connection pool."""
    pool, _ = mock_pool
    
    # Initialize pool
    await service._get_pool()
    assert service._pool is not None
    
    await service.close()
    
    pool.close.assert_called_once()
    assert service._pool is None

@pytest.mark.asyncio
async def test_context_manager(service, mock_pool):
    """Test async context manager."""
    pool, _ = mock_pool
    
    async with service as s:
        assert s == service
        await s._get_pool()
        
    pool.close.assert_called_once()
