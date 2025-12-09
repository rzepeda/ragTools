"""Unit tests for batch operations."""
import pytest
import sys
from unittest.mock import MagicMock, AsyncMock, patch

# Mock asyncpg before importing the service
with patch.dict(sys.modules, {'asyncpg': MagicMock()}):
    from rag_factory.services.database.postgres import PostgresqlDatabaseService

class TestBatchOperations:
    """Test suite for batch operations."""
    
    @pytest.fixture
    def mock_pool(self):
        pool = MagicMock()
        conn = AsyncMock()
        # Mock async context manager
        context = AsyncMock()
        context.__aenter__.return_value = conn
        context.__aexit__.return_value = None
        pool.acquire.return_value = context
        return pool, conn

    @pytest.mark.asyncio
    async def test_batch_vector_insertion(self, mock_pool):
        """Test inserting a batch of vectors."""
        pool, conn = mock_pool
        with patch('rag_factory.services.database.postgres.ASYNCPG_AVAILABLE', True):
            service = PostgresqlDatabaseService(password="test")
            service._pool = pool
            
            chunks = [
                {
                    "id": f"doc_{i}",
                    "text": f"text {i}",
                    "embedding": [0.1] * 384,
                    "metadata": {"i": i}
                }
                for i in range(100)
            ]
            
            await service.store_chunks(chunks)
            
            # Should execute insert for each chunk
            assert conn.execute.call_count == 100

    @pytest.mark.asyncio
    async def test_store_chunks_with_hierarchy(self, mock_pool):
        """Test storing chunks with hierarchy metadata."""
        pool, conn = mock_pool
        with patch('rag_factory.services.database.postgres.ASYNCPG_AVAILABLE', True):
            service = PostgresqlDatabaseService(password="test")
            service._pool = pool
            
            chunks = [{
                "id": "child_1",
                "text": "child text",
                "embedding": [0.1] * 384,
                "level": 1,
                "parent_id": "root_1",
                "path": [0, 1],
                "document_id": "doc_1"
            }]
            
            await service.store_chunks_with_hierarchy(chunks)
            
            # Verify metadata enrichment
            call_args = conn.execute.call_args
            # Query is 0, chunk_id is 1, text is 2, embedding is 3, metadata is 4
            metadata_json = call_args[0][4]
            assert '"level": 1' in metadata_json
            assert '"parent_id": "root_1"' in metadata_json
