"""Unit tests for pgvector integration."""
import pytest
import sys
from unittest.mock import MagicMock, AsyncMock, patch

# Mock asyncpg before importing the service
with patch.dict(sys.modules, {'asyncpg': MagicMock()}):
    from rag_factory.services.database.postgres import PostgresqlDatabaseService

class TestPgVectorIntegration:
    """Test suite for pgvector functionality."""
    
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
    async def test_cosine_similarity_search(self, mock_pool):
        """Test vector similarity search using cosine distance."""
        pool, conn = mock_pool
        
        with patch('rag_factory.services.database.postgres.ASYNCPG_AVAILABLE', True):
            service = PostgresqlDatabaseService(
                host="localhost",
                password="test"
            )
            service._pool = pool
            
            # Mock search results
            conn.fetch.return_value = [
                {
                    "chunk_id": "doc_0",
                    "text": "test 0",
                    "metadata": None,
                    "similarity": 0.95
                },
                {
                    "chunk_id": "doc_1",
                    "text": "test 1",
                    "metadata": None,
                    "similarity": 0.85
                }
            ]
            
            query_vec = [1.0, 0.0, 0.0]
            results = await service.search_chunks(query_vec, top_k=2)
            
            assert len(results) == 2
            assert results[0]['chunk_id'] == 'doc_0'
            assert results[1]['chunk_id'] == 'doc_1'
            assert results[0]['similarity'] > results[1]['similarity']
            
            # Verify query construction
            call_args = conn.fetch.call_args
            assert "embedding <=> $1::vector" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_vector_storage_format(self, mock_pool):
        """Test that vectors are correctly formatted for storage."""
        pool, conn = mock_pool
        
        with patch('rag_factory.services.database.postgres.ASYNCPG_AVAILABLE', True):
            service = PostgresqlDatabaseService(
                host="localhost",
                password="test"
            )
            service._pool = pool
            
            chunks = [{
                "id": "test_1",
                "text": "content",
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {}
            }]
            
            await service.store_chunks(chunks)
            
            # Verify insertion
            call_args = conn.execute.call_args
            # Check that embedding was formatted as string array
            assert "[0.1,0.2,0.3]" in str(call_args)

    @pytest.mark.asyncio
    async def test_empty_chunks_storage(self, mock_pool):
        """Test handling of empty chunk list."""
        pool, conn = mock_pool
        with patch('rag_factory.services.database.postgres.ASYNCPG_AVAILABLE', True):
            service = PostgresqlDatabaseService()
            service._pool = pool
            
            await service.store_chunks([])
            conn.execute.assert_not_called()
