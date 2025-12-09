"""Integration tests for pgvector with real database."""
import pytest
import os
import numpy as np
from rag_factory.services.database.postgres import PostgresqlDatabaseService

# Skip if no DB connection available
DB_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DB_URL,
    reason="TEST_DATABASE_URL not set"
)

class TestPgVectorIntegration:
    """Integration tests for pgvector."""
    
    @pytest.fixture
    async def db_service(self):
        """Create database service for testing."""
        # Parse DB_URL for connection params
        # This is a simplified parsing for the test
        service = PostgresqlDatabaseService(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5432)),
            database=os.getenv("DB_NAME", "test_db"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "")
        )
        yield service
        await service.close()

    @pytest.mark.asyncio
    async def test_full_vector_lifecycle(self, db_service):
        """Test storing and searching vectors."""
        # 1. Store vectors
        vectors = [
            ([1.0, 0.0, 0.0], "doc_1"),
            ([0.0, 1.0, 0.0], "doc_2"),
            ([0.0, 0.0, 1.0], "doc_3")
        ]
        
        chunks = [
            {
                "id": doc_id,
                "text": f"content for {doc_id}",
                "embedding": vec,
                "metadata": {"type": "test"}
            }
            for vec, doc_id in vectors
        ]
        
        await db_service.store_chunks(chunks)
        
        # 2. Search
        query = [1.0, 0.1, 0.0]
        results = await db_service.search_chunks(query, top_k=2)
        
        assert len(results) == 2
        assert results[0]['chunk_id'] == 'doc_1'  # Should be closest
        assert results[0]['similarity'] > 0.9
