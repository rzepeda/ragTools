"""Unit tests for vector indexing."""
import pytest
from unittest.mock import AsyncMock, Mock, MagicMock
from rag_factory.database.vector_indexing import VectorIndexManager

class TestVectorIndexing:
    """Test suite for vector index management."""
    
    @pytest.fixture
    def mock_db_service(self):
        service = Mock()
        service.table_name = "chunks"
        pool = MagicMock()
        conn = AsyncMock()
        
        # Mock async context manager
        context = AsyncMock()
        context.__aenter__.return_value = conn
        context.__aexit__.return_value = None
        pool.acquire.return_value = context
        
        service._get_pool = AsyncMock(return_value=pool)
        return service, conn
    
    @pytest.mark.asyncio
    async def test_create_hnsw_index(self, mock_db_service):
        """Test HNSW index creation with parameters."""
        service, conn = mock_db_service
        manager = VectorIndexManager(service)
        
        await manager.create_index(
            index_type='hnsw',
            m=16,
            ef_construction=64
        )
        
        # Verify SQL execution
        call_args = conn.execute.call_args[0][0]
        assert "CREATE INDEX" in call_args
        assert "USING hnsw" in call_args
        assert "m = 16" in call_args
        assert "ef_construction = 64" in call_args
    
    @pytest.mark.asyncio
    async def test_create_ivfflat_index(self, mock_db_service):
        """Test IVFFlat index creation."""
        service, conn = mock_db_service
        manager = VectorIndexManager(service)
        
        await manager.create_index(
            index_type='ivfflat',
            lists=100
        )
        
        call_args = conn.execute.call_args[0][0]
        assert "USING ivfflat" in call_args
        assert "lists = 100" in call_args

    @pytest.mark.asyncio
    async def test_drop_index(self, mock_db_service):
        """Test dropping an index."""
        service, conn = mock_db_service
        manager = VectorIndexManager(service)
        
        await manager.drop_index('hnsw')
        
        call_args = conn.execute.call_args[0][0]
        assert "DROP INDEX" in call_args
        assert "chunks_embedding_hnsw_idx" in call_args

    @pytest.mark.asyncio
    async def test_list_indexes(self, mock_db_service):
        """Test listing indexes."""
        service, conn = mock_db_service
        manager = VectorIndexManager(service)
        
        conn.fetch.return_value = [
            {"indexname": "idx1", "indexdef": "CREATE INDEX..."}
        ]
        
        indexes = await manager.list_indexes()
        assert len(indexes) == 1
        assert indexes[0]['indexname'] == "idx1"
