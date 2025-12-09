"""Unit tests for database migrations."""
import pytest
import os
from unittest.mock import AsyncMock, Mock, MagicMock, patch, mock_open
from rag_factory.database.migrations import MigrationManager

class TestDatabaseMigrations:
    """Test suite for migration system."""
    
    @pytest.fixture
    def mock_db_service(self):
        service = Mock()
        pool = MagicMock()
        conn = AsyncMock()
        
        # Mock pool acquire context manager
        pool_context = AsyncMock()
        pool_context.__aenter__.return_value = conn
        pool_context.__aexit__.return_value = None
        pool.acquire.return_value = pool_context
        
        # Mock transaction context manager
        tx_context = AsyncMock()
        tx_context.__aenter__.return_value = conn
        tx_context.__aexit__.return_value = None
        # conn.transaction must be a MagicMock (not AsyncMock) because it returns a context manager, isn't awaited itself
        conn.transaction = MagicMock(return_value=tx_context)
        
        service._get_pool = AsyncMock(return_value=pool)
        return service, conn
    
    @pytest.mark.asyncio
    async def test_migration_execution_order(self, mock_db_service):
        """Test that migrations execute in correct order."""
        service, conn = mock_db_service
        
        # Mock existing migrations
        conn.fetch.return_value = []  # No applied migrations
        
        with patch('os.listdir', return_value=['002_second.sql', '001_first.sql']), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="SELECT 1;")):
            
            manager = MigrationManager(service)
            executed = await manager.run_migrations()
            
            assert executed == ['001_first.sql', '002_second.sql']
            assert conn.execute.call_count >= 2  # Init table + 2 migrations + 2 version inserts

    @pytest.mark.asyncio
    async def test_migration_idempotency(self, mock_db_service):
        """Test that applied migrations are skipped."""
        service, conn = mock_db_service
        
        # Mock existing migrations
        conn.fetch.return_value = [{'version': '001'}]
        
        with patch('os.listdir', return_value=['001_first.sql', '002_second.sql']), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="SELECT 1;")):
            
            manager = MigrationManager(service)
            executed = await manager.run_migrations()
            
            assert executed == ['002_second.sql']  # Only second one runs

    @pytest.mark.asyncio
    async def test_get_current_version(self, mock_db_service):
        """Test version retrieval."""
        service, conn = mock_db_service
        conn.fetch.return_value = [{'version': '001'}, {'version': '002'}]
        
        manager = MigrationManager(service)
        version = await manager.get_current_version()
        
        assert version == '002'
