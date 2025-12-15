"""Unit tests for PostgresqlDatabaseService context support.

Tests the get_context() method and context caching functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from rag_factory.services.database import PostgresqlDatabaseService, DatabaseContext


class TestGetContextMethod:
    """Test PostgresqlDatabaseService.get_context() method."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock SQLAlchemy engine."""
        engine = Mock()
        engine.dispose = Mock()
        return engine

    @pytest.fixture
    def db_service(self):
        """Create PostgresqlDatabaseService instance for testing."""
        with patch('rag_factory.services.database.postgres.asyncpg'):
            service = PostgresqlDatabaseService(
                host="localhost",
                port=5432,
                database="test_db",
                user="test_user",
                password="test_pass"
            )
            yield service

    @patch('rag_factory.services.database.postgres.create_engine')
    def test_get_context_creates_context_with_mappings(self, mock_create_engine, db_service, mock_engine):
        """Test get_context creates DatabaseContext with correct mappings."""
        mock_create_engine.return_value = mock_engine

        table_mapping = {"chunks": "semantic_chunks", "vectors": "semantic_vectors"}
        field_mapping = {"content": "text_content", "embedding": "vector_embedding"}

        context = db_service.get_context(table_mapping, field_mapping)

        assert isinstance(context, DatabaseContext)
        assert context.tables == table_mapping
        assert context.fields == field_mapping
        assert context.engine is mock_engine

    @patch('rag_factory.services.database.postgres.create_engine')
    def test_get_context_without_field_mapping(self, mock_create_engine, db_service, mock_engine):
        """Test get_context works without field mapping."""
        mock_create_engine.return_value = mock_engine

        table_mapping = {"chunks": "semantic_chunks"}

        context = db_service.get_context(table_mapping)

        assert isinstance(context, DatabaseContext)
        assert context.tables == table_mapping
        assert context.fields == {}

    @patch('rag_factory.services.database.postgres.create_engine')
    def test_get_context_caches_contexts(self, mock_create_engine, db_service, mock_engine):
        """Test contexts are cached for same mappings."""
        mock_create_engine.return_value = mock_engine

        table_mapping = {"chunks": "test_chunks"}
        field_mapping = {"content": "text"}

        # Get context twice with same mappings
        context1 = db_service.get_context(table_mapping, field_mapping)
        context2 = db_service.get_context(table_mapping, field_mapping)

        # Should return same instance
        assert context1 is context2
        # Engine should only be created once
        assert mock_create_engine.call_count == 1

    @patch('rag_factory.services.database.postgres.create_engine')
    def test_get_context_different_mappings_create_different_contexts(
        self, mock_create_engine, db_service, mock_engine
    ):
        """Test different mappings create different contexts."""
        mock_create_engine.return_value = mock_engine

        context1 = db_service.get_context({"chunks": "semantic_chunks"})
        context2 = db_service.get_context({"chunks": "keyword_chunks"})

        # Should be different instances
        assert context1 is not context2
        # But share same engine
        assert context1.engine is context2.engine

    @patch('rag_factory.services.database.postgres.create_engine')
    def test_get_context_shares_engine_across_contexts(
        self, mock_create_engine, db_service, mock_engine
    ):
        """Test multiple contexts share the same engine."""
        mock_create_engine.return_value = mock_engine

        context1 = db_service.get_context({"chunks": "table1"})
        context2 = db_service.get_context({"chunks": "table2"})
        context3 = db_service.get_context({"chunks": "table3"})

        # All should share same engine
        assert context1.engine is context2.engine
        assert context2.engine is context3.engine
        # Engine should only be created once
        assert mock_create_engine.call_count == 1


class TestSyncEngineCreation:
    """Test synchronous engine creation for contexts."""

    @pytest.fixture
    def db_service(self):
        """Create PostgresqlDatabaseService instance."""
        with patch('rag_factory.services.database.postgres.asyncpg'):
            service = PostgresqlDatabaseService(
                host="testhost",
                port=5433,
                database="testdb",
                user="testuser",
                password="testpass"
            )
            yield service

    @patch('rag_factory.services.database.postgres.create_engine')
    def test_sync_engine_created_with_correct_connection_string(
        self, mock_create_engine, db_service
    ):
        """Test sync engine is created with correct connection string."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        db_service.get_context({"chunks": "test"})

        # Verify create_engine was called with correct connection string
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args
        connection_string = call_args[0][0]

        assert "postgresql://" in connection_string
        assert "testuser:testpass@" in connection_string
        assert "testhost:5433" in connection_string
        assert "/testdb" in connection_string

    @patch('rag_factory.services.database.postgres.create_engine')
    def test_sync_engine_created_with_connection_pooling(
        self, mock_create_engine, db_service
    ):
        """Test sync engine is created with connection pooling settings."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        db_service.get_context({"chunks": "test"})

        # Verify pooling parameters
        call_kwargs = mock_create_engine.call_args[1]
        assert call_kwargs['pool_size'] == 10
        assert call_kwargs['max_overflow'] == 20
        assert call_kwargs['pool_pre_ping'] is True

    @patch('rag_factory.services.database.postgres.create_engine')
    def test_sync_engine_created_without_password(self, mock_create_engine):
        """Test connection string when password is empty."""
        with patch('rag_factory.services.database.postgres.asyncpg'):
            service = PostgresqlDatabaseService(
                host="localhost",
                port=5432,
                database="db",
                user="user",
                password=""  # Empty password
            )

        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        service.get_context({"chunks": "test"})

        connection_string = mock_create_engine.call_args[0][0]
        # Should not have colon before @ when password is empty
        assert "user@localhost" in connection_string
        assert ":@" not in connection_string

    @patch('rag_factory.services.database.postgres.create_engine')
    def test_sync_engine_reused_across_calls(self, mock_create_engine, db_service):
        """Test sync engine is created once and reused."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        # Create multiple contexts
        db_service.get_context({"chunks": "table1"})
        db_service.get_context({"chunks": "table2"})
        db_service.get_context({"chunks": "table3"})

        # Engine should only be created once
        assert mock_create_engine.call_count == 1

    @patch('rag_factory.services.database.postgres.SQLALCHEMY_AVAILABLE', False)
    def test_get_context_raises_error_when_sqlalchemy_not_available(self, db_service):
        """Test error when SQLAlchemy is not installed."""
        with pytest.raises(ImportError) as exc_info:
            db_service.get_context({"chunks": "test"})

        assert "SQLAlchemy package not installed" in str(exc_info.value)


class TestContextCleanup:
    """Test context cleanup in close() method."""

    @pytest.fixture
    def db_service(self):
        """Create PostgresqlDatabaseService instance."""
        with patch('rag_factory.services.database.postgres.asyncpg'):
            service = PostgresqlDatabaseService()
            yield service

    @patch('rag_factory.services.database.postgres.create_engine')
    @pytest.mark.asyncio
    async def test_close_disposes_sync_engine(self, mock_create_engine, db_service):
        """Test close() disposes of synchronous engine."""
        mock_engine = Mock()
        mock_engine.dispose = Mock()
        mock_create_engine.return_value = mock_engine

        # Create a context (which creates the engine)
        db_service.get_context({"chunks": "test"})

        # Close the service
        await db_service.close()

        # Verify engine was disposed
        mock_engine.dispose.assert_called_once()
        assert db_service._sync_engine is None

    @patch('rag_factory.services.database.postgres.create_engine')
    @pytest.mark.asyncio
    async def test_close_clears_context_cache(self, mock_create_engine, db_service):
        """Test close() clears context cache."""
        mock_engine = Mock()
        mock_engine.dispose = Mock()
        mock_create_engine.return_value = mock_engine

        # Create multiple contexts
        db_service.get_context({"chunks": "table1"})
        db_service.get_context({"chunks": "table2"})

        assert len(db_service._contexts) == 2

        # Close the service
        await db_service.close()

        # Verify cache was cleared
        assert len(db_service._contexts) == 0

    @pytest.mark.asyncio
    async def test_close_handles_no_engine_gracefully(self, db_service):
        """Test close() works when no engine was created."""
        # Close without creating any contexts
        await db_service.close()

        # Should not raise any errors
        assert db_service._sync_engine is None
        assert len(db_service._contexts) == 0


class TestContextCacheKeyGeneration:
    """Test cache key generation for contexts."""

    @pytest.fixture
    def db_service(self):
        """Create PostgresqlDatabaseService instance."""
        with patch('rag_factory.services.database.postgres.asyncpg'):
            service = PostgresqlDatabaseService()
            yield service

    @patch('rag_factory.services.database.postgres.create_engine')
    def test_same_mappings_generate_same_cache_key(
        self, mock_create_engine, db_service
    ):
        """Test same mappings result in cached context."""
        mock_create_engine.return_value = Mock()

        table_mapping = {"chunks": "test_chunks", "vectors": "test_vectors"}
        field_mapping = {"content": "text", "embedding": "vec"}

        context1 = db_service.get_context(table_mapping, field_mapping)
        context2 = db_service.get_context(table_mapping, field_mapping)

        assert context1 is context2

    @patch('rag_factory.services.database.postgres.create_engine')
    def test_different_table_mappings_generate_different_keys(
        self, mock_create_engine, db_service
    ):
        """Test different table mappings create different contexts."""
        mock_create_engine.return_value = Mock()

        context1 = db_service.get_context({"chunks": "table1"})
        context2 = db_service.get_context({"chunks": "table2"})

        assert context1 is not context2

    @patch('rag_factory.services.database.postgres.create_engine')
    def test_different_field_mappings_generate_different_keys(
        self, mock_create_engine, db_service
    ):
        """Test different field mappings create different contexts."""
        mock_create_engine.return_value = Mock()

        table_mapping = {"chunks": "test"}
        context1 = db_service.get_context(table_mapping, {"content": "text1"})
        context2 = db_service.get_context(table_mapping, {"content": "text2"})

        assert context1 is not context2

    @patch('rag_factory.services.database.postgres.create_engine')
    def test_none_field_mapping_vs_empty_dict(
        self, mock_create_engine, db_service
    ):
        """Test None field mapping is treated same as empty dict."""
        mock_create_engine.return_value = Mock()

        table_mapping = {"chunks": "test"}
        context1 = db_service.get_context(table_mapping, None)
        context2 = db_service.get_context(table_mapping, {})

        # These should be treated as the same and return cached context
        assert context1 is context2
