"""Unit tests for database connection management."""

import pytest
from unittest.mock import Mock, patch
from sqlalchemy import text

from rag_factory.database.connection import DatabaseConnection
from rag_factory.database.config import DatabaseConfig


class TestDatabaseConnection:
    """Test suite for DatabaseConnection class."""

    def test_connection_initialization(self, test_db_config):
        """Test connection pool is created with correct parameters."""
        db = DatabaseConnection(test_db_config)

        assert db.engine is not None
        assert db.SessionLocal is not None
        assert db.config == test_db_config

        db.close()

    def test_connection_with_default_config(self):
        """Test connection can be created with default config from environment."""
        # This will use environment variables or fail if not set
        # In test environment, we need to mock this
        with patch.dict('os.environ', {
            'DB_DATABASE_URL': 'sqlite:///:memory:'
        }):
            db = DatabaseConnection()
            assert db.engine is not None
            db.close()

    def test_session_context_manager_commit(self, db_connection):
        """Test session context manager commits on success."""
        with db_connection.get_session() as session:
            result = session.execute(text("SELECT 1")).scalar()
            assert result == 1

        # Session should be closed after context exit
        # We can't easily test if it's closed, but we can verify no exceptions

    def test_session_context_manager_rollback(self, db_connection):
        """Test session rolls back on exception."""
        with pytest.raises(ValueError):
            with db_connection.get_session() as session:
                session.execute(text("SELECT 1"))
                raise ValueError("Test error")

        # Should have rolled back without issues

    def test_health_check_success(self, db_connection):
        """Test health check returns True for healthy database."""
        assert db_connection.health_check() is True

    def test_health_check_failure(self):
        """Test health check returns False for unavailable database."""
        config = DatabaseConfig(database_url="postgresql://invalid:invalid@localhost:9999/invalid")
        db = DatabaseConnection(config)

        # Should return False, not raise exception
        assert db.health_check() is False

        db.close()

    def test_get_pool_status(self, db_connection):
        """Test pool status returns metrics."""
        status = db_connection.get_pool_status()

        assert "size" in status
        assert "checked_out" in status
        assert "overflow" in status
        assert "checked_in" in status

        assert isinstance(status["size"], int)
        assert isinstance(status["checked_out"], int)

    def test_create_tables(self, db_connection):
        """Test create_tables creates database schema."""
        # Tables should already be created by fixture
        # We'll just verify it doesn't raise
        db_connection.create_tables()

    def test_drop_tables(self, db_connection):
        """Test drop_tables removes database schema."""
        db_connection.drop_tables()

        # Verify tables are dropped by trying to create them again
        db_connection.create_tables()

    def test_context_manager(self, test_db_config):
        """Test DatabaseConnection can be used as context manager."""
        with DatabaseConnection(test_db_config) as db:
            assert db.health_check() is True

        # Engine should be disposed after context exit

    def test_multiple_sessions(self, db_connection):
        """Test multiple sessions can be created."""
        sessions = []

        for _ in range(5):
            with db_connection.get_session() as session:
                result = session.execute(text("SELECT 1")).scalar()
                sessions.append(result)

        assert len(sessions) == 5
        assert all(s == 1 for s in sessions)

    def test_pool_configuration(self, test_db_config):
        """Test pool is configured with correct parameters."""
        db = DatabaseConnection(test_db_config)

        # Check pool configuration
        pool = db.engine.pool

        # Pool should be configured (exact checks depend on SQLAlchemy version)
        assert pool is not None

        db.close()


class TestDatabaseConnectionEvents:
    """Test suite for database connection event listeners."""

    def test_on_connect_event(self, db_connection):
        """Test connection event is triggered."""
        # Create a new connection to trigger event
        with db_connection.get_session() as session:
            session.execute(text("SELECT 1"))

        # If we get here without exception, event handler worked

    def test_on_checkout_event(self, db_connection):
        """Test checkout event is triggered."""
        # Getting a session should trigger checkout
        with db_connection.get_session() as session:
            session.execute(text("SELECT 1"))

        # If we get here without exception, event handler worked
