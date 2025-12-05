"""Database connection management with pooling.

This module provides connection pooling, session management, and
health check utilities for PostgreSQL database access.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import Pool

from rag_factory.database.config import DatabaseConfig
from rag_factory.database.models import Base

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Database connection manager with pooling.

    Manages SQLAlchemy engine with connection pooling, provides
    session context managers, and includes health check functionality.

    Attributes:
        config: Database configuration settings
        engine: SQLAlchemy engine with connection pool
        SessionLocal: Session factory for creating database sessions
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """Initialize database connection manager.

        Args:
            config: Database configuration. If None, loads from environment.
        """
        self.config = config or DatabaseConfig()
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False
        )

        logger.info(
            "Database connection initialized: pool_size=%d, max_overflow=%d",
            self.config.pool_size,
            self.config.max_overflow
        )

    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine with connection pooling.

        Returns:
            Configured SQLAlchemy engine
        """
        # Base configuration
        engine_kwargs = {
            "echo": self.config.echo,
            "future": True
        }

        # Add pooling parameters only for databases that support them
        # SQLite doesn't support these parameters
        if not self.config.database_url.startswith("sqlite"):
            engine_kwargs.update({
                "pool_size": self.config.pool_size,
                "max_overflow": self.config.max_overflow,
                "pool_timeout": self.config.pool_timeout,
                "pool_recycle": self.config.pool_recycle,
                "pool_pre_ping": self.config.pool_pre_ping,
            })

        engine = create_engine(
            self.config.database_url,
            **engine_kwargs
        )

        # Add connection pool event listeners for monitoring
        event.listen(engine, "connect", self._on_connect)
        event.listen(engine, "checkout", self._on_checkout)

        return engine

    @staticmethod
    def _on_connect(dbapi_conn, connection_record):
        """Event listener for new database connections.

        Args:
            dbapi_conn: Database API connection
            connection_record: Connection record
        """
        logger.debug("New database connection established")

    @staticmethod
    def _on_checkout(dbapi_conn, connection_record, connection_proxy):
        """Event listener for connection checkout from pool.

        Args:
            dbapi_conn: Database API connection
            connection_record: Connection record
            connection_proxy: Connection proxy
        """
        logger.debug("Connection checked out from pool")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic commit/rollback.

        Context manager that yields a database session and automatically
        commits on success or rolls back on exception.

        Yields:
            Database session

        Example:
            >>> db = DatabaseConnection()
            >>> with db.get_session() as session:
            ...     doc = Document(filename="test.txt")
            ...     session.add(doc)
            ...     # Automatically commits on exit
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
            logger.debug("Session committed successfully")
        except Exception as e:
            session.rollback()
            logger.error("Session rolled back due to error: %s", str(e))
            raise
        finally:
            session.close()
            logger.debug("Session closed")

    def health_check(self) -> bool:
        """Check database connectivity and health.

        Returns:
            True if database is accessible, False otherwise

        Example:
            >>> db = DatabaseConnection()
            >>> if db.health_check():
            ...     print("Database is healthy")
        """
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            logger.info("Database health check passed")
            return True
        except Exception as e:
            logger.error("Database health check failed: %s", str(e))
            return False

    def create_tables(self) -> None:
        """Create all database tables.

        Note: For production, use Alembic migrations instead.
        This method is primarily for testing and development.
        """
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error("Failed to create database tables: %s", str(e))
            raise

    def drop_tables(self) -> None:
        """Drop all database tables.

        Warning: This will delete all data. Use with caution.
        Primarily for testing environments.
        """
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error("Failed to drop database tables: %s", str(e))
            raise

    def get_pool_status(self) -> dict:
        """Get connection pool status metrics.

        Returns:
            Dictionary with pool status information

        Example:
            >>> db = DatabaseConnection()
            >>> status = db.get_pool_status()
            >>> print(f"Active connections: {status['checked_out']}")
        """
        pool = self.engine.pool
        return {
            "size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "checked_in": pool.checkedin(),
        }

    def close(self) -> None:
        """Close all connections and dispose of the engine.

        Should be called when shutting down the application.
        """
        try:
            self.engine.dispose()
            logger.info("Database connections closed and engine disposed")
        except Exception as e:
            logger.error("Error closing database connections: %s", str(e))
            raise

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
