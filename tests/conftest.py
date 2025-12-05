"""Pytest configuration and fixtures for RAG Factory tests."""

import os
import pytest
from typing import Generator

from rag_factory.database.config import DatabaseConfig
from rag_factory.database.connection import DatabaseConnection
from rag_factory.database.models import Base


@pytest.fixture(scope="session")
def test_db_config() -> DatabaseConfig:
    """Create test database configuration.

    Uses environment variable DB_DATABASE_URL if set,
    otherwise defaults to in-memory SQLite for unit tests.
    """
    db_url = os.environ.get(
        "DB_DATABASE_URL",
        "sqlite:///:memory:"
    )

    return DatabaseConfig(
        database_url=db_url,
        pool_size=5,
        max_overflow=10,
        echo=False
    )


@pytest.fixture(scope="function")
def db_connection(test_db_config: DatabaseConfig) -> Generator[DatabaseConnection, None, None]:
    """Create a database connection for testing.

    Creates tables before test and drops them after.
    """
    db = DatabaseConnection(test_db_config)

    # Create all tables
    Base.metadata.create_all(bind=db.engine)

    yield db

    # Drop all tables
    Base.metadata.drop_all(bind=db.engine)
    db.close()


@pytest.fixture(scope="function")
def db_session(db_connection: DatabaseConnection):
    """Create a database session for testing.

    Automatically rolls back after each test.
    """
    with db_connection.get_session() as session:
        yield session
