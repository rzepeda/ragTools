import sys
import importlib.util
from unittest.mock import MagicMock
import pytest

# Mock numpy and other dependencies GLOBALLY before any test collection
try:
    import numpy
except ImportError:
    sys.modules["numpy"] = MagicMock()

# Mock the services package and submodules that cause issues
# sys.modules["rag_factory.services"] = MagicMock()
# sys.modules["rag_factory.services.onnx"] = MagicMock()
# sys.modules["rag_factory.services.onnx.embedding"] = MagicMock()
# sys.modules["rag_factory.services.embedding"] = MagicMock()
# sys.modules["rag_factory.services.embedding.providers"] = MagicMock()
# sys.modules["rag_factory.services.embedding.providers.onnx_local"] = MagicMock()
# sys.modules["rag_factory.services.embedding.service"] = MagicMock()
# sys.modules["rag_factory.services.api"] = MagicMock()
# sys.modules["rag_factory.services.database"] = MagicMock()
# sys.modules["rag_factory.services.local"] = MagicMock()

# Manually load the modules we actually need
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Mock rag_factory package to prevent init
# sys.modules["rag_factory"] = MagicMock()
# sys.modules["rag_factory.strategies"] = MagicMock()

# Load modules in dependency order to avoid cycles and init triggers
load_module("rag_factory.core.capabilities", "/mnt/MCPProyects/ragTools/rag_factory/core/capabilities.py")
load_module("rag_factory.services.interfaces", "/mnt/MCPProyects/ragTools/rag_factory/services/interfaces.py")
load_module("rag_factory.services.dependencies", "/mnt/MCPProyects/ragTools/rag_factory/services/dependencies.py")
load_module("rag_factory.services.consistency", "/mnt/MCPProyects/ragTools/rag_factory/services/consistency.py")
load_module("rag_factory.core.indexing_interface", "/mnt/MCPProyects/ragTools/rag_factory/core/indexing_interface.py")
load_module("rag_factory.core.retrieval_interface", "/mnt/MCPProyects/ragTools/rag_factory/core/retrieval_interface.py")
load_module("rag_factory.core.pipeline", "/mnt/MCPProyects/ragTools/rag_factory/core/pipeline.py")
load_module("rag_factory.strategies.base", "/mnt/MCPProyects/ragTools/rag_factory/strategies/base.py")
load_module("rag_factory.exceptions", "/mnt/MCPProyects/ragTools/rag_factory/exceptions.py")
load_module("rag_factory.factory", "/mnt/MCPProyects/ragTools/rag_factory/factory.py")
load_module("rag_factory.strategies.indexing", "/mnt/MCPProyects/ragTools/rag_factory/strategies/indexing/__init__.py")
load_module("rag_factory.strategies.indexing.context_aware", "/mnt/MCPProyects/ragTools/rag_factory/strategies/indexing/context_aware.py")

# Load CLI modules for CLI tests
load_module("rag_factory.cli", "/mnt/MCPProyects/ragTools/rag_factory/cli/__init__.py")
load_module("rag_factory.cli.formatters", "/mnt/MCPProyects/ragTools/rag_factory/cli/formatters/__init__.py")
load_module("rag_factory.cli.formatters.validation", "/mnt/MCPProyects/ragTools/rag_factory/cli/formatters/validation.py")
load_module("rag_factory.cli.utils", "/mnt/MCPProyects/ragTools/rag_factory/cli/utils/__init__.py")
load_module("rag_factory.cli.utils.validation", "/mnt/MCPProyects/ragTools/rag_factory/cli/utils/validation.py")
load_module("rag_factory.cli.commands", "/mnt/MCPProyects/ragTools/rag_factory/cli/commands/__init__.py")
load_module("rag_factory.cli.commands.validate_pipeline", "/mnt/MCPProyects/ragTools/rag_factory/cli/commands/validate_pipeline.py")
load_module("rag_factory.cli.main", "/mnt/MCPProyects/ragTools/rag_factory/cli/main.py")


# =============================================================================
# Database Fixtures
# =============================================================================

import os
import asyncio
from typing import Generator, AsyncGenerator
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from urllib.parse import urlparse

from rag_factory.database.models import Base
from rag_factory.database.config import DatabaseConfig
from rag_factory.database.connection import DatabaseConnection
from rag_factory.services.database.postgres import PostgresqlDatabaseService


# =============================================================================
# Database URL Fixture
# =============================================================================

@pytest.fixture(scope="session")
def test_db_url() -> str:
    """
    Get test database URL from environment.
    
    Returns:
        Test database URL
        
    Raises:
        pytest.skip: If DB_TEST_DATABASE_URL not set
    """
    url = os.getenv("DB_TEST_DATABASE_URL")
    if not url:
        pytest.skip(
            "DB_TEST_DATABASE_URL not set. "
            "Set this environment variable to run database tests."
        )
    return url


# =============================================================================
# SQLAlchemy Engine Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def db_engine(test_db_url: str):
    """
    Create database engine for testing.
    
    Creates all tables before tests and drops them after.
    Uses session scope for efficiency.
    
    Args:
        test_db_url: Test database URL from environment
        
    Yields:
        SQLAlchemy Engine instance
    """
    # Create engine
    engine = create_engine(
        test_db_url,
        poolclass=StaticPool,  # Use static pool for tests
        echo=False,  # Set to True for SQL debugging
    )
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Drop all tables and dispose engine
    Base.metadata.drop_all(engine)
    engine.dispose()


# =============================================================================
# Database Connection Fixtures (Sync)
# =============================================================================

@pytest.fixture(scope="function")
def db_connection(test_db_config: DatabaseConfig) -> Generator[DatabaseConnection, None, None]:
    """
    Provide database connection with transaction rollback.
    
    Each test runs in an isolated transaction that is rolled back
    after the test completes. This ensures no test data persists.
    
    Args:
        test_db_config: Database configuration from test_db_config fixture
        
    Yields:
        DatabaseConnection instance with transaction isolation
        
    Example:
        ```python
        def test_insert_document(db_connection):
            with db_connection.get_session() as session:
                doc = Document(filename="test.txt")
                session.add(doc)
                session.commit()
                
                # Query to verify
                result = session.query(Document).first()
                assert result.filename == "test.txt"
            # Transaction will be rolled back after test
        ```
    """
    # Create DatabaseConnection instance
    db = DatabaseConnection(test_db_config)
    
    # Create tables
    db.create_tables()
    
    yield db
    
    # Cleanup: drop tables and close connection
    try:
        db.drop_tables()
    except Exception:
        pass  # Tables might already be dropped
    db.close()


@pytest.fixture(scope="function")
def db_session(db_engine) -> Generator[Session, None, None]:
    """
    Provide raw SQLAlchemy Session with transaction rollback.
    
    This fixture is for tests that need direct Session access (like model tests).
    Each test runs in an isolated transaction that is rolled back.
    
    Args:
        db_engine: SQLAlchemy engine from session fixture
        
    Yields:
        SQLAlchemy Session with transaction isolation
        
    Example:
        ```python
        def test_create_document(db_session):
            doc = Document(filename="test.txt")
            db_session.add(doc)
            db_session.flush()
            
            result = db_session.query(Document).first()
            assert result.filename == "test.txt"
            # Transaction automatically rolled back after test
        ```
    """
    # Create connection
    connection = db_engine.connect()
    
    # Begin transaction
    transaction = connection.begin()
    
    # Create session bound to connection
    SessionMaker = sessionmaker(bind=connection)
    session = SessionMaker()
    
    # Yield session to test
    yield session
    
    # Cleanup: close session, rollback transaction, close connection
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def test_db_config(test_db_url: str) -> DatabaseConfig:
    """
    Provide database configuration for tests.
    
    Args:
        test_db_url: Test database URL from environment
        
    Returns:
        DatabaseConfig instance
        
    Example:
        ```python
        def test_database_config(test_db_config):
            assert test_db_config.database_url.startswith("postgresql://")
        ```
    """
    return DatabaseConfig(database_url=test_db_url)


# =============================================================================
# Database Service Fixtures (Async)
# =============================================================================

# Import pytest_asyncio for async fixtures
try:
    import pytest_asyncio
    PYTEST_ASYNCIO_AVAILABLE = True
except ImportError:
    PYTEST_ASYNCIO_AVAILABLE = False

@pytest_asyncio.fixture(scope="function") if PYTEST_ASYNCIO_AVAILABLE else pytest.fixture(scope="function")
async def db_service(test_db_url: str) -> AsyncGenerator[PostgresqlDatabaseService, None]:
    """
    Provide async database service for integration tests.
    
    Creates a PostgresqlDatabaseService instance configured for testing.
    Automatically closes the service after the test.
    
    Args:
        test_db_url: Test database URL from environment
        
    Yields:
        PostgresqlDatabaseService instance
        
    Example:
        ```python
        @pytest.mark.asyncio
        async def test_store_chunks(db_service):
            chunks = [
                {"id": "1", "text": "test", "embedding": [0.1, 0.2]}
            ]
            await db_service.store_chunks(chunks)
            
            results = await db_service.search_chunks([0.1, 0.2], top_k=1)
            assert len(results) == 1
        ```
    """
    # Parse URL for connection parameters
    parsed = urlparse(test_db_url)
    
    # Create service (pool is created lazily on first use)
    service = PostgresqlDatabaseService(
        host=parsed.hostname or 'localhost',
        port=parsed.port or 5432,
        database=parsed.path.lstrip('/'),
        user=parsed.username or 'postgres',
        password=parsed.password or ''
    )
    
    yield service
    
    # Cleanup: close all connections
    await service.close()


@pytest.fixture(scope="function")
def db_config(test_db_url: str) -> DatabaseConfig:
    """
    Provide database configuration for tests.
    
    Args:
        test_db_url: Test database URL from environment
        
    Returns:
        DatabaseConfig instance
        
    Example:
        ```python
        def test_database_config(db_config):
            assert db_config.database_url.startswith("postgresql://")
        ```
    """
    return DatabaseConfig(database_url=test_db_url)


# =============================================================================
# Test Database Utilities
# =============================================================================

@pytest.fixture(scope="function")
def clean_database(db_connection):
    """
    Ensure database is clean before and after test.
    
    Deletes all data from all tables before and after the test.
    Use this for tests that need a completely empty database.
    
    Args:
        db_connection: Database connection fixture
        
    Yields:
        Database connection with clean state
        
    Example:
        ```python
        def test_with_clean_db(clean_database):
            # Database is guaranteed to be empty
            count = clean_database.query(Document).count()
            assert count == 0
        ```
    """
    # Clean before test
    _clean_all_tables(db_connection)
    
    yield db_connection
    
    # Clean after test
    _clean_all_tables(db_connection)


def _clean_all_tables(session: Session) -> None:
    """
    Delete all data from all tables.
    
    Args:
        session: SQLAlchemy session
    """
    # Get all table names in reverse order (to handle foreign keys)
    tables = reversed(Base.metadata.sorted_tables)
    
    for table in tables:
        session.execute(table.delete())
    
    session.commit()


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """
    Configure pytest with custom markers.
    
    Args:
        config: Pytest configuration object
    """
    config.addinivalue_line(
        "markers",
        "database: mark test as requiring database connection"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to skip database tests if not configured.
    
    Args:
        config: Pytest configuration
        items: Collected test items
    """
    skip_db = pytest.mark.skip(reason="DB_TEST_DATABASE_URL not set")
    
    # Check if database URL is set
    if not os.getenv("DB_TEST_DATABASE_URL"):
        for item in items:
            # Skip tests marked with @pytest.mark.database
            if "database" in item.keywords:
                item.add_marker(skip_db)


