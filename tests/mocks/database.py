"""Database mock builders for testing.

This module provides builders for creating mock database-related objects
including engines, connections, and validators.
"""

from typing import Optional, Any, List, Dict
from unittest.mock import Mock, AsyncMock, patch
from contextlib import contextmanager

from .builders import create_mock_with_context_manager, create_async_context_manager_mock


def create_mock_engine() -> Mock:
    """Create a mock SQLAlchemy engine.
    
    Returns:
        Mock engine with connection support
        
    Example:
        >>> engine = create_mock_engine()
        >>> with engine.connect() as conn:
        ...     # conn is a mock connection
        ...     pass
    """
    engine = Mock()
    
    # Create mock connection
    connection = create_mock_connection()
    engine.connect = Mock(return_value=connection)
    engine.dispose = Mock()
    engine.execute = Mock()
    
    return engine


def create_mock_connection() -> Mock:
    """Create a mock database connection.
    
    Returns:
        Mock connection that works as context manager
        
    Example:
        >>> conn = create_mock_connection()
        >>> with conn as c:
        ...     c.execute("SELECT 1")
    """
    connection = Mock()
    
    # Make it work as context manager
    connection.__enter__ = Mock(return_value=connection)
    connection.__exit__ = Mock(return_value=None)
    
    # Add common methods
    connection.execute = Mock()
    connection.commit = Mock()
    connection.rollback = Mock()
    connection.close = Mock()
    
    # Mock result
    result = Mock()
    result.fetchall = Mock(return_value=[])
    result.fetchone = Mock(return_value=None)
    connection.execute.return_value = result
    
    return connection


def create_mock_session() -> Mock:
    """Create a mock SQLAlchemy session.
    
    Returns:
        Mock session with common ORM methods
        
    Example:
        >>> session = create_mock_session()
        >>> session.add(obj)
        >>> session.commit()
    """
    session = Mock()
    
    # Context manager support
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=None)
    
    # Common methods
    session.add = Mock()
    session.delete = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.flush = Mock()
    session.close = Mock()
    session.query = Mock()
    session.execute = Mock()
    
    # Mock query result
    query_result = Mock()
    query_result.all = Mock(return_value=[])
    query_result.first = Mock(return_value=None)
    query_result.count = Mock(return_value=0)
    session.query.return_value = query_result
    
    return session


def create_mock_migration_validator(
    is_valid: bool = True,
    missing_migrations: Optional[List[str]] = None
) -> Mock:
    """Create a mock migration validator.
    
    Args:
        is_valid: Whether validation should pass (default: True)
        missing_migrations: List of missing migration names
        
    Returns:
        Mock migration validator
        
    Example:
        >>> validator = create_mock_migration_validator(is_valid=True)
        >>> is_valid, missing = validator.validate()
        >>> assert is_valid == True
    """
    validator = Mock()
    
    if missing_migrations is None:
        missing_migrations = []
    
    # Sync methods
    validator.validate = Mock(return_value=(is_valid, missing_migrations))
    validator.get_applied_migrations = Mock(return_value=[])
    validator.get_pending_migrations = Mock(return_value=missing_migrations)
    
    return validator


def create_mock_async_pool() -> Mock:
    """Create a mock async connection pool (asyncpg).
    
    Returns:
        Mock connection pool
        
    Example:
        >>> pool = create_mock_async_pool()
        >>> async with pool.acquire() as conn:
        ...     await conn.execute("SELECT 1")
    """
    pool = Mock()
    
    # Create mock connection
    conn = Mock()
    conn.execute = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=None)
    conn.close = AsyncMock()
    
    # Create async context manager for acquire
    class AsyncAcquireContext:
        async def __aenter__(self):
            return conn
        async def __aexit__(self, *args):
            pass
    
    pool.acquire = Mock(return_value=AsyncAcquireContext())
    pool.close = AsyncMock()
    
    return pool


def create_mock_database_context(
    table_name: str = "chunks",
    field_mapping: Optional[Dict[str, str]] = None
) -> Mock:
    """Create a mock DatabaseContext.
    
    Args:
        table_name: Name of the table
        field_mapping: Field name mappings
        
    Returns:
        Mock DatabaseContext
        
    Example:
        >>> ctx = create_mock_database_context(table_name="custom_chunks")
        >>> await ctx.store_chunks(chunks)
    """
    context = Mock()
    
    if field_mapping is None:
        field_mapping = {
            "id": "id",
            "text": "text",
            "embedding": "embedding",
            "metadata": "metadata"
        }
    
    # Async methods
    context.store_chunks = AsyncMock()
    context.search_chunks = AsyncMock(return_value=[])
    context.get_chunks = AsyncMock(return_value=[])
    context.delete_chunks = AsyncMock()
    context.close = AsyncMock()
    
    # Sync methods
    context.get_table_name = Mock(return_value=table_name)
    context.get_field_mapping = Mock(return_value=field_mapping)
    
    # Properties
    context.table_name = table_name
    context.field_mapping = field_mapping
    
    return context


@contextmanager
def mock_database_transaction():
    """Context manager for mocking database transactions.
    
    Yields:
        Mock transaction object
        
    Example:
        >>> with mock_database_transaction() as tx:
        ...     # Perform operations
        ...     pass
        >>> # Transaction auto-committed
    """
    transaction = Mock()
    transaction.commit = Mock()
    transaction.rollback = Mock()
    
    try:
        yield transaction
        transaction.commit()
    except Exception:
        transaction.rollback()
        raise


def create_mock_alembic_config() -> Mock:
    """Create a mock Alembic configuration.
    
    Returns:
        Mock Alembic config
        
    Example:
        >>> config = create_mock_alembic_config()
        >>> config.get_main_option("sqlalchemy.url")
    """
    config = Mock()
    
    config.get_main_option = Mock(return_value="postgresql://localhost/test")
    config.set_main_option = Mock()
    config.attributes = {}
    
    return config


def create_mock_postgres_service(
    host: str = "localhost",
    port: int = 5432,
    database: str = "test_db",
    user: str = "test_user"
) -> Mock:
    """Create a mock PostgreSQL database service.
    
    This is a convenience wrapper around create_mock_database_service
    with PostgreSQL-specific defaults.
    
    Args:
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        
    Returns:
        Mock PostgreSQL service
    """
    from .services import create_mock_database_service
    
    service = create_mock_database_service()
    
    # Add PostgreSQL-specific attributes
    service.host = host
    service.port = port
    service.database = database
    service.user = user
    service.connection_string = f"postgresql://{user}@{host}:{port}/{database}"
    
    return service


def create_mock_neo4j_driver() -> Mock:
    """Create a mock Neo4j driver.
    
    Returns:
        Mock Neo4j driver
        
    Example:
        >>> driver = create_mock_neo4j_driver()
        >>> async with driver.session() as session:
        ...     await session.run("MATCH (n) RETURN n")
    """
    driver = Mock()
    
    # Create mock session
    session = Mock()
    session.run = AsyncMock(return_value=[])
    session.close = AsyncMock()
    
    # Make session work as async context manager
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    
    # Driver methods
    driver.session = Mock(return_value=session)
    driver.close = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    
    return driver
