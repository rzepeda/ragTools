# Story 16.3: Create Database Connection Fixtures

**Story ID:** 16.3  
**Epic:** Epic 16 - Database Migration System Consolidation  
**Story Points:** 5  
**Priority:** High  
**Dependencies:** Story 16.2 (Environment variables standardized)

---

## User Story

**As a** developer  
**I want** proper pytest fixtures for database connections  
**So that** all 57 failing database tests can pass

---

## Detailed Requirements

### Functional Requirements

> [!IMPORTANT]
> **Current Problem**: 57 database tests failing with "fixture 'db_connection' not found"
> 
> **Root Cause**: Missing `db_connection` fixture in `tests/conftest.py`
> 
> **Solution**: Create comprehensive database fixtures for all test scenarios

1. **Create Core Fixtures**
   - `test_db_url` - Reads `TEST_DATABASE_URL` from environment
   - `db_engine` - Creates SQLAlchemy engine for tests
   - `db_connection` - Provides database connection with transaction rollback
   - `db_service` - Provides async database service for integration tests

2. **Implement Transaction Isolation**
   - Each test runs in a transaction
   - Automatic rollback after each test
   - No test data persists between tests
   - Parallel test execution support

3. **Support Multiple Test Types**
   - Unit tests (mocked database)
   - Integration tests (real database)
   - Async tests (async database service)
   - Sync tests (sync database connection)

4. **Add Test Database Management**
   - Automatic test database creation
   - Schema setup before tests
   - Schema teardown after tests
   - Connection pool management

### Non-Functional Requirements

1. **Performance**
   - Fast test execution
   - Efficient connection pooling
   - Minimal setup/teardown overhead

2. **Reliability**
   - Proper cleanup even on test failure
   - No database state leakage between tests
   - Clear error messages for configuration issues

3. **Usability**
   - Simple fixture usage in tests
   - Clear documentation
   - Examples for common patterns

---

## Acceptance Criteria

### AC1: Core Fixtures Created
- [ ] `test_db_url` fixture in `tests/conftest.py`
- [ ] `db_engine` fixture with session scope
- [ ] `db_connection` fixture with function scope
- [ ] `db_service` fixture for async tests
- [ ] All fixtures properly documented with docstrings

### AC2: Transaction Isolation
- [ ] Each test runs in isolated transaction
- [ ] Automatic rollback after test completion
- [ ] Rollback works even on test failure
- [ ] No data persists between tests

### AC3: Test Database Management
- [ ] Schema created automatically before tests
- [ ] Schema dropped automatically after tests
- [ ] Connection pool properly managed
- [ ] Cleanup works even on test suite interruption

### AC4: Integration with Existing Tests
- [ ] All 57 failing database tests now pass
- [ ] No changes needed to test function signatures
- [ ] Tests can use fixtures via dependency injection
- [ ] Both sync and async tests supported

### AC5: Documentation
- [ ] `tests/README.md` created with fixture documentation
- [ ] Usage examples for each fixture
- [ ] Setup instructions for test database
- [ ] Troubleshooting guide

### AC6: Test Quality
- [ ] All database tests pass (100% success rate)
- [ ] Type hints validated
- [ ] Linting passes
- [ ] No fixture-related warnings

---

## Technical Specifications

### Fixture Implementation

```python
# tests/conftest.py

import pytest
import os
import asyncio
from typing import Generator, AsyncGenerator
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from urllib.parse import urlparse

from rag_factory.database.models import Base
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
        pytest.skip: If TEST_DATABASE_URL not set
    """
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip(
            "TEST_DATABASE_URL not set. "
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
def db_connection(db_engine) -> Generator[Session, None, None]:
    """
    Provide database connection with transaction rollback.
    
    Each test runs in an isolated transaction that is rolled back
    after the test completes. This ensures no test data persists.
    
    Args:
        db_engine: SQLAlchemy engine from session fixture
        
    Yields:
        SQLAlchemy Session with transaction isolation
        
    Example:
        ```python
        def test_insert_document(db_connection):
            doc = Document(filename="test.txt")
            db_connection.add(doc)
            db_connection.commit()
            
            # Query to verify
            result = db_connection.query(Document).first()
            assert result.filename == "test.txt"
            # Transaction will be rolled back after test
        ```
    """
    # Create connection
    connection = db_engine.connect()
    
    # Begin transaction
    transaction = connection.begin()
    
    # Create session bound to connection
    Session = sessionmaker(bind=connection)
    session = Session()
    
    # Yield session to test
    yield session
    
    # Cleanup: close session, rollback transaction, close connection
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def db_session(db_connection) -> Session:
    """
    Alias for db_connection for backward compatibility.
    
    Some tests may use 'db_session' instead of 'db_connection'.
    
    Args:
        db_connection: Database connection fixture
        
    Yields:
        SQLAlchemy Session
    """
    yield db_connection


# =============================================================================
# Database Service Fixtures (Async)
# =============================================================================

@pytest.fixture(scope="function")
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
    
    # Create service
    service = PostgresqlDatabaseService(
        host=parsed.hostname or 'localhost',
        port=parsed.port or 5432,
        database=parsed.path.lstrip('/'),
        user=parsed.username or 'postgres',
        password=parsed.password or ''
    )
    
    # Initialize connection pool
    await service.initialize()
    
    yield service
    
    # Cleanup: close all connections
    await service.close()


@pytest.fixture(scope="function")
def db_config(test_db_url: str):
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
    from rag_factory.database.config import DatabaseConfig
    
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
    skip_db = pytest.mark.skip(reason="TEST_DATABASE_URL not set")
    
    # Check if database URL is set
    if not os.getenv("TEST_DATABASE_URL"):
        for item in items:
            # Skip tests marked with @pytest.mark.database
            if "database" in item.keywords:
                item.add_marker(skip_db)
```

### Test Database Setup Script

```python
# tests/setup_test_db.py

"""
Script to set up test database.

Usage:
    python tests/setup_test_db.py
"""

import os
import sys
from sqlalchemy import create_engine, text
from urllib.parse import urlparse

def setup_test_database():
    """Create test database and install extensions."""
    
    # Get test database URL
    test_db_url = os.getenv("TEST_DATABASE_URL")
    if not test_db_url:
        print("ERROR: TEST_DATABASE_URL not set")
        sys.exit(1)
    
    # Parse URL
    parsed = urlparse(test_db_url)
    db_name = parsed.path.lstrip('/')
    
    # Create URL for postgres database (to create test db)
    postgres_url = test_db_url.replace(f"/{db_name}", "/postgres")
    
    print(f"Setting up test database: {db_name}")
    
    # Connect to postgres database
    engine = create_engine(postgres_url, isolation_level="AUTOCOMMIT")
    
    with engine.connect() as conn:
        # Drop existing test database if exists
        print(f"Dropping existing database {db_name} (if exists)...")
        conn.execute(text(f"DROP DATABASE IF EXISTS {db_name}"))
        
        # Create test database
        print(f"Creating database {db_name}...")
        conn.execute(text(f"CREATE DATABASE {db_name}"))
    
    engine.dispose()
    
    # Connect to test database and install extensions
    test_engine = create_engine(test_db_url)
    
    with test_engine.connect() as conn:
        # Install pgvector extension
        print("Installing pgvector extension...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    
    test_engine.dispose()
    
    print("✅ Test database setup complete!")

if __name__ == "__main__":
    setup_test_database()
```

### Usage Examples

```python
# Example 1: Simple unit test with db_connection
def test_create_document(db_connection):
    """Test creating a document."""
    from rag_factory.database.models import Document
    
    doc = Document(
        filename="test.txt",
        source_path="/path/to/test.txt",
        content_hash="abc123"
    )
    
    db_connection.add(doc)
    db_connection.commit()
    
    # Verify
    result = db_connection.query(Document).filter_by(filename="test.txt").first()
    assert result is not None
    assert result.content_hash == "abc123"


# Example 2: Async integration test with db_service
@pytest.mark.asyncio
async def test_vector_search(db_service):
    """Test vector similarity search."""
    # Store chunks with embeddings
    chunks = [
        {
            "id": "chunk_1",
            "text": "Python programming",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {}
        },
        {
            "id": "chunk_2",
            "text": "JavaScript programming",
            "embedding": [0.2, 0.3, 0.4],
            "metadata": {}
        }
    ]
    
    await db_service.store_chunks(chunks)
    
    # Search
    query_embedding = [0.1, 0.2, 0.3]
    results = await db_service.search_chunks(query_embedding, top_k=1)
    
    assert len(results) == 1
    assert results[0]['chunk_id'] == 'chunk_1'


# Example 3: Test with clean database
def test_empty_database(clean_database):
    """Test with guaranteed empty database."""
    from rag_factory.database.models import Document, Chunk
    
    # Verify database is empty
    assert clean_database.query(Document).count() == 0
    assert clean_database.query(Chunk).count() == 0
    
    # Add test data
    doc = Document(filename="test.txt", source_path="/test", content_hash="123")
    clean_database.add(doc)
    clean_database.commit()
    
    # Verify
    assert clean_database.query(Document).count() == 1
```

### Documentation

Create `tests/README.md`:

```markdown
# Test Suite Documentation

## Running Tests

### All Tests
```bash
pytest tests/ -v
```

### Database Tests Only
```bash
pytest tests/unit/database/ tests/integration/database/ -v
```

### Skip Database Tests
```bash
pytest tests/ -v -m "not database"
```

## Database Test Setup

### 1. Set Environment Variable

```bash
export TEST_DATABASE_URL="postgresql://user:password@localhost:5432/rag_test"
```

### 2. Create Test Database

```bash
python tests/setup_test_db.py
```

### 3. Run Tests

```bash
pytest tests/unit/database/ -v
```

## Available Fixtures

### `test_db_url`
- **Scope**: session
- **Returns**: Test database URL from environment
- **Usage**: Automatically used by other fixtures

### `db_engine`
- **Scope**: session
- **Returns**: SQLAlchemy Engine
- **Usage**: Creates tables once per test session

### `db_connection`
- **Scope**: function
- **Returns**: SQLAlchemy Session with transaction rollback
- **Usage**: For sync database tests

### `db_service`
- **Scope**: function
- **Returns**: PostgresqlDatabaseService (async)
- **Usage**: For async integration tests

### `clean_database`
- **Scope**: function
- **Returns**: Database connection with all tables empty
- **Usage**: For tests needing empty database

## Troubleshooting

### "TEST_DATABASE_URL not set"
Set the environment variable:
```bash
export TEST_DATABASE_URL="postgresql://localhost/rag_test"
```

### "database does not exist"
Create the test database:
```bash
python tests/setup_test_db.py
```

### "pgvector extension not found"
Install pgvector:
```bash
# On Ubuntu/Debian
sudo apt-get install postgresql-15-pgvector

# On macOS
brew install pgvector
```
```

---

## Testing Strategy

### Fixture Tests

```python
# tests/unit/conftest/test_database_fixtures.py

import pytest
from sqlalchemy.orm import Session
from rag_factory.services.database.postgres import PostgresqlDatabaseService

class TestDatabaseFixtures:
    """Test database fixtures work correctly."""
    
    def test_test_db_url_fixture(self, test_db_url):
        """Test test_db_url fixture returns valid URL."""
        assert test_db_url.startswith("postgresql://")
    
    def test_db_connection_fixture(self, db_connection):
        """Test db_connection fixture provides session."""
        assert isinstance(db_connection, Session)
    
    def test_transaction_rollback(self, db_connection):
        """Test transactions are rolled back."""
        from rag_factory.database.models import Document
        
        # Add document
        doc = Document(filename="test.txt", source_path="/test", content_hash="123")
        db_connection.add(doc)
        db_connection.commit()
        
        # Verify it exists
        count = db_connection.query(Document).count()
        assert count == 1
        
        # After test, it should be rolled back (verified by next test)
    
    def test_transaction_isolation(self, db_connection):
        """Test previous test data was rolled back."""
        from rag_factory.database.models import Document
        
        # Should be empty (previous test rolled back)
        count = db_connection.query(Document).count()
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_db_service_fixture(self, db_service):
        """Test db_service fixture provides service."""
        assert isinstance(db_service, PostgresqlDatabaseService)
        
        # Test basic operation
        chunks = [{"id": "1", "text": "test", "embedding": [0.1]}]
        await db_service.store_chunks(chunks)
```

### Integration Tests

```bash
# Run all database tests
pytest tests/unit/database/ tests/integration/database/ -v

# Verify 57 previously failing tests now pass
pytest tests/integration/database/test_database_integration.py -v
pytest tests/integration/repositories/test_repository_integration.py -v
pytest tests/unit/database/test_connection.py -v
pytest tests/unit/database/test_models.py -v
```

---

## Definition of Done

- [ ] All fixtures implemented in `tests/conftest.py`
- [ ] Transaction isolation working correctly
- [ ] Test database setup script created
- [ ] `tests/README.md` documentation created
- [ ] All 57 failing database tests now pass
- [ ] Fixture tests pass
- [ ] Type checking passes
- [ ] Linting passes
- [ ] No fixture-related warnings
- [ ] PR approved and merged

---

## Notes

- **Transaction rollback** is critical for test isolation
- **Session scope** for engine improves performance
- **Function scope** for connection ensures isolation
- **Async support** needed for integration tests
- This story directly fixes the 57 failing database tests
- Fixtures are reusable across all database tests

---

## Success Metrics

- **Before**: 57 database tests failing
- **After**: 0 database tests failing
- **Test execution time**: Should be reasonable (< 30 seconds for all DB tests)
- **Test isolation**: No data leakage between tests
- **Developer experience**: Easy to write new database tests
