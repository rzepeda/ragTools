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

### Run Specific Test File
```bash
pytest tests/unit/database/test_connection.py -v
```

---

## Database Test Setup

### Prerequisites

1. **PostgreSQL with pgvector**
   - PostgreSQL 12+ installed and running
   - pgvector extension installed

2. **Environment Variable**
   - `TEST_DATABASE_URL` must be set in your environment

### Setup Steps

#### 1. Set Environment Variable

Add to your `.env` file or export directly:

```bash
export TEST_DATABASE_URL="postgresql://rag_user:rag_password@localhost:5432/rag_test"
```

Or if using the VM setup with host IP:
```bash
export TEST_DATABASE_URL="postgresql://rag_user:rag_password@192.168.56.1:5432/rag_test"
```

#### 2. Create Test Database

Run the setup script:

```bash
python tests/setup_test_db.py
```

This script will:
- Drop existing test database (if exists)
- Create new test database
- Install pgvector extension
- Verify setup

#### 3. Run Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all database tests
pytest tests/unit/database/ tests/integration/database/ -v
```

---

## Available Fixtures

### `test_db_url`
- **Scope**: session
- **Returns**: Test database URL from `TEST_DATABASE_URL` environment variable
- **Usage**: Automatically used by other fixtures
- **Skips**: Test if `TEST_DATABASE_URL` not set

### `db_engine`
- **Scope**: session
- **Returns**: SQLAlchemy Engine
- **Usage**: Creates all tables once at session start, drops them at session end
- **Performance**: Session scope ensures tables are created only once

### `db_connection`
- **Scope**: function
- **Returns**: SQLAlchemy Session with transaction rollback
- **Usage**: For synchronous database tests
- **Isolation**: Each test runs in a transaction that is automatically rolled back

**Example:**
```python
def test_create_document(db_connection):
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
    # Transaction automatically rolled back after test
```

### `db_session`
- **Scope**: function
- **Returns**: SQLAlchemy Session (alias for `db_connection`)
- **Usage**: Backward compatibility for tests using `db_session` name

### `test_db_config`
- **Scope**: function
- **Returns**: DatabaseConfig instance
- **Usage**: For tests that need database configuration object

**Example:**
```python
def test_database_config(test_db_config):
    assert test_db_config.database_url.startswith("postgresql://")
```

### `db_service`
- **Scope**: function
- **Returns**: PostgresqlDatabaseService (async)
- **Usage**: For async integration tests
- **Cleanup**: Automatically closes connection pool after test

**Example:**
```python
@pytest.mark.asyncio
async def test_vector_search(db_service):
    # Store chunks with embeddings
    chunks = [
        {
            "id": "chunk_1",
            "text": "Python programming",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {}
        }
    ]
    
    await db_service.store_chunks(chunks)
    
    # Search
    query_embedding = [0.1, 0.2, 0.3]
    results = await db_service.search_chunks(query_embedding, top_k=1)
    
    assert len(results) == 1
    assert results[0]['chunk_id'] == 'chunk_1'
```

### `db_config`
- **Scope**: function
- **Returns**: DatabaseConfig instance
- **Usage**: Alternative name for database configuration

### `clean_database`
- **Scope**: function
- **Returns**: Database connection with all tables empty
- **Usage**: For tests that need guaranteed empty database
- **Cleanup**: Deletes all data before and after test

**Example:**
```python
def test_empty_database(clean_database):
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
    # Data will be cleaned up after test
```

---

## Test Markers

### `@pytest.mark.database`
Mark tests that require database connection. These tests will be automatically skipped if `TEST_DATABASE_URL` is not set.

```python
@pytest.mark.database
def test_database_operation(db_connection):
    # Test code here
    pass
```

### `@pytest.mark.integration`
Mark integration tests.

```python
@pytest.mark.integration
def test_full_workflow(db_connection):
    # Integration test code
    pass
```

---

## Troubleshooting

### "TEST_DATABASE_URL not set"

**Problem**: Tests are being skipped with this message.

**Solution**: Set the environment variable:
```bash
export TEST_DATABASE_URL="postgresql://user:password@localhost:5432/rag_test"
```

Or add to your `.env` file.

---

### "database does not exist"

**Problem**: PostgreSQL reports database doesn't exist.

**Solution**: Create the test database:
```bash
python tests/setup_test_db.py
```

---

### "pgvector extension not found"

**Problem**: Tests fail with pgvector-related errors.

**Solution**: Install pgvector extension:

**On Ubuntu/Debian:**
```bash
sudo apt-get install postgresql-15-pgvector
```

**On macOS:**
```bash
brew install pgvector
```

**Manual Installation:**
```bash
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

Then run the setup script again:
```bash
python tests/setup_test_db.py
```

---

### "connection refused"

**Problem**: Cannot connect to PostgreSQL.

**Solution**: 
1. Ensure PostgreSQL is running:
   ```bash
   sudo systemctl status postgresql
   # or
   brew services list
   ```

2. Check connection parameters in `TEST_DATABASE_URL`

3. If using VM setup, verify host IP is correct:
   ```bash
   ./find-host-ip.sh
   ```

4. Test connection manually:
   ```bash
   psql $TEST_DATABASE_URL
   ```

---

### "permission denied to create database"

**Problem**: User doesn't have CREATE DATABASE privileges.

**Solution**: Grant privileges:
```sql
-- Connect as postgres superuser
psql -U postgres

-- Grant privileges
ALTER USER rag_user CREATEDB;
```

---

### Tests hang or timeout

**Problem**: Tests don't complete.

**Solution**:
1. Check for connection pool issues
2. Ensure `TEST_DATABASE_URL` points to test database, not production
3. Check PostgreSQL logs for errors
4. Try running single test to isolate issue:
   ```bash
   pytest tests/unit/database/test_connection.py::TestDatabaseConnection::test_health_check_success -v
   ```

---

## Best Practices

### 1. Use Transaction Isolation

Always use `db_connection` fixture for tests. It provides automatic transaction rollback:

```python
def test_something(db_connection):
    # Changes are automatically rolled back
    pass
```

### 2. Don't Commit in Tests (Usually)

The transaction rollback happens automatically. Explicit commits are usually not needed:

```python
def test_create_record(db_connection):
    record = MyModel(name="test")
    db_connection.add(record)
    # No need to commit - rollback happens automatically
```

### 3. Use `clean_database` for Empty State

If you need guaranteed empty database:

```python
def test_with_clean_slate(clean_database):
    # Database is empty
    assert clean_database.query(MyModel).count() == 0
```

### 4. Mark Database Tests

Always mark tests that require database:

```python
@pytest.mark.database
def test_database_feature(db_connection):
    pass
```

### 5. Async Tests

Use `db_service` for async operations:

```python
@pytest.mark.asyncio
async def test_async_operation(db_service):
    await db_service.store_chunks([...])
```

---

## Performance Tips

1. **Session-scoped fixtures**: `db_engine` creates tables once per session
2. **Transaction rollback**: Faster than deleting data
3. **Connection pooling**: Reuses connections efficiently
4. **Parallel execution**: Tests are isolated and can run in parallel

---

## Examples

### Basic CRUD Test

```python
def test_crud_operations(db_connection):
    from rag_factory.database.models import Document
    
    # Create
    doc = Document(filename="test.txt", source_path="/test", content_hash="123")
    db_connection.add(doc)
    db_connection.flush()
    
    # Read
    result = db_connection.query(Document).filter_by(filename="test.txt").first()
    assert result is not None
    
    # Update
    result.status = "completed"
    db_connection.flush()
    
    # Delete
    db_connection.delete(result)
    db_connection.flush()
    
    # Verify
    assert db_connection.query(Document).count() == 0
```

### Async Integration Test

```python
@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_pipeline(db_service):
    # Store documents
    chunks = [
        {"id": "1", "text": "Hello", "embedding": [0.1, 0.2]},
        {"id": "2", "text": "World", "embedding": [0.3, 0.4]}
    ]
    await db_service.store_chunks(chunks)
    
    # Search
    results = await db_service.search_chunks([0.1, 0.2], top_k=1)
    assert len(results) == 1
    assert results[0]['chunk_id'] == '1'
```

### Test with Clean Database

```python
def test_import_data(clean_database):
    from rag_factory.database.models import Document
    
    # Database is guaranteed empty
    assert clean_database.query(Document).count() == 0
    
    # Import data
    docs = [
        Document(filename=f"doc{i}.txt", source_path=f"/test/{i}", content_hash=f"hash{i}")
        for i in range(10)
    ]
    clean_database.add_all(docs)
    clean_database.commit()
    
    # Verify
    assert clean_database.query(Document).count() == 10
```
