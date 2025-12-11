# Vector Database Documentation

This document provides comprehensive information about the PostgreSQL + pgvector database implementation for RAG Factory.

## Overview

The database module provides:
- PostgreSQL database integration with pgvector extension for vector similarity search
- SQLAlchemy ORM models for documents and chunks
- Connection pooling for efficient resource management
- Alembic migrations for schema versioning
- Cross-platform support (PostgreSQL for production, SQLite for testing)

## Architecture

```
rag_factory/database/
├── __init__.py          # Module exports
├── config.py            # Database configuration with Pydantic
├── models.py            # SQLAlchemy ORM models
└── connection.py        # Connection pooling and session management

migrations/
├── versions/
│   └── 001_initial_schema.py  # Initial database schema
├── env.py               # Alembic environment configuration
└── alembic.ini          # Alembic settings

scripts/
├── setup_database.sh    # Local database setup script
└── test_database.sh     # Test database setup script
```

## Database Schema

### Documents Table

Stores metadata about processed documents.

| Column | Type | Description |
|--------|------|-------------|
| document_id | UUID | Primary key |
| filename | VARCHAR(255) | Original filename |
| source_path | TEXT | Path or URL to source |
| content_hash | VARCHAR(64) | SHA-256 hash for deduplication |
| total_chunks | INTEGER | Number of chunks created |
| metadata | JSONB | Flexible metadata storage |
| status | VARCHAR(50) | Processing status (pending, processing, completed, failed) |
| created_at | TIMESTAMP | Creation timestamp |
| updated_at | TIMESTAMP | Last update timestamp (auto-updated) |

**Indexes:**
- Primary key on `document_id`
- B-tree index on `content_hash` for deduplication
- B-tree index on `filename` for lookups
- B-tree index on `status` for filtering

### Chunks Table

Stores text chunks with optional vector embeddings.

| Column | Type | Description |
|--------|------|-------------|
| chunk_id | UUID | Primary key |
| document_id | UUID | Foreign key to documents (CASCADE DELETE) |
| chunk_index | INTEGER | Order within document (0-indexed) |
| text | TEXT | Chunk content |
| embedding | VECTOR(1536) | Vector embedding (nullable) |
| metadata | JSONB | Flexible metadata storage |
| created_at | TIMESTAMP | Creation timestamp |
| updated_at | TIMESTAMP | Last update timestamp (auto-updated) |

**Indexes:**
- Primary key on `chunk_id`
- B-tree index on `document_id` for lookups
- Composite index on `(document_id, chunk_index)` for ordered retrieval
- B-tree index on `created_at` for temporal queries
- HNSW index on `embedding` for vector similarity search

### Vector Index Configuration

The HNSW (Hierarchical Navigable Small World) index is optimized for fast similarity search:

```sql
CREATE INDEX idx_chunks_embedding_hnsw
ON chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
```

**Parameters:**
- `m = 16`: Number of connections per layer (higher = more accurate, slower build)
- `ef_construction = 64`: Size of dynamic candidate list (higher = better quality)
- `vector_cosine_ops`: Cosine distance for semantic similarity

## Configuration

### Environment Variables

> [!IMPORTANT]
> For complete environment variable reference, see [ENVIRONMENT_VARIABLES.md](file:///mnt/MCPProyects/ragTools/docs/database/ENVIRONMENT_VARIABLES.md)

#### Required Variables

**Production/Development:**
```bash
# Main database connection
DATABASE_URL=postgresql://user:password@host:5432/database_name
```

**Testing:**
```bash
# Test database connection (use TEST_DATABASE_URL, not DATABASE_TEST_URL)
TEST_DATABASE_URL=postgresql://user:password@host:5432/test_database
```

#### Configuration Examples

**Local Development:**
```bash
# .env file
DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_factory
TEST_DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_test
```

**VM Development (Accessing Host Services):**
```bash
# .env file
HOST_IP=192.168.56.1
DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_factory
TEST_DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_test
```

**Docker Compose:**
```bash
# .env file
DATABASE_URL=postgresql://rag_user:rag_password@postgres:5432/rag_factory
TEST_DATABASE_URL=postgresql://rag_user:rag_password@postgres:5432/rag_test
```

**Cloud (Neon, Supabase, etc.):**
```bash
# .env file
DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require
TEST_DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/test_db?sslmode=require
```

#### Environment Variable Reference

| Variable | Purpose | Required | Default | Example |
|----------|---------|----------|---------|---------|
| `DATABASE_URL` | Main database connection | Yes | - | `postgresql://localhost/db` |
| `TEST_DATABASE_URL` | Test database connection | For tests | - | `postgresql://localhost/test_db` |
| `DB_DATABASE_URL` | Alternative main connection (with prefix) | No | Uses `DATABASE_URL` | `postgresql://localhost/db` |
| `DB_TEST_DATABASE_URL` | Alternative test connection (with prefix) | No | Uses `TEST_DATABASE_URL` | `postgresql://localhost/test_db` |
| `DB_POOL_SIZE` | Connection pool size | No | `10` | `20` |
| `DB_MAX_OVERFLOW` | Max overflow connections | No | `20` | `40` |
| `DB_POOL_TIMEOUT` | Connection timeout (seconds) | No | `30` | `60` |
| `DB_POOL_RECYCLE` | Recycle connections after (seconds) | No | `3600` | `7200` |
| `DB_ECHO` | Enable SQL query logging | No | `false` | `true` |
| `DB_POOL_PRE_PING` | Test connections before use | No | `true` | `false` |
| `DB_VECTOR_DIMENSIONS` | Embedding dimensions | No | `1536` | `768` |
| `HOST_IP` | VM host machine IP | VM only | - | `192.168.56.1` |

> [!NOTE]
> The `DatabaseConfig` class uses the `DB_` prefix for all configuration variables. Both `DATABASE_URL` and `DB_DATABASE_URL` are supported for backward compatibility.

### Pydantic Configuration

```python
from rag_factory.database.config import DatabaseConfig

# Load from environment
config = DatabaseConfig()

# Or provide explicitly
config = DatabaseConfig(
    database_url="postgresql://user:pass@localhost/ragdb",
    pool_size=10,
    max_overflow=20
)
```

## Usage Examples

### Basic Usage

```python
from rag_factory.database import DatabaseConnection, Document, Chunk
import numpy as np

# Initialize connection
db = DatabaseConnection()

# Create a document
with db.get_session() as session:
    doc = Document(
        filename="example.txt",
        source_path="/path/to/example.txt",
        content_hash="abc123...",
        total_chunks=3,
        status="completed"
    )
    session.add(doc)
    session.flush()

    # Create chunks with embeddings
    for i in range(3):
        chunk = Chunk(
            document_id=doc.document_id,
            chunk_index=i,
            text=f"This is chunk {i} of the document.",
            embedding=np.random.rand(1536).tolist(),
            metadata_={"page": i + 1}
        )
        session.add(chunk)

# Query documents
with db.get_session() as session:
    docs = session.query(Document).filter_by(status="completed").all()
    for doc in docs:
        print(f"Document: {doc.filename}, Chunks: {doc.total_chunks}")
```

### Vector Similarity Search

```python
from sqlalchemy import text

# Your query vector (1536 dimensions)
query_embedding = np.random.rand(1536).tolist()

with db.get_session() as session:
    # Find top 5 most similar chunks using cosine distance
    results = session.execute(
        text("""
            SELECT chunk_id, text,
                   1 - (embedding <=> CAST(:query_vector AS vector)) as similarity
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:query_vector AS vector)
            LIMIT 5
        """),
        {"query_vector": str(query_embedding)}
    ).fetchall()

    for chunk_id, text, similarity in results:
        print(f"Similarity: {similarity:.4f} - {text[:100]}...")
```

### Filtering with Metadata

```python
with db.get_session() as session:
    # Query by JSONB metadata (PostgreSQL only)
    docs = session.query(Document).filter(
        Document.metadata_["category"].astext == "research"
    ).all()

    # Query chunks by document
    chunks = session.query(Chunk).filter_by(
        document_id=doc.document_id
    ).order_by(Chunk.chunk_index).all()
```

### Connection Pool Management

```python
# Check pool status
status = db.get_pool_status()
print(f"Active connections: {status['checked_out']}")
print(f"Available: {status['checked_in']}")

# Health check
if db.health_check():
    print("Database is healthy")

# Close connections when done
db.close()
```

### Context Manager

```python
# Use as context manager for automatic cleanup
with DatabaseConnection() as db:
    with db.get_session() as session:
        # Do work
        pass
# Connections automatically closed
```

## Quick Start

### 1. Install PostgreSQL with pgvector

#### Option A: Docker (Recommended)
```bash
docker run -d \
  --name rag-postgres \
  -e POSTGRES_USER=rag_user \
  -e POSTGRES_PASSWORD=rag_password \
  -e POSTGRES_DB=rag_factory \
  -p 5432:5432 \
  ankane/pgvector
```

#### Option B: Local Installation
```bash
# Ubuntu/Debian
sudo apt-get install postgresql-15 postgresql-15-pgvector

# macOS
brew install postgresql@15 pgvector
```

### 2. Configure Environment

Create `.env` file in project root:
```bash
DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_factory
TEST_DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_test
```

### 3. Run Migrations

```bash
# Upgrade to latest schema
alembic upgrade head
```

### 4. Verify Setup

```bash
# Test connection
python -c "from rag_factory.database.connection import DatabaseConnection; db = DatabaseConnection(); print('✓ Connection successful' if db.health_check() else '✗ Connection failed')"

# Check migration status
alembic current

# Run database tests
pytest tests/unit/database/ -v
```

### Troubleshooting

**Issue: Connection refused**
- Verify PostgreSQL is running: `pg_isready`
- Check DATABASE_URL matches your PostgreSQL configuration
- Ensure PostgreSQL is listening on the correct port

**Issue: pgvector extension not found**
- Install pgvector extension: `sudo apt-get install postgresql-15-pgvector`
- Enable in database: `psql -d rag_factory -c "CREATE EXTENSION vector"`

**Issue: Alembic can't find migrations**
- Ensure you're in the project root directory
- Verify `alembic.ini` exists
- Check `migrations/` directory exists

## Migrations

The project uses [Alembic](https://alembic.sqlalchemy.org/) for database schema migrations.

### Running Migrations

```bash
# Upgrade to latest version
alembic upgrade head

# Upgrade one version
alembic upgrade +1

# Downgrade one version
alembic downgrade -1

# Show current version
alembic current

# Show migration history
alembic history --verbose
```

### Creating Migrations

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "add user table"

# Create empty migration for manual changes
alembic revision -m "add custom index"
```

### Migration Best Practices

1. **Review Auto-Generated Migrations**
   - Always review migrations before applying
   - Auto-generation may miss some changes (indexes, constraints, custom types)
   - Add custom logic as needed
   - Verify both `upgrade()` and `downgrade()` functions

2. **Test Migrations**
   ```bash
   # Test upgrade
   alembic upgrade head
   
   # Test downgrade
   alembic downgrade -1
   
   # Test upgrade again
   alembic upgrade head
   ```

3. **Use Transactions**
   - Migrations run in transactions by default
   - Failed migrations automatically rollback
   - Use `op.execute()` for raw SQL
   - For non-transactional operations (e.g., CREATE INDEX CONCURRENTLY), use:
     ```python
     def upgrade():
         op.execute("COMMIT")  # End transaction
         op.execute("CREATE INDEX CONCURRENTLY ...")
     ```

4. **Handle Data Migrations**
   - Separate schema and data migrations when possible
   - Use `op.bulk_insert()` for data
   - Consider using `op.execute()` for complex data transformations
   - Test with production-like data volumes

5. **Version Control**
   - Commit migrations with the code changes that require them
   - Never modify existing migrations that have been deployed
   - Use descriptive migration messages

### Rollback

```bash
# Rollback to specific version
alembic downgrade <revision_id>

# Rollback all migrations
alembic downgrade base

# Rollback to previous version
alembic downgrade -1
```

### Troubleshooting

**Issue: "Target database is not up to date"**
```bash
# Check current version
alembic current

# Stamp database to specific version (if you know the current state)
alembic stamp head

# Or stamp to a specific revision
alembic stamp <revision_id>
```

**Issue: "Can't locate revision identified by 'xxx'"**
```bash
# This usually means the alembic_version table is out of sync
# Option 1: Reset to base and reapply
alembic downgrade base
alembic upgrade head

# Option 2: Manually fix the alembic_version table
psql $DATABASE_URL -c "UPDATE alembic_version SET version_num='<correct_revision>';"
```

**Issue: "Multiple head revisions are present"**
```bash
# Show all heads
alembic heads

# Merge the heads
alembic merge -m "merge heads" <revision1> <revision2>
```

**Issue: Migration fails partway through**
```bash
# Check what state the database is in
alembic current

# If transaction rolled back, you can retry
alembic upgrade head

# If migration partially applied (non-transactional operations):
# 1. Manually fix the database state
# 2. Stamp to the target revision
alembic stamp <target_revision>
```

See [Alembic Documentation](https://alembic.sqlalchemy.org/) for more details.

## Testing

### Test Database Setup

#### 1. Set Environment Variable

```bash
# In .env or export
export TEST_DATABASE_URL="postgresql://rag_user:rag_password@localhost:5432/rag_test"
```

#### 2. Create Test Database

```bash
# Using setup script
python tests/setup_test_db.py

# Or manually
createdb rag_test
psql rag_test -c "CREATE EXTENSION IF NOT EXISTS vector"
```

#### 3. Run Migrations

```bash
# Set Alembic to use test database
alembic -x test=true upgrade head

# Or use TEST_DATABASE_URL
DATABASE_URL=$TEST_DATABASE_URL alembic upgrade head
```

### Running Tests

```bash
# All database tests
pytest tests/unit/database/ tests/integration/database/ -v

# Specific test file
pytest tests/unit/database/test_models.py -v

# With coverage
pytest tests/unit/database/ --cov=rag_factory.database --cov-report=html
```

### Test Fixtures

The project provides several pytest fixtures for database testing (defined in `tests/conftest.py`):

#### `db_connection` - Sync Database Session

Provides a synchronous database session with Alembic-managed schema. The database is set up once per test session and cleaned between tests.

```python
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
```

#### `db_service` - Async Database Service

Provides an async database service for testing async operations. Useful for integration tests that need to test the full service layer.

```python
import pytest

@pytest.mark.asyncio
async def test_store_chunks(db_service):
    """Test storing chunks with embeddings."""
    chunks = [
        {
            "id": "chunk-1",
            "text": "test content",
            "embedding": [0.1, 0.2, 0.3] * 512  # 1536 dimensions
        }
    ]
    
    await db_service.store_chunks(chunks)
    
    # Search for similar chunks
    results = await db_service.search_chunks([0.1, 0.2, 0.3] * 512, top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "test content"
```

#### `clean_database` - Empty Database

Provides a guaranteed empty database session. Useful when you need to test with a known clean state.

```python
def test_with_clean_db(clean_database):
    """Test with guaranteed empty database."""
    from rag_factory.database.models import Document
    
    # Database is empty
    count = clean_database.query(Document).count()
    assert count == 0
    
    # Add test data
    doc = Document(filename="test.txt", source_path="/path", content_hash="abc")
    clean_database.add(doc)
    clean_database.commit()
    
    # Verify
    assert clean_database.query(Document).count() == 1
```

See [tests/README.md](../../tests/README.md) for complete fixture documentation.

### Integration Tests

Integration tests require a real PostgreSQL database with pgvector extension:

```bash
# Set test database URL
export TEST_DATABASE_URL="postgresql://rag_user:rag_password@localhost:5432/rag_test"

# Run integration tests
pytest tests/integration/database/ -v -m integration

# Run specific integration test
pytest tests/integration/database/test_pgvector_integration.py -v
```

**Requirements for integration tests:**
- PostgreSQL 12+ with pgvector extension
- Test database created and accessible
- `TEST_DATABASE_URL` environment variable set
- Alembic migrations applied to test database

## Performance Optimization

### Vector Search Optimization

1. **Index Parameters**: Adjust HNSW parameters based on your data size
   - Small datasets (< 100K): `m=16, ef_construction=64`
   - Medium datasets (100K-1M): `m=24, ef_construction=128`
   - Large datasets (> 1M): `m=32, ef_construction=200`

2. **Query-Time Parameters**: Set `ef_search` for accuracy vs speed trade-off
   ```sql
   SET hnsw.ef_search = 100;  -- Higher = more accurate, slower
   ```

3. **Batch Insertions**: Use bulk operations for better performance
   ```python
   session.bulk_save_objects(chunks)
   ```

### Connection Pooling

- Adjust `pool_size` based on concurrent workload
- Monitor with `get_pool_status()`
- For serverless/edge: Use smaller pools (5-10)
- For dedicated servers: Use larger pools (20-50)

### Indexing Strategy

- Create indexes AFTER bulk data insertion
- Use `CONCURRENTLY` for large tables (doesn't lock)
- Monitor index usage with `pg_stat_user_indexes`

## Troubleshooting

### Common Issues

**Issue: "CREATE EXTENSION vector" permission denied**
```sql
-- Run as superuser
CREATE EXTENSION IF NOT EXISTS vector;
```

**Issue: Connection pool exhausted**
```python
# Increase pool size
config = DatabaseConfig(
    database_url="...",
    pool_size=20,
    max_overflow=40
)
```

**Issue: Slow vector search**
```sql
-- Check if index is being used
EXPLAIN ANALYZE
SELECT * FROM chunks
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 10;

-- Should show "Index Scan using idx_chunks_embedding_hnsw"
```

**Issue: Migration conflicts**
```bash
# Stamp database to specific version
alembic stamp head

# Or reset and rerun
alembic downgrade base
alembic upgrade head
```

## Security Best Practices

1. **Never commit credentials**: Use environment variables
2. **Use SSL/TLS**: Add `?sslmode=require` to connection string
3. **Principle of least privilege**: Create dedicated database user
4. **Rotate credentials**: Update passwords regularly
5. **Monitor access**: Enable PostgreSQL audit logging

## Monitoring

### Key Metrics

```sql
-- Connection count
SELECT count(*) FROM pg_stat_activity WHERE datname = 'your_database';

-- Table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
FROM pg_tables WHERE schemaname = 'public';

-- Index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Slow queries (if logging enabled)
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

## References

### Project Documentation

- [Epic 16: Database Migration System Consolidation](../epics/epic-16-database-consolidation.md) - Migration system consolidation rationale and plan
- [Environment Variables Guide](ENVIRONMENT_VARIABLES.md) - Complete environment variable reference
- [Migration Manager Removal Guide](MIGRATION_MANAGER_REMOVAL.md) - Guide for migrating from custom MigrationManager
- [Test Documentation](../../tests/README.md) - Complete testing guide and fixture documentation
- [Migration Audit](MIGRATION_AUDIT.md) - Detailed audit of migration systems
- [Consolidation Plan](CONSOLIDATION_PLAN.md) - Step-by-step consolidation plan

### Related Epics

- [Epic 2: Database & Storage Infrastructure](../epics/epic-02-database-storage.md) - Initial database implementation
- [Epic 11: Dependency Injection](../epics/epic-11-dependency-injection.md) - Service architecture
- [Epic 15: Test Coverage Improvements](../epics/epic-15-test-coverage-improvements) - Testing strategy

### External Documentation

- [Alembic Documentation](https://alembic.sqlalchemy.org/) - Database migration tool
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/) - Python SQL toolkit and ORM
- [pgvector Documentation](https://github.com/pgvector/pgvector) - Vector similarity search for PostgreSQL
- [PostgreSQL Documentation](https://www.postgresql.org/docs/) - PostgreSQL database
- [Neon Documentation](https://neon.tech/docs) - Serverless PostgreSQL platform
