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

All database configuration is managed via environment variables with `DB_` prefix:

```bash
# Required
export DB_DATABASE_URL="postgresql://user:password@host:port/database"

# Optional (with defaults)
export DB_POOL_SIZE=10              # Connection pool size
export DB_MAX_OVERFLOW=20           # Max overflow connections
export DB_POOL_TIMEOUT=30           # Connection timeout in seconds
export DB_POOL_RECYCLE=3600         # Recycle connections after seconds
export DB_ECHO=false                # Enable SQL query logging
export DB_POOL_PRE_PING=true        # Test connections before use
export DB_VECTOR_DIMENSIONS=1536    # Embedding dimensions
```

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

## Database Setup

### Local PostgreSQL Setup

```bash
# 1. Run setup script
./scripts/setup_database.sh

# 2. Run migrations
export DB_DATABASE_URL="postgresql://postgres@localhost/rag_factory_dev"
alembic upgrade head

# 3. Verify
python -c "from rag_factory.database import DatabaseConnection; \
           db = DatabaseConnection(); \
           print('Health check:', db.health_check())"
```

### Using Neon (Managed PostgreSQL)

```bash
# 1. Create Neon project at https://neon.tech
# 2. Enable pgvector extension in project settings
# 3. Get connection string from dashboard
# 4. Run migrations
export DB_DATABASE_URL="postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require"
alembic upgrade head
```

## Migrations

### Creating a New Migration

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "description of changes"

# Create empty migration
alembic revision -m "manual migration"
```

### Running Migrations

```bash
# Upgrade to latest
alembic upgrade head

# Upgrade one version
alembic upgrade +1

# Downgrade one version
alembic downgrade -1

# Show current version
alembic current

# Show migration history
alembic history
```

### Rollback

```bash
# Rollback to specific version
alembic downgrade <revision_id>

# Rollback all migrations
alembic downgrade base
```

## Testing

### Unit Tests

```bash
# Run all database unit tests
pytest tests/unit/database/ -v

# Run specific test file
pytest tests/unit/database/test_models.py -v
```

### Integration Tests

```bash
# Requires PostgreSQL running
export DB_DATABASE_URL="postgresql://postgres@localhost/rag_factory_test"
./scripts/test_database.sh

# Run integration tests
pytest tests/integration/database/ -v -m integration
```

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

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Neon Documentation](https://neon.tech/docs)
