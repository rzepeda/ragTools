# Story 2.1: Set Up Vector Database with PG Vector

**Story ID:** 2.1
**Epic:** Epic 2 - Database & Storage Infrastructure
**Story Points:** 5
**Priority:** Critical
**Dependencies:** Story 1.1 (for data structure definitions)

---

## User Story

**As a** system
**I want** PostgreSQL with pgvector extension
**So that** I can store and search vector embeddings efficiently

---

## Detailed Requirements

### Functional Requirements

1. **Database Setup**
   - PostgreSQL 15+ installation with pgvector extension
   - Database configuration for vector operations
   - Connection pooling for efficient resource usage
   - Support for both local development and managed solutions (Neon)

2. **Schema Design**
   - **Chunks Table**: Store text chunks with vector embeddings
     - `chunk_id` (UUID, primary key)
     - `document_id` (UUID, foreign key)
     - `chunk_index` (INT, order within document)
     - `text` (TEXT, the chunk content)
     - `embedding` (VECTOR, pgvector type for embeddings)
     - `metadata` (JSONB, flexible metadata storage)
     - `created_at` (TIMESTAMP)
     - `updated_at` (TIMESTAMP)

   - **Documents Table**: Store document metadata
     - `document_id` (UUID, primary key)
     - `filename` (VARCHAR, original filename)
     - `source_path` (TEXT, path or URL)
     - `content_hash` (VARCHAR, for deduplication)
     - `total_chunks` (INT, number of chunks)
     - `metadata` (JSONB, custom metadata)
     - `status` (VARCHAR, processing status)
     - `created_at` (TIMESTAMP)
     - `updated_at` (TIMESTAMP)

3. **Indexes for Performance**
   - Vector similarity search index on `chunks.embedding` using HNSW or IVFFlat
   - B-tree index on `chunks.document_id` for filtering
   - B-tree index on `documents.content_hash` for deduplication
   - B-tree index on `chunks.created_at` for temporal queries

4. **Database Migration System**
   - Use Alembic for version-controlled migrations
   - Initial migration to create tables and indexes
   - Rollback capability for all migrations
   - Migration documentation

5. **Connection Management**
   - Connection pooling with configurable size
   - Connection retry logic with exponential backoff
   - Health check endpoint for database connectivity
   - Proper connection cleanup and resource management

### Non-Functional Requirements

1. **Performance**
   - Vector similarity search should return top-k results in <100ms for datasets up to 1M vectors
   - Support for batch insert operations (>1000 chunks/second)
   - Connection pool should handle 20+ concurrent connections

2. **Scalability**
   - Schema design should support millions of documents
   - Partitioning strategy for large tables (future consideration)
   - Support for read replicas (configuration ready)

3. **Reliability**
   - Transaction support for data consistency
   - Automatic reconnection on connection failures
   - Comprehensive error handling and logging

4. **Security**
   - Use environment variables for credentials
   - Support for SSL/TLS connections
   - Principle of least privilege for database user
   - No credentials in code or version control

5. **Maintainability**
   - Clear migration naming conventions
   - Database setup scripts for local development
   - Documentation for schema and indexes
   - Monitoring queries for database health

---

## Acceptance Criteria

### AC1: Database Installation and Configuration
- [ ] PostgreSQL 15+ installed (local) or Neon project created
- [ ] pgvector extension installed and enabled
- [ ] Database connection parameters configured via environment variables
- [ ] SSL/TLS enabled for remote connections

### AC2: Schema Creation
- [ ] Chunks table created with all specified columns
- [ ] Documents table created with all specified columns
- [ ] Foreign key relationship between chunks and documents enforced
- [ ] JSONB metadata fields support arbitrary JSON structures

### AC3: Vector Column Configuration
- [ ] `embedding` column uses pgvector VECTOR type
- [ ] Vector dimensions configurable (default: 1536 for OpenAI embeddings)
- [ ] Vector column supports NULL values for chunks without embeddings

### AC4: Indexes Created
- [ ] HNSW or IVFFlat index created on `chunks.embedding`
- [ ] B-tree indexes created on foreign keys and frequently queried columns
- [ ] Index creation performance measured and documented
- [ ] Query planner uses indexes correctly (verified with EXPLAIN)

### AC5: Connection Pooling
- [ ] Connection pool implemented with configurable min/max connections
- [ ] Pool size configurable via environment variables
- [ ] Connection timeout and retry logic implemented
- [ ] Connection pool metrics available (active, idle, waiting)

### AC6: Migration System
- [ ] Alembic initialized with migration directory
- [ ] Initial migration creates all tables and indexes
- [ ] Migration can be rolled back successfully
- [ ] Migration documentation in `docs/database/`

### AC7: Testing Infrastructure
- [ ] Test database setup script for local development
- [ ] Fixtures for populating test data
- [ ] Cleanup script to reset test database
- [ ] CI/CD integration with test database

---

## Technical Specifications

### File Structure
```
rag_factory/
├── database/
│   ├── __init__.py
│   ├── connection.py        # Connection pooling and management
│   ├── models.py            # SQLAlchemy models
│   └── config.py            # Database configuration
│
migrations/
├── alembic.ini              # Alembic configuration
├── env.py                   # Alembic environment
├── versions/
│   └── 001_initial_schema.py
│
scripts/
├── setup_database.sh        # Local database setup
└── test_database.sh         # Test database setup
```

### Dependencies
```python
# requirements.txt additions
psycopg2-binary==2.9.9      # PostgreSQL adapter
sqlalchemy==2.0.23          # ORM
alembic==1.13.1             # Migrations
pgvector==0.2.4             # pgvector Python client
```

### Database Configuration
```python
# rag_factory/database/config.py
from pydantic import BaseSettings, PostgresDsn

class DatabaseConfig(BaseSettings):
    database_url: PostgresDsn
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False

    class Config:
        env_prefix = "DB_"
        env_file = ".env"
```

### SQLAlchemy Models
```python
# rag_factory/database/models.py
from sqlalchemy import Column, String, Integer, Text, TIMESTAMP, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector
import uuid

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    document_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    source_path = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)
    total_chunks = Column(Integer, default=0)
    metadata = Column(JSONB, default={})
    status = Column(String(50), default="pending")
    created_at = Column(TIMESTAMP, nullable=False, server_default="NOW()")
    updated_at = Column(TIMESTAMP, nullable=False, server_default="NOW()", onupdate="NOW()")

class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)  # Configurable dimension
    metadata = Column(JSONB, default={})
    created_at = Column(TIMESTAMP, nullable=False, server_default="NOW()")
    updated_at = Column(TIMESTAMP, nullable=False, server_default="NOW()", onupdate="NOW()")

    __table_args__ = (
        Index("idx_chunks_embedding_hnsw", "embedding", postgresql_using="hnsw", postgresql_with={"m": 16, "ef_construction": 64}),
    )
```

### Connection Pool Setup
```python
# rag_factory/database/connection.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from .config import DatabaseConfig

class DatabaseConnection:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = create_engine(
            config.database_url,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_timeout=config.pool_timeout,
            pool_recycle=config.pool_recycle,
            echo=config.echo
        )
        self.SessionLocal = sessionmaker(bind=self.engine)

    @contextmanager
    def get_session(self) -> Session:
        """Context manager for database sessions."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception:
            return False
```

### Initial Migration
```python
# migrations/versions/001_initial_schema.py
"""Initial schema with documents and chunks tables

Revision ID: 001
Revises:
Create Date: 2024-01-XX XX:XX:XX
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector

def upgrade():
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create documents table
    op.create_table(
        "documents",
        sa.Column("document_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("total_chunks", sa.Integer(), default=0),
        sa.Column("metadata", JSONB(), default={}),
        sa.Column("status", sa.String(50), default="pending"),
        sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("NOW()"), nullable=False)
    )

    # Create chunks table
    op.create_table(
        "chunks",
        sa.Column("chunk_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("metadata", JSONB(), default={}),
        sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("NOW()"), nullable=False)
    )

    # Create indexes
    op.create_index("idx_documents_content_hash", "documents", ["content_hash"])
    op.create_index("idx_chunks_document_id", "chunks", ["document_id"])
    op.create_index("idx_chunks_created_at", "chunks", ["created_at"])

    # Create HNSW index for vector similarity search
    op.execute("""
        CREATE INDEX idx_chunks_embedding_hnsw
        ON chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

def downgrade():
    op.drop_table("chunks")
    op.drop_table("documents")
    op.execute("DROP EXTENSION IF EXISTS vector")
```

---

## Unit Tests

### Test File Location
`tests/unit/database/test_connection.py`
`tests/unit/database/test_models.py`

### Test Cases

#### TC2.1.1: Database Connection Tests
```python
import pytest
from rag_factory.database.connection import DatabaseConnection
from rag_factory.database.config import DatabaseConfig

def test_connection_pool_creation():
    """Test connection pool is created with correct parameters."""
    config = DatabaseConfig(
        database_url="postgresql://user:pass@localhost/testdb",
        pool_size=5,
        max_overflow=10
    )
    db = DatabaseConnection(config)
    assert db.engine.pool.size() == 5
    assert db.engine.pool._max_overflow == 10

def test_session_context_manager():
    """Test session context manager commits on success."""
    db = DatabaseConnection(test_config)

    with db.get_session() as session:
        # Perform operation
        session.execute("SELECT 1")

    # Session should be closed after context exit
    assert session.is_active == False

def test_session_rollback_on_error():
    """Test session rolls back on exception."""
    db = DatabaseConnection(test_config)

    with pytest.raises(Exception):
        with db.get_session() as session:
            session.execute("SELECT 1")
            raise Exception("Test error")

    # Session should be rolled back
    assert session.is_active == False

def test_health_check_success():
    """Test health check returns True for healthy database."""
    db = DatabaseConnection(test_config)
    assert db.health_check() == True

def test_health_check_failure():
    """Test health check returns False for unavailable database."""
    config = DatabaseConfig(database_url="postgresql://invalid:invalid@localhost/invalid")
    db = DatabaseConnection(config)
    assert db.health_check() == False
```

#### TC2.1.2: Model Definition Tests
```python
import pytest
from rag_factory.database.models import Document, Chunk, Base
from sqlalchemy import inspect
import numpy as np

def test_document_model_columns():
    """Test Document model has all required columns."""
    inspector = inspect(Document)
    columns = [col.name for col in inspector.columns]

    required_columns = [
        "document_id", "filename", "source_path", "content_hash",
        "total_chunks", "metadata", "status", "created_at", "updated_at"
    ]

    for col in required_columns:
        assert col in columns

def test_chunk_model_columns():
    """Test Chunk model has all required columns."""
    inspector = inspect(Chunk)
    columns = [col.name for col in inspector.columns]

    required_columns = [
        "chunk_id", "document_id", "chunk_index", "text",
        "embedding", "metadata", "created_at", "updated_at"
    ]

    for col in required_columns:
        assert col in columns

def test_document_creation():
    """Test Document instance can be created."""
    doc = Document(
        filename="test.txt",
        source_path="/path/to/test.txt",
        content_hash="abc123",
        total_chunks=5
    )

    assert doc.filename == "test.txt"
    assert doc.total_chunks == 5
    assert doc.document_id is not None

def test_chunk_creation_with_embedding():
    """Test Chunk can be created with vector embedding."""
    embedding = np.random.rand(1536).tolist()

    chunk = Chunk(
        document_id=uuid.uuid4(),
        chunk_index=0,
        text="Sample text",
        embedding=embedding
    )

    assert chunk.text == "Sample text"
    assert chunk.embedding is not None
    assert len(chunk.embedding) == 1536

def test_foreign_key_relationship():
    """Test foreign key relationship between Chunk and Document."""
    inspector = inspect(Chunk)
    fks = inspector.foreign_keys

    assert len(fks) > 0
    fk = list(fks)[0]
    assert fk.column.table.name == "documents"

def test_jsonb_metadata_field():
    """Test JSONB metadata field accepts arbitrary JSON."""
    doc = Document(
        filename="test.txt",
        source_path="/path/test.txt",
        content_hash="abc",
        metadata={"custom_field": "value", "nested": {"key": "val"}}
    )

    assert doc.metadata["custom_field"] == "value"
    assert doc.metadata["nested"]["key"] == "val"
```

#### TC2.1.3: Migration Tests
```python
import pytest
from alembic import command
from alembic.config import Config

def test_migration_upgrade():
    """Test migration upgrade creates tables."""
    alembic_cfg = Config("alembic.ini")

    # Run upgrade
    command.upgrade(alembic_cfg, "head")

    # Verify tables exist
    with engine.connect() as conn:
        result = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in result]

    assert "documents" in tables
    assert "chunks" in tables

def test_migration_downgrade():
    """Test migration downgrade removes tables."""
    alembic_cfg = Config("alembic.ini")

    # Run downgrade
    command.downgrade(alembic_cfg, "base")

    # Verify tables don't exist
    with engine.connect() as conn:
        result = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name IN ('documents', 'chunks')
        """)
        tables = [row[0] for row in result]

    assert len(tables) == 0

def test_pgvector_extension_enabled():
    """Test pgvector extension is enabled after migration."""
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")

    with engine.connect() as conn:
        result = conn.execute("""
            SELECT extname FROM pg_extension WHERE extname = 'vector'
        """)
        extensions = [row[0] for row in result]

    assert "vector" in extensions
```

---

## Integration Tests

### Test File Location
`tests/integration/database/test_database_integration.py`

### Test Scenarios

#### IS2.1.1: End-to-End Database Operations
```python
@pytest.mark.integration
def test_full_database_workflow():
    """Test complete workflow: connect, insert, query, delete."""
    db = DatabaseConnection(test_config)

    with db.get_session() as session:
        # Create document
        doc = Document(
            filename="test.txt",
            source_path="/test/test.txt",
            content_hash="hash123",
            total_chunks=2
        )
        session.add(doc)
        session.flush()

        # Create chunks
        chunks = [
            Chunk(
                document_id=doc.document_id,
                chunk_index=i,
                text=f"Chunk {i}",
                embedding=np.random.rand(1536).tolist()
            )
            for i in range(2)
        ]
        session.add_all(chunks)
        session.flush()

        # Query chunks
        results = session.query(Chunk).filter(
            Chunk.document_id == doc.document_id
        ).all()

        assert len(results) == 2

        # Delete document (cascade should delete chunks)
        session.delete(doc)
        session.flush()

        # Verify chunks deleted
        remaining = session.query(Chunk).filter(
            Chunk.document_id == doc.document_id
        ).count()

        assert remaining == 0

#### IS2.1.2: Vector Similarity Search
```python
@pytest.mark.integration
def test_vector_similarity_search():
    """Test vector similarity search with cosine distance."""
    db = DatabaseConnection(test_config)

    with db.get_session() as session:
        # Create test document
        doc = Document(
            filename="vectors.txt",
            source_path="/test/vectors.txt",
            content_hash="vec123"
        )
        session.add(doc)
        session.flush()

        # Create chunks with known embeddings
        base_vector = np.random.rand(1536)
        similar_vector = base_vector + np.random.rand(1536) * 0.1
        dissimilar_vector = np.random.rand(1536)

        chunks = [
            Chunk(document_id=doc.document_id, chunk_index=0,
                  text="Similar", embedding=similar_vector.tolist()),
            Chunk(document_id=doc.document_id, chunk_index=1,
                  text="Dissimilar", embedding=dissimilar_vector.tolist()),
        ]
        session.add_all(chunks)
        session.flush()

        # Perform similarity search
        query_vector = base_vector.tolist()
        results = session.execute(f"""
            SELECT chunk_id, text,
                   1 - (embedding <=> '{query_vector}') as similarity
            FROM chunks
            WHERE document_id = '{doc.document_id}'
            ORDER BY embedding <=> '{query_vector}'
            LIMIT 1
        """).fetchall()

        # Most similar should be returned first
        assert results[0][1] == "Similar"
        assert results[0][2] > 0.8  # High similarity score

#### IS2.1.3: Connection Pool Under Load
```python
@pytest.mark.integration
def test_connection_pool_concurrent_access():
    """Test connection pool handles concurrent requests."""
    import concurrent.futures

    db = DatabaseConnection(DatabaseConfig(
        database_url=test_url,
        pool_size=5,
        max_overflow=10
    ))

    def query_database(i):
        with db.get_session() as session:
            result = session.execute("SELECT 1").scalar()
            return result

    # Execute 50 concurrent queries
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(query_database, i) for i in range(50)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # All queries should succeed
    assert len(results) == 50
    assert all(r == 1 for r in results)

#### IS2.1.4: Large Batch Insert Performance
```python
@pytest.mark.integration
def test_batch_insert_performance():
    """Test batch insert performance meets requirements."""
    import time

    db = DatabaseConnection(test_config)

    with db.get_session() as session:
        doc = Document(
            filename="batch.txt",
            source_path="/batch/test.txt",
            content_hash="batch123"
        )
        session.add(doc)
        session.flush()

        # Insert 10000 chunks
        chunks = [
            Chunk(
                document_id=doc.document_id,
                chunk_index=i,
                text=f"Chunk {i}",
                embedding=np.random.rand(1536).tolist()
            )
            for i in range(10000)
        ]

        start = time.time()
        session.bulk_save_objects(chunks)
        session.flush()
        duration = time.time() - start

        # Should insert >1000 chunks/second
        throughput = 10000 / duration
        assert throughput > 1000, f"Throughput {throughput:.2f} chunks/sec is too low"
```

---

## Definition of Done

- [ ] PostgreSQL with pgvector is set up (local or Neon)
- [ ] All tables and indexes created via migrations
- [ ] SQLAlchemy models defined and tested
- [ ] Connection pooling implemented and configured
- [ ] All unit tests pass with >90% coverage
- [ ] All integration tests pass
- [ ] Migration upgrade and downgrade tested
- [ ] Database setup scripts documented
- [ ] Environment variable configuration documented
- [ ] Code reviewed
- [ ] No linting errors

---

## Testing Checklist

### Unit Testing
- [ ] Connection pool created correctly
- [ ] Session context manager works
- [ ] Health check function works
- [ ] Model columns defined correctly
- [ ] Foreign key relationships enforced
- [ ] JSONB metadata accepts JSON
- [ ] Migration upgrade/downgrade works

### Integration Testing
- [ ] Full CRUD workflow works
- [ ] Vector similarity search works
- [ ] Connection pool handles concurrency
- [ ] Batch inserts meet performance requirements
- [ ] Cascade delete works correctly
- [ ] Indexes improve query performance

### Performance Testing
- [ ] Vector search <100ms for 1M vectors
- [ ] Batch insert >1000 chunks/second
- [ ] Connection pool handles 20+ concurrent connections
- [ ] Query planner uses indexes

---

## Setup Instructions

### Local Development

1. **Install PostgreSQL 15+**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install postgresql-15 postgresql-contrib

   # macOS
   brew install postgresql@15
   ```

2. **Install pgvector**
   ```bash
   cd /tmp
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   make
   sudo make install
   ```

3. **Create Database**
   ```bash
   psql postgres -c "CREATE DATABASE rag_factory_dev;"
   psql rag_factory_dev -c "CREATE EXTENSION vector;"
   ```

4. **Set Environment Variables**
   ```bash
   export DB_DATABASE_URL="postgresql://user:password@localhost/rag_factory_dev"
   export DB_POOL_SIZE=10
   export DB_MAX_OVERFLOW=20
   ```

5. **Run Migrations**
   ```bash
   alembic upgrade head
   ```

### Using Neon (Managed PostgreSQL)

1. **Create Neon Project**
   - Sign up at https://neon.tech
   - Create new project with Postgres 15
   - Enable pgvector extension in project settings

2. **Get Connection String**
   - Copy connection string from Neon dashboard
   - Set as environment variable

3. **Run Migrations**
   ```bash
   export DB_DATABASE_URL="postgresql://user:password@ep-xxx.neon.tech/neondb?sslmode=require"
   alembic upgrade head
   ```

---

## Notes for Developers

1. **Vector Dimensions**: Default is 1536 (OpenAI embeddings). Update if using different embedding model.

2. **Index Selection**: HNSW is faster for queries but slower for inserts. Consider IVFFlat for write-heavy workloads.

3. **Connection Pooling**: Adjust pool size based on concurrent workload. Start with 10, increase if needed.

4. **Migrations**: Always test migrations in a staging environment before production.

5. **Performance Monitoring**: Use `EXPLAIN ANALYZE` to verify query plans use indexes correctly.

6. **Backup Strategy**: Configure automated backups (Neon provides this by default).

7. **Cost Optimization**: For Neon, use connection pooling to minimize active connections and reduce costs.
