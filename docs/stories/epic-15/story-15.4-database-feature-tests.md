# Story 15.4: Add Database Feature Tests

**Story ID:** 15.4  
**Epic:** Epic 15 - Test Coverage Improvements  
**Story Points:** 8  
**Priority:** Medium  
**Dependencies:** Epic 11 (Database Service)

---

## User Story

**As a** developer  
**I want** comprehensive tests for database features  
**So that** I can ensure vector storage, indexing, and migrations work correctly

---

## Detailed Requirements

### Functional Requirements

> [!NOTE]
> **Current Database Coverage**: 40% (4 out of 10 features tested)
> **Target Coverage**: 70%+
> **Existing Tests**: `test_connection.py`, `test_models.py`, `test_chunk_repository.py`, `test_document_repository.py`
> **Missing**: pgvector specifics, vector indexing, batch operations, migrations

1. **pgvector Integration Tests**
   - Test vector similarity search (cosine, L2)
   - Test vector dimension validation
   - Test vector normalization
   - Test bulk vector operations

2. **Vector Indexing Tests**
   - Test HNSW index creation and usage
   - Test IVFFlat index creation and usage
   - Test index performance characteristics
   - Test index rebuild operations

3. **Batch Operations Tests**
   - Test batch vector insertion
   - Test batch chunk storage
   - Test transaction rollback on batch errors
   - Test batch size optimization

4. **Migration Tests**
   - Test migration script execution
   - Test schema versioning
   - Test rollback functionality
   - Test migration idempotency

---

## Acceptance Criteria

### AC1: pgvector Integration Tests
- [ ] Test file `tests/unit/database/test_pgvector.py` created
- [ ] Test vector similarity search (cosine distance)
- [ ] Test vector similarity search (L2 distance)
- [ ] Test vector dimension mismatch errors
- [ ] Test vector normalization
- [ ] Integration test with real PostgreSQL + pgvector
- [ ] Minimum 10 test cases

### AC2: Vector Indexing Tests
- [ ] Test file `tests/unit/database/test_vector_indexing.py` created
- [ ] Test HNSW index creation with parameters
- [ ] Test IVFFlat index creation with parameters
- [ ] Test index usage in similarity queries
- [ ] Test index rebuild after data changes
- [ ] Test index performance (query time)
- [ ] Integration test with real database
- [ ] Minimum 12 test cases

### AC3: Batch Operations Tests
- [ ] Test file `tests/unit/database/test_batch_operations.py` created
- [ ] Test batch vector insertion (100+ vectors)
- [ ] Test batch chunk storage with metadata
- [ ] Test transaction commit on success
- [ ] Test transaction rollback on error
- [ ] Test optimal batch size determination
- [ ] Test memory usage during batch operations
- [ ] Minimum 10 test cases

### AC4: Migration Tests
- [ ] Test file `tests/unit/database/test_migrations.py` created
- [ ] Test migration script discovery
- [ ] Test migration execution order
- [ ] Test schema version tracking
- [ ] Test rollback to previous version
- [ ] Test migration idempotency (running twice)
- [ ] Test migration with existing data
- [ ] Integration test with real database
- [ ] Minimum 12 test cases

### AC5: Test Quality
- [ ] All tests pass (100% success rate)
- [ ] Database coverage increases from 40% to 70%+
- [ ] Integration tests use Docker PostgreSQL with pgvector
- [ ] Proper cleanup after each test
- [ ] Type hints validated
- [ ] Linting passes

---

## Technical Specifications

### File Structure

```
tests/unit/database/
├── test_connection.py          # Existing
├── test_models.py              # Existing
├── test_pgvector.py            # NEW
├── test_vector_indexing.py     # NEW
├── test_batch_operations.py    # NEW
└── test_migrations.py          # NEW

tests/integration/database/
├── test_database_integration.py  # Existing
├── test_pgvector_integration.py  # NEW
└── test_migration_integration.py # NEW
```

### pgvector Test Template

```python
"""Unit tests for pgvector integration."""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from rag_factory.database.pgvector_service import PgVectorService

class TestPgVectorIntegration:
    """Test suite for pgvector functionality."""
    
    def test_cosine_similarity_search(self, db_service):
        """Test vector similarity search using cosine distance."""
        # Insert test vectors
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.9, 0.1, 0.0]),
            np.array([0.0, 1.0, 0.0])
        ]
        
        for i, vec in enumerate(vectors):
            db_service.store_vector(f"doc_{i}", vec)
        
        # Query with similar vector
        query_vec = np.array([1.0, 0.0, 0.0])
        results = db_service.similarity_search(query_vec, top_k=2, metric='cosine')
        
        assert len(results) == 2
        assert results[0]['id'] == 'doc_0'  # Exact match
        assert results[1]['id'] == 'doc_1'  # Close match
        assert results[0]['score'] > results[1]['score']
    
    def test_vector_dimension_validation(self, db_service):
        """Test that dimension mismatches are caught."""
        # Store 3D vector
        db_service.store_vector("doc_1", np.array([1.0, 0.0, 0.0]))
        
        # Try to query with 2D vector
        with pytest.raises(ValueError, match="dimension"):
            db_service.similarity_search(np.array([1.0, 0.0]), top_k=5)
```

### Vector Indexing Test Template

```python
"""Unit tests for vector indexing."""
import pytest
from rag_factory.database.vector_indexing import VectorIndexManager

class TestVectorIndexing:
    """Test suite for vector index management."""
    
    def test_create_hnsw_index(self, db_service):
        """Test HNSW index creation with parameters."""
        index_manager = VectorIndexManager(db_service)
        
        index_manager.create_index(
            index_type='hnsw',
            m=16,  # Number of connections
            ef_construction=64
        )
        
        # Verify index exists
        indexes = db_service.list_indexes()
        assert any(idx['type'] == 'hnsw' for idx in indexes)
    
    def test_hnsw_index_improves_query_speed(self, db_service):
        """Test that HNSW index improves query performance."""
        # Insert many vectors
        for i in range(1000):
            vec = np.random.rand(384)
            db_service.store_vector(f"doc_{i}", vec)
        
        # Query without index
        import time
        start = time.time()
        db_service.similarity_search(np.random.rand(384), top_k=10)
        time_without_index = time.time() - start
        
        # Create index
        index_manager = VectorIndexManager(db_service)
        index_manager.create_index(index_type='hnsw')
        
        # Query with index
        start = time.time()
        db_service.similarity_search(np.random.rand(384), top_k=10)
        time_with_index = time.time() - start
        
        # Index should be faster (or at least not slower)
        assert time_with_index <= time_without_index * 1.5
```

### Migration Test Template

```python
"""Unit tests for database migrations."""
import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

class TestDatabaseMigrations:
    """Test suite for Alembic migration system."""
    
    def test_migration_execution_order(self, test_db_url):
        """Test that migrations execute in correct order."""
        # Configure Alembic
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", test_db_url)
        
        # Get migration script directory
        script = ScriptDirectory.from_config(alembic_cfg)
        
        # Verify migrations exist and are ordered
        revisions = list(script.walk_revisions())
        assert len(revisions) > 0, "No migrations found"
        
        # Run migrations to head
        command.upgrade(alembic_cfg, "head")
        
        # Verify current version
        from alembic.runtime.migration import MigrationContext
        from sqlalchemy import create_engine
        
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            current_version = context.get_current_revision()
            assert current_version is not None, "No migration version set"
    
    def test_migration_idempotency(self, test_db_url):
        """Test that running migrations twice doesn't cause errors."""
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", test_db_url)
        
        # Run migrations
        command.upgrade(alembic_cfg, "head")
        
        # Get version
        from alembic.runtime.migration import MigrationContext
        from sqlalchemy import create_engine
        
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            version_1 = context.get_current_revision()
        
        # Run again (should be no-op)
        command.upgrade(alembic_cfg, "head")
        
        # Get version again
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            version_2 = context.get_current_revision()
        
        # Should be same version, no errors
        assert version_1 == version_2
```


### Testing Strategy

1. **Unit Tests**
   - Mock database connections
   - Test logic in isolation
   - Fast execution

2. **Integration Tests**
   - Use Docker PostgreSQL with pgvector extension
   - Test with real database operations
   - Clean up after each test

3. **Performance Tests**
   - Measure query times with/without indexes
   - Test batch operation efficiency
   - Verify memory usage

---

## Definition of Done

- [ ] All 4 new unit test files created
- [ ] All 2 new integration test files created
- [ ] All tests pass (100% success rate)
- [ ] Database coverage reaches 70%+ (from 40%)
- [ ] Integration tests use Docker PostgreSQL
- [ ] Type checking passes
- [ ] Linting passes
- [ ] PR merged

---

## Notes

- Current database coverage is 40% (4 out of 10 features tested)
- pgvector extension must be installed in test database
- Use Docker Compose for integration test database
- Migration tests should use temporary test database
