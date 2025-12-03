# Epic 2: Database & Storage Infrastructure - Stories Summary

**Epic Goal:** Set up the database layer with PostgreSQL + pgvector for vector storage and implement repository patterns for data access.

**Total Story Points:** 13
**Dependencies:** Epic 1 (Story 1.1 for interface definitions)

---

## Stories Overview

### Story 2.1: Set Up Vector Database with PG Vector
**Story Points:** 5
**File:** `story-2.1-vector-database-setup.md`
**Status:** Ready for Development

#### What's Included:
✅ **User Story & Requirements**
- Functional requirements for PostgreSQL + pgvector setup
- Schema design for documents and chunks tables
- Vector similarity search indexes (HNSW)
- Connection pooling and migration system
- Non-functional requirements (performance, scalability, reliability, security)

✅ **Acceptance Criteria (7 ACs)**
- AC1: Database Installation and Configuration
- AC2: Schema Creation
- AC3: Vector Column Configuration
- AC4: Indexes Created
- AC5: Connection Pooling
- AC6: Migration System
- AC7: Testing Infrastructure

✅ **Technical Specifications**
- Complete file structure
- SQLAlchemy models with pgvector support
- Database configuration with Pydantic
- Connection pool implementation with context managers
- Initial migration script (upgrade/downgrade)
- Environment variable configuration

✅ **Working Code Examples**
```python
# Database Models (Document, Chunk with Vector column)
# Connection Pool Setup
# Alembic Migration Scripts
# Health Check Implementation
```

✅ **Unit Tests (14 test cases)**
- TC2.1.1: Database Connection Tests (5 tests)
  - Connection pool creation
  - Session context manager
  - Session rollback on error
  - Health check success/failure

- TC2.1.2: Model Definition Tests (6 tests)
  - Document/Chunk model columns
  - Document/Chunk creation
  - Foreign key relationships
  - JSONB metadata fields

- TC2.1.3: Migration Tests (3 tests)
  - Migration upgrade
  - Migration downgrade
  - pgvector extension enabled

✅ **Integration Tests (4 scenarios)**
- IS2.1.1: End-to-End Database Operations
- IS2.1.2: Vector Similarity Search
- IS2.1.3: Connection Pool Under Load
- IS2.1.4: Large Batch Insert Performance

✅ **Performance Benchmarks**
- Vector search <100ms for 1M vectors
- Batch insert >1000 chunks/second
- Connection pool handles 20+ concurrent connections

✅ **Setup Instructions**
- Local PostgreSQL installation
- pgvector extension setup
- Neon (managed PostgreSQL) setup
- Environment configuration
- Migration execution

✅ **Definition of Done Checklist**
- 11 items covering database setup, testing, documentation, and code review

---

### Story 2.2: Implement Database Repository Pattern
**Story Points:** 8
**File:** `story-2.2-database-repository-pattern.md`
**Status:** Ready for Development

#### What's Included:
✅ **User Story & Requirements**
- Functional requirements for repository pattern
- Base repository interface with generic type support
- DocumentRepository with full CRUD operations
- ChunkRepository with vector search capabilities
- Transaction management
- Error handling with custom exceptions
- Performance optimizations (batch operations, connection reuse)
- Non-functional requirements (performance, testability, maintainability, reliability, extensibility)

✅ **Acceptance Criteria (8 ACs)**
- AC1: Base Repository Structure
- AC2: DocumentRepository Implementation
- AC3: ChunkRepository Implementation
- AC4: Vector Search Accuracy
- AC5: Transaction Management
- AC6: Error Handling
- AC7: Performance Requirements
- AC8: Testing

✅ **Technical Specifications**
- Complete file structure for repositories
- Base repository abstract class with generics
- Custom exception hierarchy
- DocumentRepository implementation
- ChunkRepository implementation with vector search

✅ **Working Code Examples**
```python
# BaseRepository (Abstract class with Generic[T])
# Custom Exceptions (EntityNotFoundError, DuplicateEntityError, etc.)
# DocumentRepository (19 methods)
  - CRUD operations
  - Deduplication via content hash
  - Status filtering
  - Pagination
  - Bulk operations

# ChunkRepository (14 methods)
  - CRUD operations
  - Vector similarity search
  - Filtered vector search (by document_id)
  - Metadata-based filtering
  - Bulk embedding updates
```

✅ **Unit Tests (60+ test cases)**
- TC2.2.1: DocumentRepository CRUD Tests (13 tests)
  - Create, read, update, delete
  - Duplicate detection
  - Pagination
  - Status filtering
  - Bulk operations

- TC2.2.2: ChunkRepository CRUD Tests (11 tests)
  - Create with/without embeddings
  - Get by ID and by document
  - Count operations
  - Update embeddings
  - Bulk create and update
  - Delete operations

- TC2.2.3: Vector Search Tests (7 tests)
  - Basic similarity search
  - Threshold filtering
  - Document filtering
  - Metadata filtering
  - Similarity score accuracy
  - Error handling

- TC2.2.4: Transaction Tests (4 tests)
  - Commit and rollback
  - Multiple operations in transaction
  - Automatic rollback on error

✅ **Integration Tests (2 scenarios)**
- IS2.2.1: Full Repository Workflow
  - Complete document lifecycle
  - Chunk creation with embeddings
  - Vector search
  - Cascade delete

- IS2.2.2: Concurrent Repository Access
  - Multiple repository instances
  - Thread-safe operations
  - Connection pool utilization

✅ **Performance Benchmarks (3 tests)**
```python
# Single operations <10ms
# Bulk insert >1000 chunks/second
# Vector search <100ms (10K chunks)
```

✅ **Definition of Done Checklist**
- 11 items covering implementation, testing, performance, and documentation

---

## Developer Workflow

### Prerequisites
1. Complete Story 1.1 (RAG Strategy Interface) from Epic 1
2. Python 3.11+ environment set up
3. PostgreSQL 15+ or Neon account

### Story 2.1: Vector Database Setup

**Steps:**
1. Install PostgreSQL and pgvector (or setup Neon)
2. Create database configuration module (`rag_factory/database/config.py`)
3. Define SQLAlchemy models (`rag_factory/database/models.py`)
4. Implement connection pooling (`rag_factory/database/connection.py`)
5. Set up Alembic and create initial migration
6. Write unit tests for connection and models
7. Write integration tests for database operations
8. Run performance benchmarks
9. Document setup procedures

**Testing:**
```bash
# Unit tests
pytest tests/unit/database/ -v

# Integration tests
pytest tests/integration/database/ -v --integration

# Performance tests
pytest tests/integration/database/ -v -m benchmark
```

**Verification:**
- [ ] All unit tests pass (>90% coverage)
- [ ] All integration tests pass
- [ ] Performance benchmarks meet requirements
- [ ] Migration upgrade/downgrade works
- [ ] Database can be set up from scratch following docs

---

### Story 2.2: Database Repository Pattern

**Steps:**
1. Create base repository abstract class (`rag_factory/repositories/base.py`)
2. Define custom exceptions (`rag_factory/repositories/exceptions.py`)
3. Implement DocumentRepository (`rag_factory/repositories/document.py`)
4. Implement ChunkRepository (`rag_factory/repositories/chunk.py`)
5. Write unit tests for all repository methods
6. Write integration tests for workflows
7. Write performance benchmarks
8. Document repository usage

**Testing:**
```bash
# Unit tests
pytest tests/unit/repositories/ -v

# Integration tests
pytest tests/integration/repositories/ -v --integration

# Performance benchmarks
pytest tests/benchmarks/test_repository_performance.py -v
```

**Verification:**
- [ ] All CRUD operations work correctly
- [ ] Vector search returns accurate results
- [ ] Transaction management works (commit/rollback)
- [ ] Error handling covers all edge cases
- [ ] Performance requirements met
- [ ] All tests pass with >90% coverage

---

## Code Quality Standards

### Type Hints
All functions and methods must have complete type hints:
```python
def create(self, filename: str, source_path: str, content_hash: str,
           metadata: dict = None) -> Document:
    ...
```

### Documentation
- Module-level docstrings explaining purpose
- Class docstrings with usage examples
- Method docstrings with parameters and return values
- Inline comments for complex logic

### Error Handling
- Use custom exception classes
- Provide meaningful error messages
- Include context in exceptions
- Never swallow exceptions silently

### Testing Requirements
- Unit test coverage >90%
- Integration tests for all workflows
- Performance benchmarks for critical operations
- Test edge cases and error conditions

---

## Dependencies Between Stories

```
Story 2.1 (Database Setup)
    ↓
Story 2.2 (Repository Pattern)
```

**Story 2.2 depends on Story 2.1** because:
- Repositories need database models from 2.1
- Repositories use connection pool from 2.1
- Tests require working database from 2.1

**Recommended Approach:**
1. Complete Story 2.1 fully (including tests)
2. Verify database is working correctly
3. Then start Story 2.2

---

## Testing Strategy

### Test Pyramid

```
         /\
        /  \    Integration Tests (20%)
       /----\   - Full workflows
      /      \  - Multiple components
     /--------\ - Real database
    /          \
   /------------\ Unit Tests (80%)
  /______________\ - Individual functions
                  - Mocked dependencies
                  - Fast execution
```

### Test Database

Create separate test database:
```bash
createdb rag_factory_test
psql rag_factory_test -c "CREATE EXTENSION vector;"
```

Use environment variable:
```bash
export DB_DATABASE_URL_TEST="postgresql://user:pass@localhost/rag_factory_test"
```

### Fixtures

Common test fixtures provided:
```python
@pytest.fixture
def db_session():
    """Provides clean database session for each test."""

@pytest.fixture
def doc_repo(db_session):
    """Provides DocumentRepository instance."""

@pytest.fixture
def chunk_repo(db_session):
    """Provides ChunkRepository instance."""

@pytest.fixture
def sample_document(doc_repo):
    """Provides a sample document for testing."""
```

---

## Performance Requirements Summary

| Operation | Target | Story |
|-----------|--------|-------|
| Single document operation | <10ms | 2.2 |
| Bulk insert (chunks) | >1000/sec | 2.1, 2.2 |
| Vector search (1M vectors) | <100ms | 2.1, 2.2 |
| Connection pool capacity | 20+ concurrent | 2.1 |

---

## Files Created

### Story 2.1 Files
```
rag_factory/
├── database/
│   ├── __init__.py
│   ├── config.py          # Database configuration
│   ├── models.py          # SQLAlchemy models
│   └── connection.py      # Connection pooling

migrations/
├── alembic.ini
├── env.py
└── versions/
    └── 001_initial_schema.py

tests/
├── unit/database/
│   ├── test_connection.py
│   └── test_models.py
└── integration/database/
    └── test_database_integration.py
```

### Story 2.2 Files
```
rag_factory/
├── repositories/
│   ├── __init__.py
│   ├── base.py           # Base repository
│   ├── exceptions.py     # Custom exceptions
│   ├── document.py       # DocumentRepository
│   └── chunk.py          # ChunkRepository

tests/
├── unit/repositories/
│   ├── test_base.py
│   ├── test_document_repository.py
│   ├── test_chunk_repository.py
│   └── test_exceptions.py
├── integration/repositories/
│   └── test_repository_integration.py
└── benchmarks/
    └── test_repository_performance.py
```

---

## Getting Started

### For Developers Starting Story 2.1:

1. Read `story-2.1-vector-database-setup.md` completely
2. Review acceptance criteria and test cases
3. Set up PostgreSQL or Neon account
4. Create feature branch: `git checkout -b feature/story-2.1-vector-database`
5. Start with database configuration and models
6. Write tests as you go (TDD approach)
7. Run tests frequently: `pytest tests/unit/database/ -v`
8. Complete all acceptance criteria
9. Run full test suite and benchmarks
10. Update Definition of Done checklist
11. Create PR for review

### For Developers Starting Story 2.2:

1. Ensure Story 2.1 is complete and merged
2. Read `story-2.2-database-repository-pattern.md` completely
3. Review acceptance criteria and test cases
4. Create feature branch: `git checkout -b feature/story-2.2-repositories`
5. Start with base repository and exceptions
6. Implement DocumentRepository, then ChunkRepository
7. Write tests as you go (TDD approach)
8. Run tests frequently: `pytest tests/unit/repositories/ -v`
9. Test vector search thoroughly with real embeddings
10. Complete all acceptance criteria
11. Run performance benchmarks
12. Update Definition of Done checklist
13. Create PR for review

---

## Questions or Issues?

If you encounter any issues or have questions while implementing these stories:

1. **Ambiguous Requirements**: Raise during sprint planning or daily standup
2. **Technical Blockers**: Check documentation, then ask team
3. **Test Failures**: Review test output, check database state
4. **Performance Issues**: Use profiling tools, review indexes
5. **Integration Problems**: Ensure Story 2.1 is fully working

---

## Success Criteria for Epic 2

Epic 2 is complete when:

- [x] Story 2.1: Vector Database Setup (5 points)
  - PostgreSQL with pgvector running
  - All tables and indexes created
  - Migrations working
  - All tests passing

- [x] Story 2.2: Database Repository Pattern (8 points)
  - All repository classes implemented
  - Vector search working accurately
  - All tests passing
  - Performance benchmarks met

- [x] Integration: Stories work together
  - Repositories use database from 2.1
  - Vector search works end-to-end
  - Full document lifecycle tested

- [x] Documentation complete
  - Setup instructions verified
  - API documentation written
  - Developer notes included

---

**Total Epic Points:** 13
**Estimated Sprint Velocity:** Sprint 1 (with Epic 1)
**Story Dependencies:** 2.1 → 2.2 (sequential)
