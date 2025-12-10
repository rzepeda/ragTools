# Epic 16: Database Migration System Consolidation

**Epic Goal:** Consolidate database migration management to use Alembic exclusively, removing the custom `MigrationManager` implementation and establishing clear environment variable standards for database connections across development, testing, and production environments.

**Epic Story Points Total:** 23


**Dependencies:** 
- Epic 2 (Database & Storage Infrastructure - COMPLETED ✅)
- Epic 11 (Dependency Injection - COMPLETED ✅)
- Epic 15 (Test Coverage Improvements - IN PROGRESS)

**Status:** Ready for implementation

---

## Background

The project currently has **two separate migration systems** running in parallel:

1. **Alembic** (Production-ready) - Located in `migrations/` directory
   - Industry-standard migration tool for SQLAlchemy
   - Python-based migrations with upgrade/downgrade functions
   - Tracks versions in `alembic_version` table
   - Supports auto-generation from model changes

2. **Custom MigrationManager** (Test implementation) - Located in `rag_factory/database/migrations.py`
   - Simple SQL file executor created for Story 15.4 testing requirements
   - Tracks versions in `schema_migrations` table (conflicts with Alembic!)
   - Limited functionality compared to Alembic

### Problems with Current State

1. **Duplication & Confusion**: Two version tracking tables (`alembic_version` vs `schema_migrations`)
2. **Inconsistent Migration Formats**: Python files (`.py`) vs SQL files (`.sql`)
3. **Maintenance Burden**: Two systems to maintain, document, and test
4. **Environment Variable Mismatch**: Tests expect `TEST_DATABASE_URL` but `.env` provides `DATABASE_TEST_URL`
5. **Missing Test Fixtures**: `db_connection` fixture not defined, causing 57 test failures
6. **Documentation Confusion**: Docs reference only Alembic, but code has both systems

### Why Alembic Should Be the Only System

✅ **Industry Standard** - Battle-tested by thousands of production applications  
✅ **Auto-generation** - Detects model changes automatically  
✅ **Proper Rollback** - Full upgrade/downgrade support  
✅ **SQLAlchemy Integration** - Seamless ORM integration  
✅ **Already Configured** - `alembic.ini` and working migrations exist  
✅ **Better Tooling** - CLI commands, history tracking, branching support  

---

## Environment Variable Standards

### Production Environment Variables

**Primary Database Connection:**
```bash
# Main database URL (used by application and Alembic)
DATABASE_URL=postgresql://user:password@host:5432/database_name

# Optional: Separate read replica
DATABASE_READ_URL=postgresql://user:password@read-host:5432/database_name
```

### Test Environment Variables

**Test Database Connection:**
```bash
# Test database URL (used by pytest and integration tests)
TEST_DATABASE_URL=postgresql://user:password@host:5432/test_database_name

# Alternative: Use separate test credentials
DATABASE_TEST_URL=postgresql://test_user:test_password@host:5432/test_db
```

### Development Environment Variables

**Local Development:**
```bash
# Development database (local PostgreSQL or Docker)
DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_factory

# Test database for local testing
TEST_DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_test
```

**VM/Remote Development:**
```bash
# Host IP for VM accessing host machine services
HOST_IP=192.168.56.1

# Development database via host
DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_factory

# Test database via host
TEST_DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_test
```

### Environment Variable Usage Matrix

| Variable | Used By | Purpose | Required |
|----------|---------|---------|----------|
| `DATABASE_URL` | Application, Alembic, Examples | Main database connection | ✅ Yes |
| `TEST_DATABASE_URL` | Pytest, Integration Tests | Test database connection | ✅ For tests |
| `DATABASE_TEST_URL` | Legacy (deprecated) | Old test connection | ❌ Remove |
| `HOST_IP` | VM Development | Host machine IP for services | ⚠️ VM only |
| `RUN_DB_TESTS` | Pytest | Enable/disable database tests | Optional |
| `RUN_INTEGRATION_TESTS` | Pytest | Enable/disable integration tests | Optional |

---

## Story 16.1: Audit and Document Migration Systems

**As a** developer  
**I want** a comprehensive audit of both migration systems  
**So that** I understand what needs to be migrated and what can be removed

**Acceptance Criteria:**
- [ ] Document all existing Alembic migrations and their purpose
- [ ] Document all custom `MigrationManager` usage in codebase
- [ ] Identify all tests using `MigrationManager`
- [ ] Identify all tests using Alembic
- [ ] Create migration plan document
- [ ] List all environment variables currently in use
- [ ] Document breaking changes for users

**Deliverables:**
- `docs/database/MIGRATION_AUDIT.md` - Audit report
- `docs/database/CONSOLIDATION_PLAN.md` - Migration plan
- Updated `docs/database/README.md` - Clear migration documentation

**Story Points:** 3

---

## Story 16.2: Standardize Environment Variables

**As a** developer  
**I want** consistent environment variable naming across all environments  
**So that** configuration is predictable and tests work correctly

**Acceptance Criteria:**
- [ ] Update `.env` to use `TEST_DATABASE_URL` instead of `DATABASE_TEST_URL`
- [ ] Update `.env.example` with all standard variables
- [ ] Create `tests/.env.test` template for test configuration
- [ ] Update all test files to use `TEST_DATABASE_URL`
- [ ] Update `migrations/env.py` to support both `DATABASE_URL` and `TEST_DATABASE_URL`
- [ ] Add environment variable validation on startup
- [ ] Document all variables in `docs/database/ENVIRONMENT_VARIABLES.md`
- [ ] Update CI/CD configuration files

**Environment Variable Files:**

**.env (Development):**
```bash
# Database Configuration
HOST_IP=192.168.56.1
DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_factory
TEST_DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_test

# Test Flags
RUN_DB_TESTS=true
RUN_INTEGRATION_TESTS=true
```

**tests/.env.test (Test Template):**
```bash
# Test Database Configuration
# Copy to .env.test and update with your test database credentials

TEST_DATABASE_URL=postgresql://user:password@localhost:5432/rag_test

# Optional: Individual connection parameters
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_test
DB_USER=test_user
DB_PASSWORD=test_password

# Test Execution Flags
RUN_DB_TESTS=true
RUN_INTEGRATION_TESTS=true
RUN_LLM_TESTS=false  # Requires API keys
```

**.env.example (Template):**
```bash
# =============================================================================
# RAG Factory - Environment Configuration Template
# =============================================================================

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/rag_factory
TEST_DATABASE_URL=postgresql://user:password@localhost:5432/rag_test

# Optional: VM Development (if running in VM accessing host services)
# HOST_IP=192.168.56.1
# DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_factory

# Test Execution Flags
RUN_DB_TESTS=true
RUN_INTEGRATION_TESTS=true
RUN_LLM_TESTS=false

# LLM Configuration (Optional)
# OPENAI_API_KEY=sk-...
# COHERE_API_KEY=...

# Neo4j Configuration (Optional)
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=password
```

**Story Points:** 5

---

## Story 16.3: Create Database Connection Fixtures

**As a** developer  
**I want** proper pytest fixtures for database connections  
**So that** all 57 failing database tests can pass

**Acceptance Criteria:**
- [ ] Create `db_connection` fixture in `tests/conftest.py`
- [ ] Create `db_service` fixture for async database operations
- [ ] Create `test_db_url` fixture that reads `TEST_DATABASE_URL`
- [ ] Implement automatic test database creation/cleanup
- [ ] Add transaction rollback for test isolation
- [ ] Support both sync and async database operations
- [ ] Add fixture documentation in `tests/README.md`
- [ ] All 57 database tests pass with new fixtures

**Fixture Implementation:**

```python
# tests/conftest.py

import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from rag_factory.database.models import Base
from rag_factory.database.connection import DatabaseConnection
from rag_factory.services.database.postgres import PostgresqlDatabaseService

@pytest.fixture(scope="session")
def test_db_url():
    """Get test database URL from environment."""
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    return url

@pytest.fixture(scope="session")
def db_engine(test_db_url):
    """Create database engine for testing."""
    engine = create_engine(test_db_url)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()

@pytest.fixture(scope="function")
def db_connection(db_engine):
    """Provide database connection with transaction rollback."""
    connection = db_engine.connect()
    transaction = connection.begin()
    
    Session = sessionmaker(bind=connection)
    session = Session()
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(scope="function")
async def db_service(test_db_url):
    """Provide async database service for integration tests."""
    # Parse URL for connection params
    from urllib.parse import urlparse
    parsed = urlparse(test_db_url)
    
    service = PostgresqlDatabaseService(
        host=parsed.hostname,
        port=parsed.port or 5432,
        database=parsed.path.lstrip('/'),
        user=parsed.username,
        password=parsed.password
    )
    
    yield service
    
    await service.close()
```

**Story Points:** 5

---

## Story 16.4: Migrate Tests to Alembic

**As a** developer  
**I want** all migration tests to use Alembic instead of custom MigrationManager  
**So that** tests validate the production migration system

**Acceptance Criteria:**
- [ ] Update `tests/unit/database/test_migrations.py` to test Alembic
- [ ] Update `tests/integration/database/test_migration_integration.py` to use Alembic
- [ ] Remove all `MigrationManager` imports from tests
- [ ] Test Alembic upgrade/downgrade operations
- [ ] Test Alembic auto-generation from model changes
- [ ] Test migration idempotency
- [ ] Test migration rollback functionality
- [ ] All migration tests pass

**Updated Test Structure:**

```python
# tests/unit/database/test_migrations.py
"""Unit tests for Alembic migrations."""
import pytest
from alembic import command
from alembic.config import Config

class TestAlembicMigrations:
    """Test suite for Alembic migration system."""
    
    def test_migration_upgrade(self, test_db_url):
        """Test upgrading to latest migration."""
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", test_db_url)
        
        # Upgrade to head
        command.upgrade(alembic_cfg, "head")
        
        # Verify current version
        from alembic.script import ScriptDirectory
        script = ScriptDirectory.from_config(alembic_cfg)
        head_revision = script.get_current_head()
        
        # Get current database version
        from alembic.runtime.migration import MigrationContext
        from sqlalchemy import create_engine
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            current = context.get_current_revision()
        
        assert current == head_revision
    
    def test_migration_downgrade(self, test_db_url):
        """Test downgrading migrations."""
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", test_db_url)
        
        # Upgrade to head
        command.upgrade(alembic_cfg, "head")
        
        # Downgrade one version
        command.downgrade(alembic_cfg, "-1")
        
        # Verify downgrade worked
        # (check that we're not at head anymore)
```

**Story Points:** 5

---

## Story 16.5: Remove Custom MigrationManager

**As a** developer  
**I want** the custom MigrationManager code removed  
**So that** there's only one migration system to maintain

**Acceptance Criteria:**
- [ ] Remove `rag_factory/database/migrations.py`
- [ ] Remove `MigrationManager` imports from all files
- [ ] Update documentation to remove MigrationManager references
- [ ] Remove `schema_migrations` table creation code
- [ ] Update Story 15.4 documentation to reflect Alembic usage
- [ ] Verify no broken imports or references
- [ ] All tests still pass
- [ ] Update `docs/database/README.md` to show only Alembic

**Files to Remove:**
- `rag_factory/database/migrations.py`
- Any SQL migration files in non-Alembic locations

**Files to Update:**
- `docs/stories/epic-15/story-15.4-database-feature-tests.md` - Update to reference Alembic
- `docs/database/README.md` - Remove MigrationManager references
- Any example files using MigrationManager

**Story Points:** 3

---

## Story 16.6: Update Database Documentation

**As a** developer  
**I want** the database documentation updated to reflect Epic 16 decisions  
**So that** documentation is accurate and consistent with the consolidated migration system

**Acceptance Criteria:**
- [ ] Remove all MigrationManager references from `docs/database/README.md`
- [ ] Add Environment Variables section with examples
- [ ] Add Testing section with fixture documentation
- [ ] Update Migrations section (Alembic only)
- [ ] Add References section linking to Epic 16 and related docs
- [ ] Update Quick Start guide with standard variable names
- [ ] All code examples tested and working
- [ ] No broken links

**Key Updates:**
- Environment Variables section (new)
- Testing section with fixture examples (new)
- Migrations section (Alembic only)
- Quick Start updated with `TEST_DATABASE_URL`
- References section (new)

**Story Points:** 2

---

## Sprint Planning

This epic is recommended for **Sprint 16** as a cleanup/consolidation sprint.

**Total Sprint 16:** 23 points (Epic 16 only)

**Recommended Order:**
1. Story 16.1 (Audit) - Week 1
2. Story 16.2 (Environment Variables) - Week 1
3. Story 16.3 (Fixtures) - Week 2
4. Story 16.4 (Migrate Tests) - Week 2
5. Story 16.5 (Remove Custom Code) - Week 3
6. Story 16.6 (Update Documentation) - Week 3

---

## Technical Stack

**Migration Tool:**
- Alembic (already installed and configured)

**Database:**
- PostgreSQL 15+ with pgvector extension

**Python Libraries:**
- SQLAlchemy (ORM)
- asyncpg (async database driver)
- pytest (testing framework)
- python-dotenv (environment variable loading)

---

## Success Criteria

- [ ] Only Alembic migrations exist (custom MigrationManager removed)
- [ ] All environment variables follow standard naming convention
- [ ] `.env`, `.env.example`, and `tests/.env.test` are consistent
- [ ] `db_connection` fixture works for all database tests
- [ ] All 57 previously failing database tests now pass
- [ ] Migration tests validate Alembic functionality
- [ ] Documentation clearly explains Alembic-only approach
- [ ] No references to `MigrationManager` in codebase
- [ ] CI/CD uses standardized environment variables
- [ ] Zero migration-related test failures

---

## Migration Guide for Users

### Breaking Changes

1. **Environment Variable Rename:**
   - **Old:** `DATABASE_TEST_URL`
   - **New:** `TEST_DATABASE_URL`
   - **Action:** Update your `.env` files

2. **Migration System:**
   - **Old:** Custom `MigrationManager` class
   - **New:** Alembic CLI commands only
   - **Action:** Use `alembic upgrade head` instead of custom scripts

### Migration Steps

**Step 1: Update Environment Variables**
```bash
# In your .env file, change:
# DATABASE_TEST_URL=postgresql://...
# to:
TEST_DATABASE_URL=postgresql://...
```

**Step 2: Run Alembic Migrations**
```bash
# Instead of custom migration manager:
# python -m rag_factory.database.migrations

# Use Alembic:
alembic upgrade head
```

**Step 3: Update Test Configuration**
```bash
# Create tests/.env.test from template
cp tests/.env.test.example tests/.env.test

# Edit with your test database credentials
nano tests/.env.test
```

**Step 4: Verify Setup**
```bash
# Test database connection
pytest tests/unit/database/test_connection.py -v

# Run all database tests
pytest tests/unit/database/ -v
pytest tests/integration/database/ -v
```

---

## Documentation Updates

### New Documents to Create

1. **`docs/database/ENVIRONMENT_VARIABLES.md`**
   - Complete reference for all database environment variables
   - Examples for different deployment scenarios
   - Troubleshooting guide

2. **`docs/database/MIGRATION_GUIDE.md`**
   - How to create new migrations with Alembic
   - How to upgrade/downgrade
   - Best practices for schema changes

3. **`tests/README.md`**
   - How to set up test environment
   - Available fixtures and their usage
   - Running database tests

### Documents to Update

1. **`docs/database/README.md`**
   - Remove MigrationManager references
   - Expand Alembic documentation
   - Add environment variable section

2. **`docs/getting-started/installation.md`**
   - Update database setup instructions
   - Add environment variable configuration
   - Update migration commands

3. **`docs/stories/epic-15/story-15.4-database-feature-tests.md`**
   - Update to reference Alembic instead of MigrationManager
   - Update test templates

---

## Rollback Plan

If issues arise during consolidation:

1. **Keep Alembic migrations** - They are the source of truth
2. **Revert environment variable changes** - Can be done independently
3. **Restore MigrationManager temporarily** - Only if critical tests fail
4. **Fix forward** - Preferred approach, fix issues rather than rollback

---

## Notes

- Current database coverage is 40% (4 out of 10 features tested)
- This epic will help achieve the 70%+ target from Epic 15
- Alembic is already configured and working in the project
- The custom MigrationManager was created for Story 15.4 but is redundant
- 57 database tests are currently failing due to missing fixtures
- Environment variable inconsistency affects both development and CI/CD

---

## References

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [pytest Fixtures Documentation](https://docs.pytest.org/en/stable/fixture.html)
- Epic 2: Database & Storage Infrastructure
- Epic 11: Dependency Injection & Service Interface Decoupling
- Epic 15: Test Coverage Improvements
