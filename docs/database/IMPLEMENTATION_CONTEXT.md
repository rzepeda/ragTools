# Database Migration Consolidation - Implementation Context

**Quick Reference for Stories 16.2-16.5**

---

## The Problem (TL;DR)

We have **two migration systems** running in parallel:
1. **Alembic** (production-ready) in `migrations/`
2. **MigrationManager** (test-only) in `rag_factory/database/migrations.py`

This causes:
- Environment variable mismatch → Alembic fails
- Missing test fixtures → 31 tests fail
- Confusion about which system to use

**Solution**: Consolidate to Alembic only.

---

## Story 16.2: Environment Variables

### What's Wrong
```python
# DatabaseConfig expects (with DB_ prefix):
DB_DATABASE_URL

# But .env provides (no prefix):
DATABASE_URL
DATABASE_TEST_URL
```

### What to Change

**1. Update `.env`:**
```bash
# Add these (keep old ones for backward compatibility):
DB_DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_factory
DB_TEST_DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_test
```

**2. Update test files:**
- `tests/integration/database/test_pgvector_integration.py` line 8
- `tests/integration/database/test_migration_integration.py` line 9

Change:
```python
DB_URL = os.getenv("TEST_DATABASE_URL")  # OLD
DB_URL = os.getenv("DB_TEST_DATABASE_URL")  # NEW
```

**3. Verify:**
```bash
alembic current  # Should work now
```

---

## Story 16.3: Test Fixtures

### What's Missing
31 tests expect `db_connection` fixture but it doesn't exist in `tests/conftest.py`.

### What to Create

Add to `tests/conftest.py`:

```python
import os
import pytest
from alembic import command
from alembic.config import Config
from rag_factory.database.config import DatabaseConfig
from rag_factory.database.connection import DatabaseConnection

@pytest.fixture(scope="session")
def db_connection():
    """Database connection with Alembic-managed schema."""
    # Get test database URL
    db_url = os.getenv("DB_TEST_DATABASE_URL")
    if not db_url:
        pytest.skip("DB_TEST_DATABASE_URL not set")
    
    # Run Alembic migrations
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(alembic_cfg, "head")
    
    # Create connection
    config = DatabaseConfig(database_url=db_url)
    connection = DatabaseConnection(config)
    
    yield connection
    
    # Cleanup
    connection.drop_tables()
    connection.close()
```

### Tests That Will Pass
All 31 tests in:
- `tests/unit/database/test_connection.py` (12 tests)
- `tests/unit/database/test_models.py` (5 tests)
- `tests/integration/database/test_database_integration.py` (14 tests)

---

## Story 16.4: Migrate Tests to Alembic

### Files to Update

**1. `tests/unit/database/test_migrations.py`** (3 tests)

Replace:
```python
from rag_factory.database.migrations import MigrationManager

manager = MigrationManager(db_service)
await manager.run_migrations()
version = await manager.get_current_version()
```

With:
```python
from alembic import command
from alembic.config import Config

alembic_cfg = Config("alembic.ini")
command.upgrade(alembic_cfg, "head")
# Use alembic.script.ScriptDirectory for version info
```

**2. `tests/integration/database/test_migration_integration.py`** (1 test)

Same replacement as above.

---

## Story 16.5: Cleanup

### What to Delete
- `rag_factory/database/migrations.py` (entire file)

### Verification Before Deleting
```bash
# Should return nothing:
grep -r "MigrationManager" rag_factory/ tests/ --include="*.py"
```

### Documentation to Update
- `README.md` - Add Alembic usage section
- Create `docs/database/README.md` - Migration workflow guide

---

## Quick Reference

### Alembic Commands
```bash
# Check current version
alembic current

# Apply all migrations
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "description"

# Rollback one migration
alembic downgrade -1
```

### File Locations
- Alembic config: `alembic.ini`
- Alembic env: `migrations/env.py`
- Migrations: `migrations/versions/`
- Database config: `rag_factory/database/config.py`
- Test fixtures: `tests/conftest.py`

---

## Need More Details?

See full documentation:
- `docs/database/MIGRATION_AUDIT.md` - Complete analysis
- `docs/database/CONSOLIDATION_PLAN.md` - Detailed plan with rollback procedures
