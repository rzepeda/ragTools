# Story 16.4: Migrate Tests to Alembic

**Story ID:** 16.4  
**Epic:** Epic 16 - Database Migration System Consolidation  
**Story Points:** 5  
**Priority:** High  
**Dependencies:** Story 16.3 (Database fixtures created)

---

## User Story

**As a** developer  
**I want** all migration tests to use Alembic instead of custom MigrationManager  
**So that** tests validate the production migration system

---

## Detailed Requirements

### Functional Requirements

> [!NOTE]
> **Current State**: Migration tests use custom `MigrationManager` class
> 
> **Target State**: All tests use Alembic CLI and API
> 
> **Benefit**: Tests validate the actual production migration system

1. **Update Unit Tests**
   - Modify `tests/unit/database/test_migrations.py` to test Alembic
   - Test migration upgrade operations
   - Test migration downgrade operations
   - Test migration history tracking
   - Test migration idempotency

2. **Update Integration Tests**
   - Modify `tests/integration/database/test_migration_integration.py`
   - Test real database migration execution
   - Test schema version tracking
   - Test rollback functionality
   - Test migration with existing data

3. **Remove MigrationManager Dependencies**
   - Remove all `from rag_factory.database.migrations import MigrationManager` imports
   - Replace with Alembic API calls
   - Update test fixtures if needed
   - Update test data and mocks

4. **Add Alembic Test Utilities**
   - Create helper functions for Alembic operations in tests
   - Add fixtures for Alembic configuration
   - Add utilities for migration verification

### Non-Functional Requirements

1. **Test Coverage**
   - Maintain or improve test coverage
   - All migration scenarios covered
   - Edge cases tested

2. **Test Quality**
   - Clear test names and documentation
   - Proper assertions
   - Good error messages

---

## Acceptance Criteria

### AC1: Unit Test Updates
- [ ] `tests/unit/database/test_migrations.py` updated to use Alembic
- [ ] Test migration upgrade to head
- [ ] Test migration downgrade
- [ ] Test migration history
- [ ] Test current version retrieval
- [ ] Test migration idempotency
- [ ] No `MigrationManager` imports in unit tests

### AC2: Integration Test Updates
- [ ] `tests/integration/database/test_migration_integration.py` updated
- [ ] Test real database migration execution
- [ ] Test schema changes are applied
- [ ] Test rollback functionality
- [ ] Test migration with existing data
- [ ] No `MigrationManager` imports in integration tests

### AC3: Test Utilities
- [ ] Alembic test helper functions created
- [ ] Alembic configuration fixture added
- [ ] Migration verification utilities added
- [ ] Documentation for test utilities

### AC4: Test Quality
- [ ] All migration tests pass (100% success rate)
- [ ] Test coverage maintained or improved
- [ ] Type hints validated
- [ ] Linting passes
- [ ] Clear test documentation

---

## Technical Specifications

### Updated Unit Tests

```python
# tests/unit/database/test_migrations.py

"""Unit tests for Alembic database migrations."""

import pytest
import tempfile
import os
from pathlib import Path
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect


class TestAlembicMigrations:
    """Test suite for Alembic migration system."""
    
    @pytest.fixture
    def alembic_config(self, test_db_url):
        """
        Create Alembic configuration for testing.
        
        Args:
            test_db_url: Test database URL from fixture
            
        Returns:
            Alembic Config instance
        """
        # Get path to alembic.ini
        project_root = Path(__file__).parent.parent.parent.parent
        alembic_ini = project_root / "alembic.ini"
        
        # Create config
        config = Config(str(alembic_ini))
        config.set_main_option("sqlalchemy.url", test_db_url)
        
        return config
    
    def test_migration_upgrade_to_head(self, alembic_config, test_db_url):
        """Test upgrading migrations to head."""
        # Upgrade to head
        command.upgrade(alembic_config, "head")
        
        # Verify we're at head
        script = ScriptDirectory.from_config(alembic_config)
        head_revision = script.get_current_head()
        
        # Check database version
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            current = context.get_current_revision()
        
        assert current == head_revision, f"Expected {head_revision}, got {current}"
        engine.dispose()
    
    def test_migration_downgrade(self, alembic_config, test_db_url):
        """Test downgrading migrations."""
        # Upgrade to head first
        command.upgrade(alembic_config, "head")
        
        # Get current version
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            before_downgrade = context.get_current_revision()
        engine.dispose()
        
        # Downgrade one version
        command.downgrade(alembic_config, "-1")
        
        # Verify version changed
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            after_downgrade = context.get_current_revision()
        engine.dispose()
        
        assert after_downgrade != before_downgrade
    
    def test_migration_history(self, alembic_config):
        """Test retrieving migration history."""
        script = ScriptDirectory.from_config(alembic_config)
        
        # Get all revisions
        revisions = list(script.walk_revisions())
        
        # Should have at least 2 migrations (001 and 002)
        assert len(revisions) >= 2
        
        # Verify revision IDs
        revision_ids = [rev.revision for rev in revisions]
        assert "001" in revision_ids
        assert "002" in revision_ids
    
    def test_migration_idempotency(self, alembic_config, test_db_url):
        """Test that running migrations twice doesn't cause errors."""
        # Upgrade to head
        command.upgrade(alembic_config, "head")
        
        # Get version
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            version_1 = context.get_current_revision()
        engine.dispose()
        
        # Upgrade again (should be no-op)
        command.upgrade(alembic_config, "head")
        
        # Verify version unchanged
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            version_2 = context.get_current_revision()
        engine.dispose()
        
        assert version_1 == version_2
    
    def test_get_current_version(self, alembic_config, test_db_url):
        """Test retrieving current schema version."""
        # Upgrade to head
        command.upgrade(alembic_config, "head")
        
        # Get current version
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            current = context.get_current_revision()
        engine.dispose()
        
        assert current is not None
        assert len(current) > 0
    
    def test_migration_creates_tables(self, alembic_config, test_db_url):
        """Test that migrations create expected tables."""
        # Upgrade to head
        command.upgrade(alembic_config, "head")
        
        # Check tables exist
        engine = create_engine(test_db_url)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        # Should have documents and chunks tables
        assert "documents" in tables
        assert "chunks" in tables
        assert "alembic_version" in tables
        
        engine.dispose()
    
    def test_migration_creates_indexes(self, alembic_config, test_db_url):
        """Test that migrations create expected indexes."""
        # Upgrade to head
        command.upgrade(alembic_config, "head")
        
        # Check indexes exist
        engine = create_engine(test_db_url)
        inspector = inspect(engine)
        
        # Check chunks table indexes
        indexes = inspector.get_indexes("chunks")
        index_names = [idx['name'] for idx in indexes]
        
        # Should have vector index
        assert any("embedding" in name for name in index_names)
        
        engine.dispose()
```

### Updated Integration Tests

```python
# tests/integration/database/test_migration_integration.py

"""Integration tests for Alembic migrations with real database."""

import pytest
import os
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, text, inspect
from pathlib import Path


@pytest.mark.integration
class TestMigrationIntegration:
    """Integration tests for migration system."""
    
    @pytest.fixture
    def alembic_config(self, test_db_url):
        """Create Alembic configuration for testing."""
        project_root = Path(__file__).parent.parent.parent.parent
        alembic_ini = project_root / "alembic.ini"
        
        config = Config(str(alembic_ini))
        config.set_main_option("sqlalchemy.url", test_db_url)
        
        return config
    
    @pytest.mark.asyncio
    async def test_real_migration_execution(self, alembic_config, test_db_url):
        """Test running migrations against real database."""
        # Start from clean state
        command.downgrade(alembic_config, "base")
        
        # Upgrade to head
        command.upgrade(alembic_config, "head")
        
        # Verify tables exist
        engine = create_engine(test_db_url)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        assert "documents" in tables
        assert "chunks" in tables
        
        engine.dispose()
    
    @pytest.mark.asyncio
    async def test_migration_with_existing_data(self, alembic_config, test_db_url):
        """Test migration with existing data in database."""
        # Upgrade to initial schema
        command.upgrade(alembic_config, "001")
        
        # Insert test data
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO documents (document_id, filename, source_path, content_hash)
                VALUES (gen_random_uuid(), 'test.txt', '/test', 'abc123')
            """))
            conn.commit()
        engine.dispose()
        
        # Upgrade to head (should preserve data)
        command.upgrade(alembic_config, "head")
        
        # Verify data still exists
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM documents"))
            count = result.scalar()
        engine.dispose()
        
        assert count == 1
    
    @pytest.mark.asyncio
    async def test_rollback_functionality(self, alembic_config, test_db_url):
        """Test rolling back migrations."""
        # Upgrade to head
        command.upgrade(alembic_config, "head")
        
        # Verify hierarchy columns exist (from migration 002)
        engine = create_engine(test_db_url)
        inspector = inspect(engine)
        columns = [col['name'] for col in inspector.get_columns("chunks")]
        
        # Should have hierarchy columns
        has_hierarchy = any("parent" in col or "level" in col for col in columns)
        
        # Downgrade to 001
        command.downgrade(alembic_config, "001")
        
        # Verify hierarchy columns removed
        inspector = inspect(engine)
        columns_after = [col['name'] for col in inspector.get_columns("chunks")]
        
        engine.dispose()
        
        # Columns should be different
        assert len(columns) != len(columns_after)
    
    @pytest.mark.asyncio
    async def test_pgvector_extension_installed(self, alembic_config, test_db_url):
        """Test that pgvector extension is installed by migrations."""
        # Upgrade to head
        command.upgrade(alembic_config, "head")
        
        # Check pgvector extension
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                )
            """))
            has_pgvector = result.scalar()
        engine.dispose()
        
        assert has_pgvector is True
```

### Test Utilities

```python
# tests/utils/alembic_helpers.py

"""Helper utilities for Alembic testing."""

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine
from pathlib import Path
from typing import Optional


def get_alembic_config(database_url: str) -> Config:
    """
    Get Alembic configuration for testing.
    
    Args:
        database_url: Database URL
        
    Returns:
        Alembic Config instance
    """
    project_root = Path(__file__).parent.parent.parent
    alembic_ini = project_root / "alembic.ini"
    
    config = Config(str(alembic_ini))
    config.set_main_option("sqlalchemy.url", database_url)
    
    return config


def get_current_revision(database_url: str) -> Optional[str]:
    """
    Get current migration revision from database.
    
    Args:
        database_url: Database URL
        
    Returns:
        Current revision ID or None
    """
    engine = create_engine(database_url)
    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        current = context.get_current_revision()
    engine.dispose()
    
    return current


def get_head_revision(config: Config) -> str:
    """
    Get head revision from migration scripts.
    
    Args:
        config: Alembic configuration
        
    Returns:
        Head revision ID
    """
    script = ScriptDirectory.from_config(config)
    return script.get_current_head()


def upgrade_to_head(database_url: str) -> None:
    """
    Upgrade database to head revision.
    
    Args:
        database_url: Database URL
    """
    config = get_alembic_config(database_url)
    command.upgrade(config, "head")


def downgrade_to_base(database_url: str) -> None:
    """
    Downgrade database to base (no migrations).
    
    Args:
        database_url: Database URL
    """
    config = get_alembic_config(database_url)
    command.downgrade(config, "base")


def verify_migration_applied(database_url: str, revision: str) -> bool:
    """
    Verify a specific migration is applied.
    
    Args:
        database_url: Database URL
        revision: Migration revision ID
        
    Returns:
        True if migration is applied
    """
    current = get_current_revision(database_url)
    
    if current == revision:
        return True
    
    # Check if current is descendant of revision
    config = get_alembic_config(database_url)
    script = ScriptDirectory.from_config(config)
    
    # Walk from current to base
    for rev in script.walk_revisions(current, "base"):
        if rev.revision == revision:
            return True
    
    return False
```

---

## Testing Strategy

### Test Execution

```bash
# Run updated unit tests
pytest tests/unit/database/test_migrations.py -v

# Run updated integration tests
pytest tests/integration/database/test_migration_integration.py -v

# Verify no MigrationManager imports
grep -r "MigrationManager" tests/ --include="*.py"
# Should only find this story document, not in test code

# Run all database tests
pytest tests/unit/database/ tests/integration/database/ -v
```

### Verification Checklist

- [ ] All migration tests pass
- [ ] No `MigrationManager` imports in tests
- [ ] Alembic upgrade/downgrade tested
- [ ] Migration idempotency tested
- [ ] Real database integration tested
- [ ] Test utilities work correctly

---

## Definition of Done

- [ ] `tests/unit/database/test_migrations.py` updated to use Alembic
- [ ] `tests/integration/database/test_migration_integration.py` updated
- [ ] All `MigrationManager` imports removed from tests
- [ ] Alembic test utilities created
- [ ] All migration tests pass (100% success rate)
- [ ] Test coverage maintained or improved
- [ ] Type checking passes
- [ ] Linting passes
- [ ] Documentation updated
- [ ] PR approved and merged

---

## Notes

- **Alembic API** is well-documented and powerful
- **Test utilities** make tests cleaner and more maintainable
- **Integration tests** validate real migration behavior
- **This story** prepares for Story 16.5 (removing MigrationManager code)
- Tests now validate the **actual production migration system**

---

## Migration from MigrationManager to Alembic

### Before (MigrationManager)

```python
from rag_factory.database.migrations import MigrationManager

def test_migration_execution(db_service):
    manager = MigrationManager(db_service)
    executed = await manager.run_migrations()
    assert len(executed) > 0
```

### After (Alembic)

```python
from alembic import command
from tests.utils.alembic_helpers import get_alembic_config

def test_migration_execution(test_db_url):
    config = get_alembic_config(test_db_url)
    command.upgrade(config, "head")
    # Verify with Alembic API
```
