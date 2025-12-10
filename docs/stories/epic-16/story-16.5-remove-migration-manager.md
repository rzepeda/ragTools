# Story 16.5: Remove Custom MigrationManager

**Story ID:** 16.5  
**Epic:** Epic 16 - Database Migration System Consolidation  
**Story Points:** 3  
**Priority:** Medium  
**Dependencies:** Story 16.4 (Tests migrated to Alembic)

---

## User Story

**As a** developer  
**I want** the custom MigrationManager code removed  
**So that** there's only one migration system to maintain

---

## Detailed Requirements

### Functional Requirements

> [!CAUTION]
> **Breaking Change**: The `MigrationManager` class will be completely removed
> 
> **Impact**: Any code using `MigrationManager` must be updated to use Alembic
> 
> **Mitigation**: All known usages updated in Story 16.4

1. **Remove MigrationManager Code**
   - Delete `rag_factory/database/migrations.py`
   - Remove any compiled Python cache files
   - Verify no broken imports remain

2. **Update Documentation**
   - Remove MigrationManager references from all docs
   - Update Story 15.4 to reference Alembic
   - Update database README
   - Update getting started guide

3. **Clean Up Database Schema**
   - Document that `schema_migrations` table is deprecated
   - Add note about `alembic_version` table being the source of truth
   - Optionally: Create script to drop `schema_migrations` table

4. **Verify No Broken References**
   - Search entire codebase for MigrationManager imports
   - Check for any remaining SQL migration files
   - Verify all tests still pass

### Non-Functional Requirements

1. **Completeness**
   - All MigrationManager code removed
   - All documentation updated
   - No broken references

2. **Safety**
   - All tests pass after removal
   - No runtime errors
   - Clear migration guide for users

---

## Acceptance Criteria

### AC1: Code Removal
- [ ] `rag_factory/database/migrations.py` deleted
- [ ] Compiled cache files removed (`__pycache__/*.pyc`)
- [ ] No MigrationManager imports in codebase
- [ ] No broken imports or references

### AC2: Documentation Updates
- [ ] `docs/database/README.md` updated (MigrationManager removed)
- [ ] `docs/stories/epic-15/story-15.4-database-feature-tests.md` updated
- [ ] `docs/getting-started/installation.md` updated
- [ ] All migration docs reference only Alembic

### AC3: Schema Cleanup Documentation
- [ ] `schema_migrations` table deprecation documented
- [ ] Migration guide for users with existing `schema_migrations` table
- [ ] Optional cleanup script created

### AC4: Verification
- [ ] All tests pass (100% success rate)
- [ ] No MigrationManager references in code
- [ ] No import errors
- [ ] Type checking passes
- [ ] Linting passes

---

## Technical Specifications

### Files to Remove

```bash
# Main file to delete
rag_factory/database/migrations.py

# Cache files to clean
rag_factory/database/__pycache__/migrations.cpython-*.pyc
```

### Verification Commands

```bash
# Search for any MigrationManager references
grep -r "MigrationManager" rag_factory/ tests/ docs/ --include="*.py" --include="*.md"

# Search for migrations.py imports
grep -r "from.*migrations import" rag_factory/ tests/ --include="*.py"
grep -r "import.*migrations" rag_factory/ tests/ --include="*.py"

# Verify no broken imports
python -c "import rag_factory; print('OK')"

# Run all tests
pytest tests/ -v

# Type checking
mypy rag_factory/

# Linting
pylint rag_factory/
```

### Documentation Updates

#### 1. Update `docs/database/README.md`

**Remove section:**
```markdown
## Custom Migration Manager

The project includes a custom MigrationManager class...
```

**Update to:**
```markdown
## Migrations

The project uses Alembic for database schema migrations.

### Running Migrations

```bash
# Upgrade to latest
alembic upgrade head

# Downgrade one version
alembic downgrade -1

# Show current version
alembic current

# Show migration history
alembic history
```

### Creating Migrations

```bash
# Auto-generate from model changes
alembic revision --autogenerate -m "description"

# Create empty migration
alembic revision -m "manual migration"
```

See [Alembic Documentation](https://alembic.sqlalchemy.org/) for more details.
```

#### 2. Update `docs/stories/epic-15/story-15.4-database-feature-tests.md`

**Find and replace:**
```markdown
# Old
from rag_factory.database.migrations import MigrationManager

class TestDatabaseMigrations:
    def test_migration_execution_order(self, db_service):
        manager = MigrationManager(db_service)
        executed = manager.run_migrations()

# New
from alembic import command
from alembic.config import Config

class TestDatabaseMigrations:
    def test_migration_execution_order(self, test_db_url):
        config = Config("alembic.ini")
        config.set_main_option("sqlalchemy.url", test_db_url)
        command.upgrade(config, "head")
```

#### 3. Update `docs/getting-started/installation.md`

**Remove:**
```markdown
## Option 2: Using Custom Migration Manager

```python
from rag_factory.database.migrations import MigrationManager
manager = MigrationManager(db_service)
await manager.run_migrations()
```
```

**Keep only:**
```markdown
## Run Database Migrations

```bash
# Using Alembic
alembic upgrade head
```
```

### Schema Cleanup Script

```python
# scripts/cleanup_schema_migrations.py

"""
Script to clean up deprecated schema_migrations table.

This table was used by the old MigrationManager system.
The project now uses Alembic, which uses the alembic_version table.

Usage:
    python scripts/cleanup_schema_migrations.py
"""

import os
import sys
from sqlalchemy import create_engine, text

def cleanup_schema_migrations():
    """Drop deprecated schema_migrations table."""
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)
    
    print("⚠️  This will drop the 'schema_migrations' table")
    print("   This table is no longer used (replaced by 'alembic_version')")
    
    response = input("\nContinue? (yes/no): ")
    if response.lower() != "yes":
        print("Cancelled")
        sys.exit(0)
    
    # Connect and drop table
    engine = create_engine(database_url)
    
    with engine.connect() as conn:
        # Check if table exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'schema_migrations'
            )
        """))
        exists = result.scalar()
        
        if not exists:
            print("✅ Table 'schema_migrations' does not exist (already clean)")
            return
        
        # Drop table
        print("Dropping 'schema_migrations' table...")
        conn.execute(text("DROP TABLE schema_migrations"))
        conn.commit()
        
        print("✅ Successfully dropped 'schema_migrations' table")
        print("   Migration tracking now uses 'alembic_version' table only")
    
    engine.dispose()

if __name__ == "__main__":
    cleanup_schema_migrations()
```

### Migration Guide for Users

Create `docs/database/MIGRATION_MANAGER_REMOVAL.md`:

```markdown
# MigrationManager Removal Guide

## What Changed

The custom `MigrationManager` class has been removed. The project now uses **Alembic exclusively** for database migrations.

## Impact

### If you were using MigrationManager in your code:

**Before:**
```python
from rag_factory.database.migrations import MigrationManager

manager = MigrationManager(db_service, migrations_dir="migrations")
executed = await manager.run_migrations()
version = await manager.get_current_version()
```

**After:**
```python
from alembic import command
from alembic.config import Config

config = Config("alembic.ini")
config.set_main_option("sqlalchemy.url", database_url)

# Run migrations
command.upgrade(config, "head")

# Get current version
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine

engine = create_engine(database_url)
with engine.connect() as conn:
    context = MigrationContext.configure(conn)
    current_version = context.get_current_revision()
```

### Database Schema

The `schema_migrations` table is no longer used. Alembic uses the `alembic_version` table instead.

**To clean up (optional):**
```bash
python scripts/cleanup_schema_migrations.py
```

## Why This Change

1. **Industry Standard**: Alembic is the standard migration tool for SQLAlchemy
2. **Better Features**: Auto-generation, proper rollback, branching support
3. **Less Maintenance**: One system instead of two
4. **Better Testing**: Tests now validate the actual production migration system

## Migration Steps

1. **Update your code** to use Alembic instead of MigrationManager
2. **Run migrations** with Alembic: `alembic upgrade head`
3. **Optional**: Clean up `schema_migrations` table
4. **Update documentation** in your projects

## Need Help?

See:
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [docs/database/README.md](README.md)
- [docs/epics/epic-16-database-consolidation.md](../epics/epic-16-database-consolidation.md)
```

---

## Testing Strategy

### Pre-Removal Verification

```bash
# Verify tests pass before removal
pytest tests/ -v

# Find all MigrationManager usage
grep -r "MigrationManager" rag_factory/ tests/ --include="*.py"
# Should only be in migrations.py (about to be deleted)
```

### Post-Removal Verification

```bash
# Verify file is deleted
ls rag_factory/database/migrations.py
# Should show "No such file"

# Verify no broken imports
python -c "import rag_factory; print('OK')"

# Run all tests
pytest tests/ -v

# Verify no MigrationManager references
grep -r "MigrationManager" rag_factory/ tests/ --include="*.py"
# Should show no results

# Type checking
mypy rag_factory/

# Linting
pylint rag_factory/
```

### Documentation Verification

```bash
# Check documentation for MigrationManager references
grep -r "MigrationManager" docs/ --include="*.md"
# Should only be in migration guide explaining the removal

# Verify Alembic is documented
grep -r "alembic" docs/ --include="*.md"
# Should show multiple references
```

---

## Definition of Done

- [ ] `rag_factory/database/migrations.py` deleted
- [ ] All cache files cleaned
- [ ] No MigrationManager imports in codebase
- [ ] All documentation updated
- [ ] `MIGRATION_MANAGER_REMOVAL.md` created
- [ ] Schema cleanup script created
- [ ] All tests pass (100% success rate)
- [ ] No import errors
- [ ] Type checking passes
- [ ] Linting passes
- [ ] PR approved and merged

---

## Rollback Plan

If issues are discovered after removal:

1. **Restore from Git**
   ```bash
   git checkout HEAD~1 -- rag_factory/database/migrations.py
   ```

2. **Fix Forward (Preferred)**
   - Identify the issue
   - Fix using Alembic
   - Don't restore MigrationManager

3. **Temporary Workaround**
   - If critical issue found
   - Can temporarily restore file
   - Must create follow-up story to remove again

---

## Notes

- **This is the final cleanup story** in Epic 16
- **All dependencies completed** before this story
- **Low risk** - all usage already migrated in Story 16.4
- **Clear migration guide** helps users transition
- **Optional cleanup script** for `schema_migrations` table
- **Documentation is critical** - users need to know what changed

---

## Success Metrics

- **Before**: 2 migration systems (Alembic + MigrationManager)
- **After**: 1 migration system (Alembic only)
- **Code removed**: ~107 lines (migrations.py)
- **Maintenance burden**: Reduced
- **Confusion**: Eliminated
- **Tests**: All passing
- **Documentation**: Clear and accurate

---

## Cleanup Checklist

### Code
- [ ] Delete `rag_factory/database/migrations.py`
- [ ] Delete `__pycache__/migrations.*.pyc`
- [ ] Verify no broken imports

### Documentation
- [ ] Update `docs/database/README.md`
- [ ] Update `docs/stories/epic-15/story-15.4-database-feature-tests.md`
- [ ] Update `docs/getting-started/installation.md`
- [ ] Create `docs/database/MIGRATION_MANAGER_REMOVAL.md`

### Scripts
- [ ] Create `scripts/cleanup_schema_migrations.py`

### Verification
- [ ] All tests pass
- [ ] No MigrationManager references
- [ ] Type checking passes
- [ ] Linting passes
- [ ] Documentation accurate
