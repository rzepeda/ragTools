# MigrationManager Removal Guide

## What Changed

The custom `MigrationManager` class has been removed. The project now uses **Alembic exclusively** for database migrations.

## Impact

### If you were using MigrationManager in your code:

**Before:**
```python
async def example():
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

## Alembic Quick Reference

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

### Programmatic Usage

```python
from alembic import command
from alembic.config import Config

# Configure Alembic
config = Config("alembic.ini")
config.set_main_option("sqlalchemy.url", database_url)

# Run migrations
command.upgrade(config, "head")

# Downgrade
command.downgrade(config, "-1")

# Get current revision
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine

engine = create_engine(database_url)
with engine.connect() as conn:
    context = MigrationContext.configure(conn)
    current_rev = context.get_current_revision()
    print(f"Current revision: {current_rev}")
```

## Need Help?

See:
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [docs/database/README.md](README.md)
- [docs/epics/epic-16-database-consolidation.md](../epics/epic-16-database-consolidation.md)
- [Story 16.5 Documentation](../stories/Completed/epic-16/story-16.5-remove-migration-manager.md)
