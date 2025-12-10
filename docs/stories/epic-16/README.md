# Epic 16: Database Migration System Consolidation - Stories

This directory contains the individual story documents for Epic 16, which consolidates database migration management to use Alembic exclusively.

## Overview

**Epic Goal:** Consolidate database migration management to use Alembic exclusively, removing the custom `MigrationManager` implementation and establishing clear environment variable standards for database connections across development, testing, and production environments.

**Total Story Points:** 23

## Stories

### Story 16.1: Audit and Document Migration Systems (3 points)
**File:** [story-16.1-audit-migration-systems.md](story-16.1-audit-migration-systems.md)

Comprehensive audit of both migration systems (Alembic and custom MigrationManager), environment variables, and test dependencies. Creates detailed documentation and consolidation plan.

**Key Deliverables:**
- `MIGRATION_AUDIT.md` - Complete audit of current state
- `CONSOLIDATION_PLAN.md` - Step-by-step migration plan
- Environment variable matrix
- Test dependency map

---

### Story 16.2: Standardize Environment Variables (5 points)
**File:** [story-16.2-standardize-environment-variables.md](story-16.2-standardize-environment-variables.md)

Standardizes environment variable naming across all environments, fixing the mismatch between `DATABASE_TEST_URL` and `TEST_DATABASE_URL`.

**Key Changes:**
- Rename `DATABASE_TEST_URL` → `TEST_DATABASE_URL`
- Create `.env.example` and `tests/.env.test` templates
- Implement environment variable validation
- Add backward compatibility with deprecation warnings

**Breaking Changes:**
- Environment variable rename (with backward compatibility)

---

### Story 16.3: Create Database Connection Fixtures (5 points)
**File:** [story-16.3-create-database-fixtures.md](story-16.3-create-database-fixtures.md)

Creates proper pytest fixtures for database connections, fixing 57 failing database tests.

**Key Deliverables:**
- `db_connection` fixture with transaction rollback
- `db_service` fixture for async tests
- `test_db_url` fixture for environment configuration
- Test database setup script
- `tests/README.md` documentation

**Impact:**
- Fixes 57 failing database tests
- Enables proper test isolation
- Supports both sync and async tests

---

### Story 16.4: Migrate Tests to Alembic (5 points)
**File:** [story-16.4-migrate-tests-to-alembic.md](story-16.4-migrate-tests-to-alembic.md)

Updates all migration tests to use Alembic instead of custom MigrationManager, ensuring tests validate the production migration system.

**Key Changes:**
- Update `tests/unit/database/test_migrations.py`
- Update `tests/integration/database/test_migration_integration.py`
- Remove all `MigrationManager` imports from tests
- Create Alembic test utilities

**Benefits:**
- Tests validate actual production system
- Better test coverage
- Alembic API usage examples

---

### Story 16.5: Remove Custom MigrationManager (3 points)
**File:** [story-16.5-remove-migration-manager.md](story-16.5-remove-migration-manager.md)

Removes the custom MigrationManager code and updates all documentation to reference only Alembic.

**Key Changes:**
- Delete `rag_factory/database/migrations.py`
- Update all documentation
- Create migration guide for users
- Optional schema cleanup script

**Impact:**
- Single migration system (Alembic only)
- Reduced maintenance burden
- Clearer documentation

---

### Story 16.6: Update Database Documentation (2 points)
**File:** [story-16.6-update-database-documentation.md](story-16.6-update-database-documentation.md)

Updates `docs/database/README.md` to be consistent with Epic 16 decisions, removing all MigrationManager references and adding comprehensive sections for environment variables, testing, and Alembic usage.

**Key Changes:**
- Remove all MigrationManager references
- Add Environment Variables section
- Add Testing section with fixture documentation
- Update Migrations section (Alembic only)
- Add References section

**Impact:**
- Accurate, up-to-date documentation
- Clear guidance for developers
- Consistent with consolidated codebase


---

## Implementation Order

The stories must be implemented in order due to dependencies:

```
16.1 (Audit)
  ↓
16.2 (Environment Variables)
  ↓
16.3 (Database Fixtures)
  ↓
16.4 (Migrate Tests)
  ↓
16.5 (Remove MigrationManager)
  ↓
16.6 (Update Documentation)
```

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

## Key Benefits

1. **Single Source of Truth**: Alembic is the only migration system
2. **Industry Standard**: Using battle-tested Alembic instead of custom code
3. **Better Testing**: Tests validate the actual production migration system
4. **Clearer Configuration**: Standardized environment variable names
5. **Fixed Tests**: 57 database tests now pass
6. **Less Maintenance**: One system to maintain instead of two
7. **Better Documentation**: Clear, accurate documentation

## Breaking Changes

### Environment Variables
- `DATABASE_TEST_URL` renamed to `TEST_DATABASE_URL`
- Backward compatibility provided with deprecation warnings

### Migration API
- `MigrationManager` class removed
- Users must use Alembic CLI commands instead

## Migration Guide

See [Epic 16 Documentation](../../epics/epic-16-database-consolidation.md) for complete migration guide and user-facing documentation.

## Related Documentation

- [Epic 16: Database Migration System Consolidation](../../epics/epic-16-database-consolidation.md)
- [Database README](../../database/README.md)
- [Getting Started Guide](../../getting-started/installation.md)
- [Epic 15: Test Coverage Improvements](../epic-15/)

## Timeline

**Estimated Duration:** 2-3 weeks

- Week 1: Stories 16.1 and 16.2 (Audit and Environment Variables)
- Week 2: Stories 16.3 and 16.4 (Fixtures and Test Migration)
- Week 3: Story 16.5 (Cleanup and Documentation)

## Notes

- This epic addresses technical debt from Epic 15 (Test Coverage Improvements)
- The custom MigrationManager was created for Story 15.4 but is redundant with Alembic
- 57 database tests are currently failing due to missing fixtures
- Environment variable inconsistency affects both development and CI/CD
- This consolidation improves maintainability and reduces confusion
