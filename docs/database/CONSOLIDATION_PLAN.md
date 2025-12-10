# Database Migration Consolidation Plan

**Date:** 2025-12-09  
**Author:** Development Team  
**Status:** Pending Approval

---

## Goals

1. **Single Migration System**: Use Alembic exclusively, remove custom `MigrationManager`
2. **Standardized Environment Variables**: Consistent naming matching `DatabaseConfig` expectations
3. **All Database Tests Passing**: Fix 31+ failing tests by creating proper fixtures
4. **Clear Documentation**: Update all docs to reflect new standards

---

## Phase 1: Preparation (Story 16.1) ✅

- [x] Complete audit of both migration systems
- [x] Document all environment variables
- [x] Identify all test dependencies
- [x] Create consolidation plan
- [ ] Get team approval for plan
- [ ] Notify team of upcoming changes

---

## Phase 2: Environment Variables (Story 16.2)

**Estimated Effort**: 3 days  
**Risk Level**: Low (backward compatible with fallback)

### Changes Required

#### 1. Update `.env` File
```bash
# OLD (current)
DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_factory
DATABASE_TEST_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_test

# NEW (standardized)
DB_DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_factory
DB_TEST_DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_test

# OPTIONAL: Keep old names for backward compatibility during transition
DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_factory
DATABASE_TEST_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_test
```

#### 2. Update `DatabaseConfig` (Optional Enhancement)
Add fallback support for backward compatibility:
```python
class DatabaseConfig(BaseSettings):
    database_url: str = Field(
        default=None,
        validation_alias=AliasChoices('db_database_url', 'database_url')
    )
```

#### 3. Update Test Files
- `tests/integration/database/test_pgvector_integration.py`
- `tests/integration/database/test_migration_integration.py`

Change from:
```python
DB_URL = os.getenv("TEST_DATABASE_URL")
```

To:
```python
DB_URL = os.getenv("DB_TEST_DATABASE_URL")
```

#### 4. Update Documentation
- `README.md` - Update environment variable examples
- `docs/getting-started/` - Update setup instructions
- `.env.example` - Create/update with new variable names

### Verification Steps
- [ ] Alembic commands work (`alembic current`, `alembic upgrade head`)
- [ ] Environment variables load correctly
- [ ] No breaking changes for existing deployments (if using fallback)

---

## Phase 3: Test Fixtures (Story 16.3)

**Estimated Effort**: 3 days  
**Risk Level**: Low (additive change)

### Fixtures to Create

#### 1. `db_connection` Fixture
**Location**: `tests/conftest.py`

**Requirements**:
- Use `DB_TEST_DATABASE_URL` environment variable
- Create test database if it doesn't exist
- Run Alembic migrations to set up schema
- Provide database connection/session
- Clean up after tests (drop tables or rollback)
- Support both sync and async tests

**Pseudo-code**:
```python
@pytest.fixture(scope="session")
def db_connection():
    """Provide database connection for tests."""
    # Load test database URL
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

#### 2. `db_service` Fixture (If Needed)
**Location**: `tests/conftest.py`

**Requirements**:
- Wrap `db_connection` for service-level operations
- Provide async database service interface
- May be needed for repository/service integration tests

### Tests to Update
- **31 tests** expecting `db_connection` fixture will now work
- No test code changes needed (fixtures are drop-in replacements)

### Verification Steps
- [ ] All 31 tests expecting `db_connection` pass
- [ ] Fixtures work with both sync and async tests
- [ ] Database schema is properly set up via Alembic
- [ ] Cleanup works correctly (no leftover data)

---

## Phase 4: Test Migration (Story 16.4)

**Estimated Effort**: 3 days  
**Risk Level**: Low (isolated to test suite)

### Tests to Migrate from MigrationManager to Alembic

#### 1. `tests/unit/database/test_migrations.py`
**Current Tests** (3):
- `test_migration_execution_order` - Verify migrations run in order
- `test_migration_idempotency` - Verify migrations can be re-run safely
- `test_get_current_version` - Verify version tracking

**Migration Strategy**:
- Replace `MigrationManager` with Alembic commands
- Use `alembic.command` API for programmatic access
- Test Alembic's built-in functionality instead of custom code

**Example**:
```python
# OLD
manager = MigrationManager(db_service)
version = await manager.get_current_version()

# NEW
from alembic import command
from alembic.config import Config

alembic_cfg = Config("alembic.ini")
# Use alembic.command.current() or alembic.script.ScriptDirectory
```

#### 2. `tests/integration/database/test_migration_integration.py`
**Current Tests** (1):
- `test_real_migration_execution` - End-to-end migration test

**Migration Strategy**:
- Test actual Alembic migration execution
- Verify schema changes are applied correctly
- Test upgrade and downgrade operations

### Verification Steps
- [ ] All 4 migration tests pass with Alembic
- [ ] Tests verify Alembic functionality (not custom code)
- [ ] Integration test covers real migration scenarios
- [ ] No references to `MigrationManager` remain in tests

---

## Phase 5: Cleanup (Story 16.5)

**Estimated Effort**: 2 days  
**Risk Level**: Low (removing unused code)

### Code to Remove

#### 1. Delete `MigrationManager` Implementation
**File**: `rag_factory/database/migrations.py`

**Verification Before Deletion**:
- [ ] No production code imports `MigrationManager`
- [ ] No tests use `MigrationManager` (migrated in Phase 4)
- [ ] Run `grep -r "MigrationManager" rag_factory/ tests/` to confirm

#### 2. Remove Migration-Related Imports
**Files to Check**:
- Any files importing `from rag_factory.database.migrations import MigrationManager`

### Documentation Updates

#### 1. Update `README.md`
- Remove references to custom migration system
- Add Alembic usage instructions
- Update environment variable documentation

#### 2. Create `docs/database/README.md`
**Contents**:
- How to create new migrations (`alembic revision --autogenerate`)
- How to apply migrations (`alembic upgrade head`)
- How to rollback migrations (`alembic downgrade -1`)
- Environment variable configuration
- Testing with migrations

#### 3. Update Developer Documentation
- Migration workflow guide
- Best practices for schema changes
- How to handle migration conflicts

### Final Verification
- [ ] All database tests pass (62 tests)
- [ ] No references to `MigrationManager` in codebase
- [ ] Alembic commands work correctly
- [ ] Documentation is complete and accurate
- [ ] Team is trained on new workflow

---

## Breaking Changes

> [!WARNING]
> **Breaking Changes for Developers**

### 1. Environment Variable Rename
**Impact**: Developers must update their `.env` files

**Before**:
```bash
DATABASE_URL=...
DATABASE_TEST_URL=...
```

**After**:
```bash
DB_DATABASE_URL=...
DB_TEST_DATABASE_URL=...
```

**Mitigation**: 
- Provide `.env.example` with new names
- Add fallback support in `DatabaseConfig` (optional)
- Communicate change in team meeting

### 2. Migration API Change
**Impact**: Any code using `MigrationManager` must switch to Alembic

**Before**:
```python
from rag_factory.database.migrations import MigrationManager
manager = MigrationManager(db_service)
await manager.run_migrations()
```

**After**:
```python
from alembic import command
from alembic.config import Config
alembic_cfg = Config("alembic.ini")
command.upgrade(alembic_cfg, "head")
```

**Mitigation**: 
- Only affects test code (no production usage found)
- Update all tests in Phase 4

### 3. Test Fixture Change
**Impact**: Tests must use new `db_connection` fixture

**Before**:
```python
def test_something(db_connection):  # Fixture didn't exist
    ...
```

**After**:
```python
def test_something(db_connection):  # Fixture now exists and works
    ...
```

**Mitigation**: 
- This is a fix, not a breaking change
- Tests that were failing will now pass

---

## Rollback Plan

### If Issues Arise During Migration

#### Phase 2 Rollback (Environment Variables)
- Revert `.env` changes
- Keep old variable names
- Alembic will fail but system continues working

#### Phase 3 Rollback (Test Fixtures)
- Remove new fixtures from `tests/conftest.py`
- Tests will fail again (same as before)
- No impact on production code

#### Phase 4 Rollback (Test Migration)
- Revert test file changes
- Restore `MigrationManager` usage in tests
- Tests will pass with old system

#### Phase 5 Rollback (Cleanup)
- Restore `rag_factory/database/migrations.py` from git
- Restore deleted imports
- System returns to dual-migration state

### Emergency Rollback
If critical issues are discovered:
1. Revert all changes via git
2. Keep Alembic migrations (they are source of truth)
3. Temporarily restore `MigrationManager` if needed
4. Environment variables can be reverted independently

**Note**: Alembic migrations should **never** be rolled back in production. They are the source of truth for schema changes.

---

## Timeline

| Phase | Story | Effort | Dependencies | Start | End |
|-------|-------|--------|--------------|-------|-----|
| 1 | 16.1 - Audit | 2 days | None | Day 1 | Day 2 |
| 2 | 16.2 - Env Vars | 3 days | 16.1 complete | Day 3 | Day 5 |
| 3 | 16.3 - Fixtures | 3 days | 16.2 complete | Day 6 | Day 8 |
| 4 | 16.4 - Test Migration | 3 days | 16.3 complete | Day 9 | Day 11 |
| 5 | 16.5 - Cleanup | 2 days | 16.4 complete | Day 12 | Day 13 |

**Total Duration**: ~2.5 weeks (13 working days)

**Buffer**: Add 20% buffer for unexpected issues = **3 weeks total**

---

## Success Criteria

### Technical Metrics
- [ ] All 62 database tests passing
- [ ] Zero references to `MigrationManager` in codebase
- [ ] Alembic commands execute successfully
- [ ] Environment variables standardized
- [ ] Code coverage maintained or improved

### Documentation Metrics
- [ ] Migration guide created
- [ ] Environment variable docs updated
- [ ] Developer workflow documented
- [ ] Team trained on new process

### Quality Metrics
- [ ] No regressions in existing functionality
- [ ] Type checking passes
- [ ] Linting passes
- [ ] CI/CD pipeline green

---

## Communication Plan

### Before Starting
- [ ] Present plan to team
- [ ] Get approval from tech lead
- [ ] Schedule team training session
- [ ] Create migration checklist for developers

### During Migration
- [ ] Daily updates in team standup
- [ ] Document any issues encountered
- [ ] Update plan if timeline changes
- [ ] Keep stakeholders informed

### After Completion
- [ ] Announce completion to team
- [ ] Conduct training session
- [ ] Update onboarding documentation
- [ ] Retrospective on migration process

---

## Risk Assessment

### Low Risk ✅
- Environment variable rename (backward compatible)
- Creating test fixtures (additive)
- Updating test files (isolated)
- Documentation updates (no code impact)

### Medium Risk ⚠️
- Removing `MigrationManager` (ensure no hidden dependencies)
- Alembic configuration changes (test thoroughly)

### High Risk 🔴
- None identified

**Overall Risk**: **LOW** - This is primarily a cleanup and standardization effort with minimal production impact.

---

## Approval

**Reviewed By**: _________________  
**Approved By**: _________________  
**Date**: _________________  

**Approval Status**: ⏳ Pending

---

## References

- [MIGRATION_AUDIT.md](./MIGRATION_AUDIT.md) - Detailed audit findings
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [Epic 16 - Database Migration System Consolidation](../epics/epic-16-database-consolidation.md)
