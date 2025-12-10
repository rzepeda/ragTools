# Story 16.6: Update Database Documentation

**Story ID:** 16.6  
**Epic:** Epic 16 - Database Migration System Consolidation  
**Story Points:** 2  
**Priority:** Medium  
**Dependencies:** Story 16.5 (MigrationManager removed)

---

## User Story

**As a** developer  
**I want** the database documentation updated to reflect Epic 16 decisions  
**So that** documentation is accurate and consistent with the consolidated migration system

---

## Detailed Requirements

### Functional Requirements

> [!NOTE]
> **Current State**: `docs/database/README.md` contains outdated information
> - References to custom MigrationManager
> - Inconsistent environment variable names
> - Missing information about test fixtures
> - Outdated migration instructions
> 
> **Target State**: Documentation reflects Alembic-only approach with standardized configuration

1. **Update Migration Section**
   - Remove all MigrationManager references
   - Expand Alembic documentation
   - Add clear examples for common operations
   - Document migration best practices

2. **Update Environment Variables Section**
   - Add dedicated environment variables section
   - Document `DATABASE_URL` and `TEST_DATABASE_URL`
   - Include examples for different environments (dev, test, prod)
   - Document VM development setup

3. **Add Testing Section**
   - Document test database setup
   - Explain test fixtures (`db_connection`, `db_service`)
   - Include examples of using fixtures
   - Link to `tests/README.md`

4. **Update Quick Start Guide**
   - Update setup instructions
   - Use standardized environment variables
   - Include Alembic migration commands
   - Add troubleshooting section

5. **Add References Section**
   - Link to Epic 16 documentation
   - Link to Alembic documentation
   - Link to test documentation
   - Link to environment variable guide

### Non-Functional Requirements

1. **Clarity**
   - Clear, concise explanations
   - Good examples
   - Proper formatting

2. **Completeness**
   - Cover all common use cases
   - Include troubleshooting
   - Link to related documentation

3. **Accuracy**
   - All information current
   - No outdated references
   - Consistent with codebase

---

## Acceptance Criteria

### AC1: Migration Section Updated
- [ ] All MigrationManager references removed
- [ ] Alembic commands documented with examples
- [ ] Migration creation process explained
- [ ] Rollback procedures documented
- [ ] Best practices included

### AC2: Environment Variables Section Added
- [ ] New "Environment Variables" section created
- [ ] `DATABASE_URL` documented with examples
- [ ] `TEST_DATABASE_URL` documented with examples
- [ ] VM development setup explained
- [ ] Examples for local, Docker, and cloud setups

### AC3: Testing Section Added
- [ ] New "Testing" section created
- [ ] Test database setup documented
- [ ] Fixtures explained (`db_connection`, `db_service`)
- [ ] Usage examples provided
- [ ] Link to `tests/README.md`

### AC4: Quick Start Updated
- [ ] Setup instructions use standard variable names
- [ ] Alembic commands included
- [ ] Step-by-step guide clear
- [ ] Troubleshooting section added

### AC5: References Section Added
- [ ] Links to Epic 16 documentation
- [ ] Links to Alembic docs
- [ ] Links to test documentation
- [ ] Links to related guides

### AC6: Quality Checks
- [ ] No broken links
- [ ] Proper markdown formatting
- [ ] Code examples tested
- [ ] Consistent terminology

---

## Technical Specifications

### Sections to Update/Add

#### 1. Environment Variables (NEW SECTION)

Add after the "Overview" section:

```markdown
## Environment Variables

### Required Variables

#### Production
```bash
# Main database connection
DATABASE_URL=postgresql://user:password@host:5432/database_name
```

#### Testing
```bash
# Test database connection
TEST_DATABASE_URL=postgresql://user:password@host:5432/test_database
```

### Configuration Examples

#### Local Development
```bash
# .env file
DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_factory
TEST_DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_test
```

#### VM Development (Accessing Host Services)
```bash
# .env file
HOST_IP=192.168.56.1
DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_factory
TEST_DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_test
```

#### Docker Compose
```bash
# .env file
DATABASE_URL=postgresql://rag_user:rag_password@postgres:5432/rag_factory
TEST_DATABASE_URL=postgresql://rag_user:rag_password@postgres:5432/rag_test
```

#### Cloud (Neon, Supabase, etc.)
```bash
# .env file
DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require
TEST_DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/test_db?sslmode=require
```

### Environment Variable Reference

| Variable | Purpose | Required | Example |
|----------|---------|----------|---------|
| `DATABASE_URL` | Main database connection | Yes | `postgresql://localhost/db` |
| `TEST_DATABASE_URL` | Test database connection | For tests | `postgresql://localhost/test_db` |
| `HOST_IP` | VM host machine IP | VM only | `192.168.56.1` |

See [Environment Variables Guide](ENVIRONMENT_VARIABLES.md) for complete reference.
```

#### 2. Migrations Section (UPDATE)

Replace existing migration section with:

```markdown
## Migrations

The project uses [Alembic](https://alembic.sqlalchemy.org/) for database schema migrations.

### Running Migrations

```bash
# Upgrade to latest version
alembic upgrade head

# Upgrade one version
alembic upgrade +1

# Downgrade one version
alembic downgrade -1

# Show current version
alembic current

# Show migration history
alembic history --verbose
```

### Creating Migrations

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "add user table"

# Create empty migration for manual changes
alembic revision -m "add custom index"
```

### Migration Best Practices

1. **Review Auto-Generated Migrations**
   - Always review migrations before applying
   - Auto-generation may miss some changes
   - Add custom logic as needed

2. **Test Migrations**
   ```bash
   # Test upgrade
   alembic upgrade head
   
   # Test downgrade
   alembic downgrade -1
   
   # Test upgrade again
   alembic upgrade head
   ```

3. **Use Transactions**
   - Migrations run in transactions by default
   - Failed migrations automatically rollback
   - Use `op.execute()` for raw SQL

4. **Handle Data Migrations**
   - Separate schema and data migrations when possible
   - Use `op.bulk_insert()` for data
   - Consider using `op.execute()` for complex data transformations

### Rollback

```bash
# Rollback to specific version
alembic downgrade <revision_id>

# Rollback all migrations
alembic downgrade base

# Rollback to previous version
alembic downgrade -1
```

### Troubleshooting

**Issue: "Target database is not up to date"**
```bash
# Check current version
alembic current

# Stamp database to specific version
alembic stamp head
```

**Issue: "Can't locate revision identified by 'xxx'"**
```bash
# Reset to base and reapply
alembic downgrade base
alembic upgrade head
```

See [Alembic Documentation](https://alembic.sqlalchemy.org/) for more details.
```

#### 3. Testing Section (NEW SECTION)

Add before "Performance Optimization":

```markdown
## Testing

### Test Database Setup

#### 1. Set Environment Variable

```bash
# In .env or export
export TEST_DATABASE_URL="postgresql://user:password@localhost:5432/rag_test"
```

#### 2. Create Test Database

```bash
# Using setup script
python tests/setup_test_db.py

# Or manually
createdb rag_test
psql rag_test -c "CREATE EXTENSION IF NOT EXISTS vector"
```

#### 3. Run Migrations

```bash
# Set Alembic to use test database
alembic -x test=true upgrade head

# Or use TEST_DATABASE_URL
DATABASE_URL=$TEST_DATABASE_URL alembic upgrade head
```

### Running Tests

```bash
# All database tests
pytest tests/unit/database/ tests/integration/database/ -v

# Specific test file
pytest tests/unit/database/test_models.py -v

# With coverage
pytest tests/unit/database/ --cov=rag_factory.database --cov-report=html
```

### Test Fixtures

The project provides several pytest fixtures for database testing:

#### `db_connection` - Sync Database Session

```python
def test_create_document(db_connection):
    """Test creating a document."""
    from rag_factory.database.models import Document
    
    doc = Document(
        filename="test.txt",
        source_path="/path/to/test.txt",
        content_hash="abc123"
    )
    
    db_connection.add(doc)
    db_connection.commit()
    
    # Verify
    result = db_connection.query(Document).filter_by(filename="test.txt").first()
    assert result is not None
```

#### `db_service` - Async Database Service

```python
@pytest.mark.asyncio
async def test_store_chunks(db_service):
    """Test storing chunks."""
    chunks = [
        {"id": "1", "text": "test", "embedding": [0.1, 0.2]}
    ]
    await db_service.store_chunks(chunks)
    
    results = await db_service.search_chunks([0.1, 0.2], top_k=1)
    assert len(results) == 1
```

#### `clean_database` - Empty Database

```python
def test_with_clean_db(clean_database):
    """Test with guaranteed empty database."""
    from rag_factory.database.models import Document
    
    # Database is empty
    count = clean_database.query(Document).count()
    assert count == 0
```

See [tests/README.md](../../tests/README.md) for complete fixture documentation.

### Integration Tests

Integration tests require a real PostgreSQL database with pgvector:

```bash
# Set test database URL
export TEST_DATABASE_URL="postgresql://localhost/rag_test"

# Run integration tests
pytest tests/integration/database/ -v -m integration
```
```

#### 4. Quick Start Section (UPDATE)

Update the quick start to use standard variables:

```markdown
## Quick Start

### 1. Install PostgreSQL with pgvector

#### Option A: Docker (Recommended)
```bash
docker run -d \
  --name rag-postgres \
  -e POSTGRES_USER=rag_user \
  -e POSTGRES_PASSWORD=rag_password \
  -e POSTGRES_DB=rag_factory \
  -p 5432:5432 \
  ankane/pgvector
```

#### Option B: Local Installation
```bash
# Ubuntu/Debian
sudo apt-get install postgresql-15 postgresql-15-pgvector

# macOS
brew install postgresql@15 pgvector
```

### 2. Configure Environment

Create `.env` file:
```bash
DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_factory
TEST_DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_test
```

### 3. Run Migrations

```bash
# Upgrade to latest schema
alembic upgrade head
```

### 4. Verify Setup

```bash
# Test connection
python -c "from rag_factory.database.connection import DatabaseConnection; print('OK')"

# Run database tests
pytest tests/unit/database/ -v
```
```

#### 5. References Section (NEW SECTION)

Add at the end:

```markdown
## References

### Project Documentation
- [Epic 16: Database Migration System Consolidation](../epics/epic-16-database-consolidation.md)
- [Environment Variables Guide](ENVIRONMENT_VARIABLES.md)
- [Test Documentation](../../tests/README.md)
- [Getting Started Guide](../getting-started/installation.md)

### External Documentation
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

### Related Epics
- [Epic 2: Database & Storage Infrastructure](../epics/epic-02-database-storage.md)
- [Epic 11: Dependency Injection](../epics/epic-11-dependency-injection.md)
- [Epic 15: Test Coverage Improvements](../epics/epic-15-test-coverage-improvements/)
```

### Content to Remove

1. **Remove all MigrationManager references:**
   - Any mentions of `rag_factory.database.migrations`
   - Custom migration manager examples
   - `schema_migrations` table references

2. **Remove outdated variable names:**
   - `DATABASE_TEST_URL` (replaced with `TEST_DATABASE_URL`)
   - Any inconsistent naming

3. **Remove redundant sections:**
   - Duplicate migration instructions
   - Outdated setup procedures

---

## Testing Strategy

### Documentation Review

```bash
# Check for broken links
markdown-link-check docs/database/README.md

# Check for MigrationManager references
grep -i "migrationmanager" docs/database/README.md
# Should return no results

# Check for old variable names
grep "DATABASE_TEST_URL" docs/database/README.md
# Should return no results

# Verify new variable names
grep "TEST_DATABASE_URL" docs/database/README.md
# Should return multiple results
```

### Code Example Verification

Test all code examples in the documentation:

```bash
# Test environment variable setup
export DATABASE_URL="postgresql://localhost/test"
export TEST_DATABASE_URL="postgresql://localhost/test"

# Test Alembic commands
alembic current
alembic history

# Test Python imports
python -c "from rag_factory.database.connection import DatabaseConnection"
```

---

## Definition of Done

- [ ] All MigrationManager references removed
- [ ] Environment Variables section added
- [ ] Testing section added
- [ ] Migrations section updated with Alembic only
- [ ] Quick Start section updated
- [ ] References section added
- [ ] All code examples tested
- [ ] No broken links
- [ ] Consistent terminology throughout
- [ ] Proper markdown formatting
- [ ] PR approved and merged

---

## Notes

- This story completes the documentation updates for Epic 16
- Should be done **after** Story 16.5 (MigrationManager removed)
- Ensures documentation matches the consolidated codebase
- Provides clear guidance for new developers
- Links to all related documentation

---

## Documentation Quality Checklist

### Content
- [ ] Accurate and up-to-date
- [ ] Clear and concise
- [ ] Complete coverage of topics
- [ ] Good examples provided
- [ ] Troubleshooting included

### Structure
- [ ] Logical organization
- [ ] Clear headings
- [ ] Proper nesting
- [ ] Good flow

### Formatting
- [ ] Proper markdown syntax
- [ ] Code blocks with language tags
- [ ] Tables formatted correctly
- [ ] Lists properly indented
- [ ] Links working

### Usability
- [ ] Easy to navigate
- [ ] Quick start guide clear
- [ ] Examples copy-pasteable
- [ ] References helpful
- [ ] Searchable content

---

## Impact

- **Before**: Outdated documentation with MigrationManager references
- **After**: Accurate, comprehensive documentation for Alembic-only system
- **Benefit**: Developers have clear, correct guidance
- **Maintenance**: Easier to keep documentation current
