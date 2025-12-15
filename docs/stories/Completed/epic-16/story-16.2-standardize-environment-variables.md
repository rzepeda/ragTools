# Story 16.2: Standardize Environment Variables

**Story ID:** 16.2  
**Epic:** Epic 16 - Database Migration System Consolidation  
**Story Points:** 5  
**Priority:** High  
**Dependencies:** Story 16.1 (Audit complete)

---

## User Story

**As a** developer  
**I want** consistent environment variable naming across all environments  
**So that** configuration is predictable and tests work correctly

---

## Detailed Requirements

### Functional Requirements

> [!WARNING]
> **Breaking Change**: Environment variable `DATABASE_TEST_URL` will be renamed to `TEST_DATABASE_URL`
> 
> **Impact**: Users must update their `.env` files and CI/CD configuration

1. **Standardize Variable Names**
   - Rename `DATABASE_TEST_URL` to `TEST_DATABASE_URL`
   - Keep `DATABASE_URL` as-is (already standard)
   - Remove any deprecated or unused variables
   - Document all standard variables

2. **Update Configuration Files**
   - Update `.env` with new variable names
   - Update `.env.example` as template for users
   - Create `tests/.env.test` template for test configuration
   - Update `alembic.ini` if needed
   - Update `migrations/env.py` to support both variables during transition

3. **Update Code References**
   - Update all Python files reading `DATABASE_TEST_URL`
   - Update test files to use `TEST_DATABASE_URL`
   - Update CI/CD configuration files
   - Update Docker Compose files (if any)

4. **Add Validation**
   - Validate required environment variables on startup
   - Provide helpful error messages for missing variables
   - Warn about deprecated variable names
   - Support graceful fallback during transition period

### Non-Functional Requirements

1. **Backward Compatibility**
   - Support both old and new variable names during transition
   - Warn users about deprecated names
   - Document migration path

2. **Documentation**
   - Clear documentation of all variables
   - Examples for different environments
   - Migration guide for users

---

## Acceptance Criteria

### AC1: Environment File Updates
- [ ] `.env` updated with `TEST_DATABASE_URL`
- [ ] `.env` removes `DATABASE_TEST_URL`
- [ ] `.env.example` created with all standard variables
- [ ] `tests/.env.test` template created
- [ ] All files have clear comments explaining each variable

### AC2: Code Updates
- [ ] All Python files updated to use `TEST_DATABASE_URL`
- [ ] `migrations/env.py` supports both variable names
- [ ] Test files use `TEST_DATABASE_URL`
- [ ] No references to `DATABASE_TEST_URL` remain (except deprecation warnings)

### AC3: Validation Implementation
- [ ] Environment variable validation function created
- [ ] Validation runs on application startup
- [ ] Clear error messages for missing variables
- [ ] Deprecation warnings for old variable names

### AC4: Documentation
- [ ] `docs/database/ENVIRONMENT_VARIABLES.md` created
- [ ] All variables documented with purpose and examples
- [ ] Migration guide for users included
- [ ] `docs/database/README.md` updated

### AC5: CI/CD Updates
- [ ] GitHub Actions workflows updated (if any)
- [ ] Docker Compose files updated (if any)
- [ ] Any other CI/CD configuration updated

### AC6: Testing
- [ ] All tests pass with new variable names
- [ ] Backward compatibility tested
- [ ] Deprecation warnings tested
- [ ] Validation error messages tested

---

## Technical Specifications

### Environment Variable Standards

#### Production Variables

```bash
# Main database connection
DATABASE_URL=postgresql://user:password@host:5432/database_name

# Optional: Read replica
DATABASE_READ_URL=postgresql://user:password@read-host:5432/database_name
```

#### Test Variables

```bash
# Test database connection (STANDARD NAME)
TEST_DATABASE_URL=postgresql://user:password@host:5432/test_database

# Deprecated (will be removed in future version)
# DATABASE_TEST_URL=postgresql://...  # DO NOT USE
```

#### Development Variables

```bash
# Local development
DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_factory
TEST_DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_test

# VM development (accessing host machine)
HOST_IP=192.168.56.1
DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_factory
TEST_DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_test
```

#### Test Execution Flags

```bash
# Enable/disable test categories
RUN_DB_TESTS=true
RUN_INTEGRATION_TESTS=true
RUN_LLM_TESTS=false  # Requires API keys
```

### File Templates

#### `.env` (Development)

```bash
# =============================================================================
# RAG Factory - Development Configuration
# =============================================================================

# Host IP Configuration (for VM development)
# Common values:
#   - 10.0.2.2        (VirtualBox NAT)
#   - 192.168.56.1    (VirtualBox Host-Only)
#   - 192.168.122.1   (KVM/QEMU)
#   - 172.16.0.1      (VMware NAT)
HOST_IP=192.168.56.1

# =============================================================================
# Database Configuration
# =============================================================================

# Main database URL
DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_factory

# Test database URL
TEST_DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_test

# =============================================================================
# Neo4j Configuration (Optional)
# =============================================================================

NEO4J_URI=bolt://${HOST_IP}:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=rag_password

# =============================================================================
# LLM Configuration (Optional)
# =============================================================================

# LM Studio (local)
LM_STUDIO_BASE_URL=http://${HOST_IP}:1234/v1
LM_STUDIO_MODEL=local-model

# OpenAI (cloud)
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4

# Cohere (cloud)
# COHERE_API_KEY=...

# =============================================================================
# Test Configuration
# =============================================================================

# Enable all tests for local development
RUN_DB_TESTS=true
RUN_LLM_TESTS=true
RUN_INTEGRATION_TESTS=true

# =============================================================================
# Development Settings
# =============================================================================

LOG_LEVEL=INFO
```

#### `.env.example` (Template for Users)

```bash
# =============================================================================
# RAG Factory - Environment Configuration Template
# =============================================================================
# Copy this file to .env and update with your configuration

# =============================================================================
# Database Configuration
# =============================================================================

# Main database URL (required)
DATABASE_URL=postgresql://user:password@localhost:5432/rag_factory

# Test database URL (required for running tests)
TEST_DATABASE_URL=postgresql://user:password@localhost:5432/rag_test

# =============================================================================
# Optional: VM Development
# =============================================================================
# If running in a VM and accessing services on the host machine:
# HOST_IP=192.168.56.1
# DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_factory
# TEST_DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_test

# =============================================================================
# Neo4j Configuration (Optional)
# =============================================================================

# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=password

# =============================================================================
# LLM Configuration (Optional)
# =============================================================================

# OpenAI
# OPENAI_API_KEY=sk-your-key-here
# OPENAI_MODEL=gpt-4

# Cohere
# COHERE_API_KEY=your-key-here

# Local LM Studio
# LM_STUDIO_BASE_URL=http://localhost:1234/v1
# LM_STUDIO_MODEL=local-model

# =============================================================================
# Test Execution Flags
# =============================================================================

# Enable/disable test categories
RUN_DB_TESTS=true
RUN_INTEGRATION_TESTS=true
RUN_LLM_TESTS=false  # Set to true if you have API keys configured

# =============================================================================
# Development Settings
# =============================================================================

LOG_LEVEL=INFO
```

#### `tests/.env.test` (Test Configuration Template)

```bash
# =============================================================================
# RAG Factory - Test Environment Configuration
# =============================================================================
# Copy this file to tests/.env.test and update with your test database credentials

# =============================================================================
# Test Database Configuration
# =============================================================================

# Test database URL (required)
TEST_DATABASE_URL=postgresql://test_user:test_password@localhost:5432/rag_test

# Alternative: Individual connection parameters
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=rag_test
# DB_USER=test_user
# DB_PASSWORD=test_password

# =============================================================================
# Test Execution Flags
# =============================================================================

# Database tests (requires PostgreSQL with pgvector)
RUN_DB_TESTS=true

# Integration tests (requires all services)
RUN_INTEGRATION_TESTS=true

# LLM tests (requires API keys)
RUN_LLM_TESTS=false

# =============================================================================
# Optional: Test Service Configuration
# =============================================================================

# Neo4j for graph tests
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=test_password

# API keys for integration tests
# OPENAI_API_KEY=sk-test-key
# COHERE_API_KEY=test-key
```

### Environment Variable Validation

```python
# rag_factory/config/env_validator.py

import os
import warnings
from typing import Dict, List, Optional

class EnvironmentValidator:
    """Validates and manages environment variables."""
    
    # Required variables for different modes
    REQUIRED_VARS = {
        'production': ['DATABASE_URL'],
        'development': ['DATABASE_URL'],
        'test': ['TEST_DATABASE_URL'],
    }
    
    # Deprecated variables and their replacements
    DEPRECATED_VARS = {
        'DATABASE_TEST_URL': 'TEST_DATABASE_URL',
    }
    
    @classmethod
    def validate(cls, mode: str = 'development') -> None:
        """
        Validate environment variables for the given mode.
        
        Args:
            mode: Environment mode (production, development, test)
            
        Raises:
            ValueError: If required variables are missing
        """
        missing = cls._check_required(mode)
        if missing:
            raise ValueError(
                f"Missing required environment variables for {mode} mode: {', '.join(missing)}\n"
                f"Please set these variables in your .env file or environment."
            )
        
        cls._check_deprecated()
    
    @classmethod
    def _check_required(cls, mode: str) -> List[str]:
        """Check for missing required variables."""
        required = cls.REQUIRED_VARS.get(mode, [])
        missing = [var for var in required if not os.getenv(var)]
        return missing
    
    @classmethod
    def _check_deprecated(cls) -> None:
        """Warn about deprecated variables."""
        for old_var, new_var in cls.DEPRECATED_VARS.items():
            if os.getenv(old_var):
                warnings.warn(
                    f"Environment variable '{old_var}' is deprecated. "
                    f"Please use '{new_var}' instead. "
                    f"Support for '{old_var}' will be removed in a future version.",
                    DeprecationWarning,
                    stacklevel=2
                )
    
    @classmethod
    def get_database_url(cls, for_tests: bool = False) -> Optional[str]:
        """
        Get database URL with backward compatibility.
        
        Args:
            for_tests: If True, get test database URL
            
        Returns:
            Database URL or None if not set
        """
        if for_tests:
            # Try new name first, fall back to old name
            url = os.getenv('TEST_DATABASE_URL') or os.getenv('DATABASE_TEST_URL')
            if os.getenv('DATABASE_TEST_URL'):
                warnings.warn(
                    "Using deprecated 'DATABASE_TEST_URL'. Please rename to 'TEST_DATABASE_URL'.",
                    DeprecationWarning,
                    stacklevel=2
                )
            return url
        else:
            return os.getenv('DATABASE_URL')

# Usage in application startup
from rag_factory.config.env_validator import EnvironmentValidator

# In main application
EnvironmentValidator.validate(mode='production')

# In tests
EnvironmentValidator.validate(mode='test')
```

### Code Update Locations

**Files to Update:**

1. **Test Configuration**
   - `tests/conftest.py` - Update fixture to use `TEST_DATABASE_URL`
   - `tests/integration/database/test_pgvector_integration.py` - Line 8
   - `tests/integration/database/test_migration_integration.py` - Line 9

2. **Migration Configuration**
   - `migrations/env.py` - Support both variables with deprecation warning

3. **Database Configuration**
   - `rag_factory/database/config.py` - Add validation and backward compatibility

4. **Documentation**
   - `docs/database/README.md` - Update environment variable section
   - `docs/getting-started/installation.md` - Update setup instructions
   - `README.md` - Update quick start guide

### Migration Commands

```bash
# Update .env file
sed -i 's/DATABASE_TEST_URL=/TEST_DATABASE_URL=/g' .env

# Find all code references to update
grep -r "DATABASE_TEST_URL" rag_factory/ tests/ --include="*.py"

# Update test files
find tests/ -name "*.py" -exec sed -i 's/DATABASE_TEST_URL/TEST_DATABASE_URL/g' {} \;

# Verify no old references remain (except in deprecation handling)
grep -r "DATABASE_TEST_URL" rag_factory/ tests/ --include="*.py" | grep -v "deprecated" | grep -v "DEPRECATED"
```

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/config/test_env_validator.py

import pytest
import os
from rag_factory.config.env_validator import EnvironmentValidator

class TestEnvironmentValidator:
    """Test environment variable validation."""
    
    def test_validate_production_success(self, monkeypatch):
        """Test validation passes with required variables."""
        monkeypatch.setenv('DATABASE_URL', 'postgresql://localhost/db')
        EnvironmentValidator.validate(mode='production')  # Should not raise
    
    def test_validate_production_missing(self, monkeypatch):
        """Test validation fails with missing variables."""
        monkeypatch.delenv('DATABASE_URL', raising=False)
        with pytest.raises(ValueError, match="Missing required environment variables"):
            EnvironmentValidator.validate(mode='production')
    
    def test_deprecated_warning(self, monkeypatch):
        """Test deprecation warning for old variable names."""
        monkeypatch.setenv('DATABASE_TEST_URL', 'postgresql://localhost/test')
        
        with pytest.warns(DeprecationWarning, match="DATABASE_TEST_URL.*deprecated"):
            EnvironmentValidator._check_deprecated()
    
    def test_get_database_url_backward_compatible(self, monkeypatch):
        """Test backward compatibility for test database URL."""
        # Old variable name should still work
        monkeypatch.setenv('DATABASE_TEST_URL', 'postgresql://localhost/old')
        monkeypatch.delenv('TEST_DATABASE_URL', raising=False)
        
        with pytest.warns(DeprecationWarning):
            url = EnvironmentValidator.get_database_url(for_tests=True)
        
        assert url == 'postgresql://localhost/old'
    
    def test_get_database_url_prefers_new_name(self, monkeypatch):
        """Test new variable name takes precedence."""
        monkeypatch.setenv('TEST_DATABASE_URL', 'postgresql://localhost/new')
        monkeypatch.setenv('DATABASE_TEST_URL', 'postgresql://localhost/old')
        
        url = EnvironmentValidator.get_database_url(for_tests=True)
        assert url == 'postgresql://localhost/new'
```

### Integration Tests

```bash
# Test with new variable name
export TEST_DATABASE_URL="postgresql://localhost/test"
pytest tests/integration/database/ -v

# Test backward compatibility
unset TEST_DATABASE_URL
export DATABASE_TEST_URL="postgresql://localhost/test"
pytest tests/integration/database/ -v  # Should work with warning

# Test validation
unset DATABASE_URL
python -c "from rag_factory.config.env_validator import EnvironmentValidator; EnvironmentValidator.validate('production')"
# Should raise ValueError
```

---

## Definition of Done

- [ ] All environment files updated (`.env`, `.env.example`, `tests/.env.test`)
- [ ] All code updated to use `TEST_DATABASE_URL`
- [ ] Environment variable validation implemented
- [ ] Backward compatibility with deprecation warnings
- [ ] All tests pass with new variable names
- [ ] Documentation updated (`ENVIRONMENT_VARIABLES.md`, `README.md`)
- [ ] Migration guide created for users
- [ ] CI/CD configuration updated
- [ ] Type checking passes
- [ ] Linting passes
- [ ] PR approved and merged

---

## Notes

- **Backward compatibility** is important - support both variable names during transition
- **Deprecation warnings** help users migrate smoothly
- **Clear documentation** prevents confusion
- **Validation** catches configuration errors early
- This story enables Story 16.3 (test fixtures need correct variable names)

---

## Migration Guide for Users

### Quick Migration

```bash
# 1. Update your .env file
sed -i 's/DATABASE_TEST_URL=/TEST_DATABASE_URL=/g' .env

# 2. Verify tests still work
pytest tests/ -v

# 3. Update any custom scripts or CI/CD configuration
```

### Manual Migration

1. Open your `.env` file
2. Find the line: `DATABASE_TEST_URL=...`
3. Rename to: `TEST_DATABASE_URL=...`
4. Save the file
5. Run tests to verify: `pytest tests/`

### Transition Period

- Both variable names will work during transition
- You'll see deprecation warnings if using old names
- Update at your convenience before next major release
