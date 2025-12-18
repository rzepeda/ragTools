# Test Report

**Generated:** 2025-12-18  
**Test Execution Date:** Thu Dec 18 01:44:44 AM -03 2025  
**Total Duration:** 69m 39s

---

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Test Files** | 197 | 100% |
| ✅ Passed Files | 172 | 87.3% |
| ❌ Failed Files | 22 | 11.2% |
| ⏭️ Skipped Files | 3 | 1.5% |
| **Total Tests** | 2,198 | 100% |
| ✅ Passed Tests | 2,008 | 91.4% |
| ❌ Failed Tests | 126 | 5.7% |
| ⏭️ Skipped Tests | 64 | 2.9% |

---

## Error Categories

### 1. Configuration Issues (🔧)

These tests fail due to missing or incorrect configuration such as API keys, database connections, or model files.

#### 1.1 Database Connection Issues
**Impact:** Low - Tests require PostgreSQL/Neo4j running  
**Affected Files:** 0 (All database tests are passing with proper configuration)

#### 1.2 Missing Model Files
**Impact:** Low - Tests require ONNX models to be downloaded  
**Affected Files:** 1

- `tests/integration/registry/test_registry_integration.py`
  - **Error:** `FileNotFoundError` - ONNX model files not found
  - **Solution:** Download required ONNX models or skip tests when models unavailable

#### 1.3 Import/Package Issues
**Impact:** Medium - Package installation or import path issues  
**Affected Files:** 2

- `tests/integration/test_package_integration.py`
  - **Error:** `ModuleNotFoundError` - Cannot import installed package
  - **Solution:** Ensure package is properly installed in test environment

- `tests/unit/test_package.py`
  - **Error:** `ModuleNotFoundError` - Import path issues
  - **Solution:** Fix import paths or package structure

---

### 2. Requirement Issues (📋)

These tests fail due to missing requirements, dependencies, or environmental prerequisites that need to be addressed in the codebase.

#### 2.1 Database Migration - Vector Extension Dependencies
**Impact:** High - Affects database migration rollback functionality  
**Affected Files:** 3  
**Total Failures:** 28 tests

**Root Cause:** Migration downgrade attempts to drop the `vector` extension, but dependent objects (like `test_chunks_real` table) still exist.

**Error Message:**
```
psycopg2.errors.DependentObjectsStillExist: cannot drop extension vector because other objects depend on it
DETAIL: column embedding of table test_chunks_real depends on type vector
HINT: Use DROP ... CASCADE to drop the dependent objects too.
```

**Affected Files:**
- `tests/integration/database/test_migration_integration.py` (3 tests)
  - `test_real_migration_execution`
  - `test_migration_with_existing_data`
  - `test_rollback_functionality`

- `tests/integration/database/test_migration_validator_integration.py` (12 tests)
  - `test_validate_with_no_migrations`
  - `test_validate_with_partial_migrations`
  - `test_validate_or_raise_failure`
  - `test_get_current_revision`
  - `test_is_at_head`
  - `test_error_message_details`
  - And 6 more...

- `tests/unit/database/test_migrations.py` (6 tests)
  - `test_migration_upgrade_to_head`
  - `test_migration_downgrade`
  - `test_migration_creates_tables`
  - `test_migration_creates_indexes`
  - `test_migration_idempotency`
  - `test_get_current_version`

**Recommendation:**
- Update migration downgrade scripts to use `DROP ... CASCADE` or properly clean up dependent objects before dropping the vector extension
- Add test fixtures to properly clean up database state between tests

#### 2.2 Database Migration - Missing Tables
**Impact:** High - Affects migration upgrade functionality  
**Affected Files:** 2  
**Total Failures:** 8 tests

**Root Cause:** Migration 002 attempts to add columns to the `chunks` table, but the table doesn't exist because migration 001 was not properly executed.

**Error Message:**
```
psycopg2.errors.UndefinedTable: relation "chunks" does not exist
[SQL: ALTER TABLE chunks ADD COLUMN parent_chunk_id UUID]
```

**Affected Files:**
- `tests/integration/database/test_migration_integration.py` (1 test)
  - `test_pgvector_extension_installed`

- `tests/integration/database/test_migration_validator_integration.py` (7 tests)
  - `test_validate_with_all_migrations`
  - `test_validate_or_raise_success`
  - `test_validate_single_migration`
  - `test_validate_nonexistent_migration`
  - `test_validate_after_downgrade`
  - `test_multiple_validators_same_database`

**Recommendation:**
- Fix migration test fixtures to ensure proper migration order
- Add migration dependency checks before running upgrade scripts
- Ensure test database is in a clean state before each test

---

### 3. Code Issues (🐛)

These tests fail due to bugs, incorrect implementations, or API mismatches in the code.

#### 3.1 Type Errors
**Impact:** Medium - Code implementation issues  
**Affected Files:** 9  
**Total Failures:** ~35 tests

**Common Patterns:**
- Incorrect function signatures
- Missing or extra parameters
- Type mismatches in function calls

**Affected Files:**
- `tests/integration/services/test_service_integration.py` (2 tests)
  - `test_embedding_database_consistency`
  - `test_rag_workflow`

- `tests/integration/strategies/test_self_reflective_integration.py` (4 tests)
  - `test_end_to_end_workflow`
  - `test_performance_within_limits`
  - `test_retry_with_poor_results`
  - `test_with_real_llm`

- `tests/unit/database/test_pgvector.py` (1 test)
  - `test_cosine_similarity_search`

- `tests/unit/services/embedding/test_onnx_local_provider.py` (1 test)
  - `test_calculate_cost_is_zero`

- `tests/unit/services/llm/test_anthropic_provider.py` (multiple tests)
  - `test_stream` and others

- `tests/unit/services/llm/test_ollama_provider.py` (multiple tests)

- `tests/unit/services/llm/test_openai_provider.py` (multiple tests)

- `tests/unit/services/llm/test_service.py` (multiple tests)

- `tests/unit/services/test_database_service.py` (1 test)
  - `test_store_chunks_with_hierarchy`

**Recommendation:**
- Review and fix function signatures to match expected interfaces
- Update tests to use correct API calls
- Add type hints and validation

#### 3.2 Assertion Errors
**Impact:** Medium - Test expectations don't match implementation  
**Affected Files:** 3  
**Total Failures:** ~10 tests

**Affected Files:**
- `tests/unit/config/test_strategy_pair_manager.py` (2 tests)
  - `test_db_context_creation`
  - `test_load_pair_success`

- `tests/unit/documentation/test_code_examples.py` (1 test)
  - `test_all_code_examples_have_valid_syntax`

- `tests/unit/documentation/test_links.py` (1 test)
  - `test_no_broken_internal_links`

**Recommendation:**
- Update test assertions to match current implementation
- Fix code examples in documentation
- Repair broken internal links

#### 3.3 Attribute Errors
**Impact:** Low - Missing or incorrect attribute access  
**Affected Files:** 1  
**Total Failures:** ~9 tests

**Affected Files:**
- `tests/unit/registry/test_service_factory.py` (9 tests)
  - Database service creation tests
  - Embedding service creation tests
  - LLM service creation tests

**Recommendation:**
- Fix attribute access in service factory
- Ensure all required attributes are properly initialized

#### 3.4 Other Code Issues
**Affected Files:** 2

- `tests/test_mock_registry.py`
  - Various mock registry issues

- `tests/unit/strategies/indexing/test_context_aware.py`
  - Context-aware indexing strategy issues

- `tests/unit/database/test_batch_operations.py` (1 test)
  - `test_store_chunks_with_hierarchy`

---

## Failing Test Files

### Integration Tests (6 files)

1. **tests/integration/database/test_migration_integration.py**
   - Category: Requirement Issue (Database Migration)
   - Failures: 4/4 tests (100%)
   - Priority: High

2. **tests/integration/database/test_migration_validator_integration.py**
   - Category: Requirement Issue (Database Migration)
   - Failures: 12/15 tests (80%)
   - Priority: High

3. **tests/integration/registry/test_registry_integration.py**
   - Category: Configuration Issue (Missing Model Files)
   - Failures: 3 tests
   - Priority: Low

4. **tests/integration/services/test_service_integration.py**
   - Category: Code Issue (Type Errors)
   - Failures: 2 tests
   - Priority: Medium

5. **tests/integration/strategies/test_self_reflective_integration.py**
   - Category: Code Issue (Type Errors)
   - Failures: 4 tests
   - Priority: Medium

6. **tests/integration/test_package_integration.py**
   - Category: Configuration Issue (Import Errors)
   - Failures: 2 tests
   - Priority: Medium

### Unit Tests (16 files)

7. **tests/test_mock_registry.py**
   - Category: Code Issue
   - Priority: Medium

8. **tests/unit/config/test_strategy_pair_manager.py**
   - Category: Code Issue (Assertion Errors)
   - Failures: 2 tests
   - Priority: Medium

9. **tests/unit/database/test_batch_operations.py**
   - Category: Code Issue
   - Failures: 1 test
   - Priority: Low

10. **tests/unit/database/test_migrations.py**
    - Category: Requirement Issue (Database Migration)
    - Failures: 6 tests
    - Priority: High

11. **tests/unit/database/test_pgvector.py**
    - Category: Code Issue (Type Errors)
    - Failures: 1 test
    - Priority: Low

12. **tests/unit/documentation/test_code_examples.py**
    - Category: Code Issue (Assertion Errors)
    - Failures: 1 test
    - Priority: Low

13. **tests/unit/documentation/test_links.py**
    - Category: Code Issue (Assertion Errors)
    - Failures: 1 test
    - Priority: Low

14. **tests/unit/registry/test_service_factory.py**
    - Category: Code Issue (Attribute Errors)
    - Failures: 9 tests
    - Priority: Medium

15. **tests/unit/services/embedding/test_onnx_local_provider.py**
    - Category: Code Issue (Type Errors)
    - Failures: 1 test
    - Priority: Low

16. **tests/unit/services/llm/test_anthropic_provider.py**
    - Category: Code Issue (Type Errors)
    - Failures: Multiple tests
    - Priority: Medium

17. **tests/unit/services/llm/test_ollama_provider.py**
    - Category: Code Issue (Type Errors)
    - Failures: Multiple tests
    - Priority: Medium

18. **tests/unit/services/llm/test_openai_provider.py**
    - Category: Code Issue (Type Errors)
    - Failures: Multiple tests
    - Priority: Medium

19. **tests/unit/services/llm/test_service.py**
    - Category: Code Issue (Type Errors)
    - Failures: Multiple tests
    - Priority: Medium

20. **tests/unit/services/test_database_service.py**
    - Category: Code Issue (Type Errors)
    - Failures: 1 test
    - Priority: Low

21. **tests/unit/strategies/indexing/test_context_aware.py**
    - Category: Code Issue
    - Failures: Unknown
    - Priority: Low

22. **tests/unit/test_package.py**
    - Category: Configuration Issue (Import Errors)
    - Failures: Unknown
    - Priority: Medium

---

## Individual Failing Tests

### High Priority (Database Migration Issues)

#### tests/integration/database/test_migration_integration.py
- `TestMigrationIntegration::test_real_migration_execution`
- `TestMigrationIntegration::test_migration_with_existing_data`
- `TestMigrationIntegration::test_rollback_functionality`
- `TestMigrationIntegration::test_pgvector_extension_installed`

#### tests/integration/database/test_migration_validator_integration.py
- `TestMigrationValidatorIntegration::test_validate_with_no_migrations`
- `TestMigrationValidatorIntegration::test_validate_with_partial_migrations`
- `TestMigrationValidatorIntegration::test_validate_with_all_migrations`
- `TestMigrationValidatorIntegration::test_validate_or_raise_success`
- `TestMigrationValidatorIntegration::test_validate_or_raise_failure`
- `TestMigrationValidatorIntegration::test_get_current_revision`
- `TestMigrationValidatorIntegration::test_is_at_head`
- `TestMigrationValidatorIntegration::test_error_message_details`
- `TestMigrationValidatorIntegration::test_validate_single_migration`
- `TestMigrationValidatorIntegration::test_validate_nonexistent_migration`
- `TestMigrationValidatorIntegration::test_validate_after_downgrade`
- `TestMigrationValidatorIntegration::test_multiple_validators_same_database`

#### tests/unit/database/test_migrations.py
- `TestAlembicMigrations::test_migration_upgrade_to_head`
- `TestAlembicMigrations::test_migration_downgrade`
- `TestAlembicMigrations::test_migration_creates_tables`
- `TestAlembicMigrations::test_migration_creates_indexes`
- `TestAlembicMigrations::test_migration_idempotency`
- `TestAlembicMigrations::test_get_current_version`

### Medium Priority (Code Issues)

#### tests/integration/services/test_service_integration.py
- `test_embedding_database_consistency`
- `test_rag_workflow`

#### tests/integration/strategies/test_self_reflective_integration.py
- `TestSelfReflectiveIntegration::test_end_to_end_workflow`
- `TestSelfReflectiveIntegration::test_performance_within_limits`
- `TestSelfReflectiveIntegration::test_retry_with_poor_results`
- `TestSelfReflectiveWithLMStudio::test_with_real_llm`

#### tests/unit/config/test_strategy_pair_manager.py
- `test_db_context_creation`
- `test_load_pair_success`

#### tests/unit/registry/test_service_factory.py
- `TestDatabaseServiceCreation::test_create_database_service_neo4j`
- `TestDatabaseServiceCreation::test_create_database_service_neo4j_with_defaults`
- `TestDatabaseServiceCreation::test_create_database_service_postgres_with_components`
- `TestDatabaseServiceCreation::test_create_database_service_postgres_with_connection_string`
- `TestDatabaseServiceCreation::test_create_database_service_postgres_with_defaults`
- `TestEmbeddingServiceCreation::test_create_embedding_service_onnx`
- `TestEmbeddingServiceCreation::test_create_embedding_service_onnx_with_defaults`
- `TestEmbeddingServiceCreation::test_create_embedding_service_openai`
- `TestLLMServiceCreation::test_create_llm_service_openai`

#### tests/unit/services/llm/test_anthropic_provider.py
- `test_stream`
- `test_generate`
- `test_generate_with_system_prompt`
- And others...

#### tests/unit/services/llm/test_ollama_provider.py
- Multiple streaming and generation tests

#### tests/unit/services/llm/test_openai_provider.py
- Multiple streaming and generation tests

#### tests/unit/services/llm/test_service.py
- Multiple LLM service tests

### Low Priority (Configuration & Minor Issues)

#### tests/integration/registry/test_registry_integration.py
- `TestConfigurationValidation::test_configuration_warnings`
- `TestErrorHandling::test_invalid_service_config`
- `TestRealServiceInstantiation::test_multiple_service_instantiation`

#### tests/integration/test_package_integration.py
- `TestFullWorkflow::test_full_workflow_with_installed_package`
- `TestSmokeTest::test_basic_usage_smoke_test`

#### tests/unit/database/test_batch_operations.py
- `TestBatchOperations::test_store_chunks_with_hierarchy`

#### tests/unit/database/test_pgvector.py
- `TestPgVectorIntegration::test_cosine_similarity_search`

#### tests/unit/documentation/test_code_examples.py
- `TestCodeExamples::test_all_code_examples_have_valid_syntax`

#### tests/unit/documentation/test_links.py
- `TestDocumentationLinks::test_no_broken_internal_links`

#### tests/unit/services/embedding/test_onnx_local_provider.py
- `test_calculate_cost_is_zero`

#### tests/unit/services/test_database_service.py
- `test_store_chunks_with_hierarchy`

---

## Recommendations

### Immediate Actions (High Priority)

1. **Fix Database Migration Issues** (28 failing tests)
   - Update migration downgrade scripts to use `DROP ... CASCADE` or properly clean up dependent objects
   - Fix migration test fixtures to ensure proper migration order and clean database state
   - Add migration dependency checks

2. **Fix Service Factory Attribute Errors** (9 failing tests)
   - Review and fix attribute access in service factory
   - Ensure all required attributes are properly initialized

### Short-term Actions (Medium Priority)

3. **Fix LLM Provider Type Errors** (~20 failing tests)
   - Review and update function signatures for LLM providers
   - Ensure streaming and generation methods match expected interfaces
   - Add proper type hints

4. **Fix Integration Test Type Errors** (6 failing tests)
   - Update service integration tests to use correct API calls
   - Fix self-reflective strategy integration tests

5. **Fix Package Import Issues** (2 failing tests)
   - Ensure package is properly installed in test environment
   - Fix import paths

### Long-term Actions (Low Priority)

6. **Fix Documentation Tests** (2 failing tests)
   - Update code examples in documentation
   - Repair broken internal links

7. **Handle Missing Model Files** (3 failing tests)
   - Add proper skip decorators for tests requiring ONNX models
   - Document model download requirements

8. **Fix Minor Code Issues** (5 failing tests)
   - Fix batch operations tests
   - Fix pgvector integration tests
   - Fix context-aware indexing tests

---

## Test Coverage

Based on the coverage report from the test run:

- **Overall Coverage:** 12%
- **Total Statements:** 12,307
- **Missed Statements:** 10,858

### Low Coverage Areas

- Evaluation modules: 0% coverage
- Strategies (chunking, contextual, knowledge graph, etc.): 0% coverage
- Observability modules: 0% coverage
- Repository modules: 0% coverage
- Legacy config: 0% coverage
- Many service providers: 0% coverage

### Good Coverage Areas

- Core exceptions: 100%
- Registry base: 100%
- Service interfaces: 100%
- Base models: 60-100%
- CLI main: 68%
- Core capabilities: 74%
- Query expansion base: 85%

**Recommendation:** Increase test coverage for untested modules, especially evaluation, strategies, and observability components.

---

## Conclusion

The test suite shows **91.4% test pass rate** with 2,008 out of 2,198 tests passing. The main issues are:

1. **Database migration problems** (28 tests) - Requires fixing migration scripts
2. **Type errors in LLM providers** (~20 tests) - Requires API signature fixes
3. **Service factory issues** (9 tests) - Requires attribute initialization fixes
4. **Configuration issues** (5 tests) - Requires proper environment setup

Addressing the high-priority database migration issues and service factory problems would significantly improve the test pass rate to approximately **95%**.