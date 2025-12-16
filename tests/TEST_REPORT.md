# Test Results Report

**Generated:** Mon Dec 15 03:40:15 PM -03 2025  
**Total Test Files:** 156  
**Total Duration:** 69m 7s

---

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **Passed Tests** | 1,754 | 97.1% |
| ❌ **Failed Tests** | 14 | 0.8% |
| ⏭️ **Skipped Tests** | 38 | 2.1% |
| **Total Tests** | 1,806 | 100% |

| File Status | Count | Percentage |
|-------------|-------|------------|
| ✅ **Passed Files** | 152 | 97.4% |
| ❌ **Failed Files** | 2 | 1.3% |
| ⏭️ **Skipped Files** | 2 | 1.3% |
| **Total Files** | 156 | 100% |

---

## Test Failures by Category

### 1. Configuration Issues (15 errors)

#### 1.1 Database Service Configuration - Migration Validator Tests
**File:** `tests/integration/database/test_migration_validator_integration.py`  
**Status:** ❌ 15 ERRORS  
**Root Cause:** `AttributeError: property 'engine' of 'PostgresqlDatabaseService' object has no setter`

**Affected Tests:**
- `test_validate_with_no_migrations`
- `test_validate_with_partial_migrations`
- `test_validate_with_all_migrations`
- `test_validate_or_raise_success`
- `test_validate_or_raise_failure`
- `test_get_current_revision`
- `test_get_all_revisions`
- `test_is_at_head`
- `test_error_message_details`
- `test_validate_single_migration`
- `test_validate_nonexistent_migration`
- `test_validate_after_downgrade`
- `test_multiple_validators_same_database`
- `test_validator_with_auto_discovered_config`
- `test_validate_empty_requirements`

**Issue:** Test fixture tries to set `service.engine = create_engine(test_db_url)` but the `engine` property is read-only.

**Category:** Configuration/Test Setup Issue

---

### 2. External Dependency Issues (3 failures)

#### 2.1 ONNX Model Download Failure
**File:** `tests/integration/registry/test_registry_integration.py`  
**Test:** `TestRealServiceInstantiation::test_multiple_service_instantiation`  
**Status:** ❌ FAILED

**Error:**
```
ServiceInstantiationError: Could not download ONNX model 'Xenova/all-MiniLM-L6-v2'.
404 Client Error: Entry Not Found for url: https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/model.onnx
```

**Category:** External Model Dependency Issue

**Solution:**
- Download model manually: `python scripts/download_embedding_model.py --model Xenova/all-MiniLM-L6-v2`
- Or use a different model

---

### 3. Schema Validation Issues (2 failures)

#### 3.1 Invalid Service Configuration Test
**File:** `tests/integration/registry/test_registry_integration.py`  
**Test:** `TestErrorHandling::test_invalid_service_config`  
**Status:** ❌ FAILED

**Error:**
```
ConfigValidationError: Schema validation failed: {'unknown_field': 'value'} is not valid under any of the given schemas
```

**Category:** Schema Validation - Test Expectation Mismatch

**Issue:** Test expects validation to pass or handle gracefully, but schema validation raises an exception during registry initialization.

#### 3.2 Configuration Warnings Test
**File:** `tests/integration/registry/test_registry_integration.py`  
**Test:** `TestConfigurationValidation::test_configuration_warnings`  
**Status:** ❌ FAILED

**Error:**
```
ConfigValidationError: Schema validation failed: {'provider': 'onnx', 'model': 'test-model', 'api_key': 'plaintext-secret'} is not valid under any of the given schemas
```

**Category:** Schema Validation - Test Expectation Mismatch

**Issue:** Test configuration doesn't match the required schema structure for ONNX embedding services.

---

### 4. Implementation Issues (10 failures)

#### 4.1 Vector Embedding Indexing Strategy
**File:** `tests/integration/strategies/test_vector_embedding_indexing.py`  
**Status:** ❌ 10 FAILED (5 tests with multiple assertion failures)

**Test 1:** `test_capabilities_and_dependencies`
- **Error:** Capability mismatch - strategy produces `CHUNKS` capability but test expects it not to
- **Expected:** `{VECTORS, DATABASE}`
- **Actual:** `{VECTORS, DATABASE, CHUNKS}`
- **Category:** Implementation/Test Expectation Mismatch

**Test 2:** `test_process_success`
- **Error:** `AssertionError: Expected 'get_chunks_for_documents' to be called once. Called 0 times.`
- **Root Cause:** Documents with empty text are being skipped, no chunks created
- **Log:** `"Skipping empty document: doc1"`, `"No chunks created from documents (all empty?)"`
- **Category:** Test Data Issue - Empty Documents

**Test 3:** `test_process_batching`
- **Error:** `AssertionError: assert 0 == 2` (embed_batch not called)
- **Root Cause:** Same as Test 2 - empty documents, no embedding calls made
- **Category:** Test Data Issue - Empty Documents

**Test 4:** `test_no_chunks_error`
- **Error:** `Failed: DID NOT RAISE <class 'ValueError'>`
- **Root Cause:** Strategy returns empty result instead of raising ValueError when no chunks are created
- **Category:** Implementation Issue - Error Handling

**Test 5:** `test_service_error`
- **Error:** `Failed: DID NOT RAISE <class 'Exception'>`
- **Root Cause:** Same as Test 4 - early return prevents error propagation
- **Category:** Implementation Issue - Error Handling

---

## Skipped Tests (38 tests)

### Configuration-Dependent Skips

#### 1. Model Comparison Performance Tests
**File:** `tests/benchmarks/test_model_comparison_performance.py`  
**Tests:** 6 skipped  
**Reason:** Requires specific model configurations or API keys

#### 2. Documentation Integration Tests
**File:** `tests/integration/documentation/test_documentation_integration.py`  
**Tests:** 6 skipped  
**Reason:** Documentation generation dependencies not available

#### 3. Database Integration Tests
**File:** `tests/integration/database/test_database_integration.py`  
**Tests:** 1 skipped (13 passed)  
**Reason:** Specific database configuration required

#### 4. Multi-Context Isolation Tests
**File:** `tests/integration/database/test_multi_context_isolation.py`  
**Tests:** 8 skipped (2 passed)  
**Reason:** Requires specific database setup

#### 5. Embedding Integration Tests
**File:** `tests/integration/services/test_embedding_integration.py`  
**Tests:** 2 skipped (6 passed)  
**Reason:** Specific embedding provider configuration required

#### 6. LLM Integration Tests
**File:** `tests/integration/services/test_llm_integration.py`  
**Tests:** 7 skipped (1 passed)  
**Reason:** API keys or LLM service configuration required

#### 7. Service Implementation Tests
**File:** `tests/integration/services/test_service_implementations.py`  
**Tests:** 1 skipped (18 passed)  
**Reason:** Specific service configuration required

#### 8. Reranking Integration Tests
**File:** `tests/integration/strategies/test_reranking_integration.py`  
**Tests:** 1 skipped (8 passed)  
**Reason:** Reranking model or service configuration required

---

## Detailed Failure Analysis

### Critical Issues (Require Immediate Attention)

1. **PostgresqlDatabaseService Engine Property** (15 errors)
   - **Impact:** High - Blocks all migration validator integration tests
   - **Fix:** Make `engine` property settable or refactor test fixtures to use proper initialization
   - **Priority:** HIGH

2. **VectorEmbeddingIndexing Error Handling** (5 failures)
   - **Impact:** Medium - Core indexing strategy not handling edge cases properly
   - **Fix:** 
     - Raise `ValueError` when no chunks are created instead of returning empty result
     - Ensure test documents have non-empty text content
     - Update capability declarations if CHUNKS should be produced
   - **Priority:** MEDIUM

### Non-Critical Issues

3. **ONNX Model Download** (1 failure)
   - **Impact:** Low - Only affects one test, model availability issue
   - **Fix:** Pre-download model or mock the download in tests
   - **Priority:** LOW

4. **Schema Validation Tests** (2 failures)
   - **Impact:** Low - Test expectations don't match current validation behavior
   - **Fix:** Update tests to expect exceptions or adjust validation logic
   - **Priority:** LOW

---

## Failing Test Files

1. `tests/integration/database/test_migration_validator_integration.py` (15 errors)
2. `tests/integration/registry/test_registry_integration.py` (3 failures)
3. `tests/integration/strategies/test_vector_embedding_indexing.py` (10 failures)

---

## Individual Failing Tests

### Migration Validator Integration (15 tests)
1. `TestMigrationValidatorIntegration::test_validate_with_no_migrations`
2. `TestMigrationValidatorIntegration::test_validate_with_partial_migrations`
3. `TestMigrationValidatorIntegration::test_validate_with_all_migrations`
4. `TestMigrationValidatorIntegration::test_validate_or_raise_success`
5. `TestMigrationValidatorIntegration::test_validate_or_raise_failure`
6. `TestMigrationValidatorIntegration::test_get_current_revision`
7. `TestMigrationValidatorIntegration::test_get_all_revisions`
8. `TestMigrationValidatorIntegration::test_is_at_head`
9. `TestMigrationValidatorIntegration::test_error_message_details`
10. `TestMigrationValidatorIntegration::test_validate_single_migration`
11. `TestMigrationValidatorIntegration::test_validate_nonexistent_migration`
12. `TestMigrationValidatorIntegration::test_validate_after_downgrade`
13. `TestMigrationValidatorIntegration::test_multiple_validators_same_database`
14. `TestMigrationValidatorEdgeCases::test_validator_with_auto_discovered_config`
15. `TestMigrationValidatorEdgeCases::test_validate_empty_requirements`

### Registry Integration (3 tests)
16. `TestRealServiceInstantiation::test_multiple_service_instantiation`
17. `TestErrorHandling::test_invalid_service_config`
18. `TestConfigurationValidation::test_configuration_warnings`

### Vector Embedding Indexing (5 tests)
19. `TestVectorEmbeddingIndexing::test_capabilities_and_dependencies`
20. `TestVectorEmbeddingIndexing::test_process_success`
21. `TestVectorEmbeddingIndexing::test_process_batching`
22. `TestVectorEmbeddingIndexing::test_no_chunks_error`
23. `TestVectorEmbeddingIndexing::test_service_error`

---

## Recommendations

### Immediate Actions

1. **Fix PostgresqlDatabaseService Engine Property**
   - Add setter for `engine` property or refactor test fixtures
   - This will resolve 15 test errors immediately

2. **Fix VectorEmbeddingIndexing Error Handling**
   - Update strategy to raise proper exceptions for edge cases
   - Fix test data to use non-empty documents
   - Clarify capability declarations

3. **Update Schema Validation Tests**
   - Align test expectations with current validation behavior
   - Consider whether validation should be more permissive or tests should expect exceptions

### Future Improvements

1. **Model Dependency Management**
   - Implement model caching/pre-download for CI/CD
   - Add fallback mechanisms for model availability issues

2. **Test Data Quality**
   - Review all test fixtures to ensure proper data setup
   - Add validation for test data before running tests

3. **Configuration Documentation**
   - Document all required configurations for skipped tests
   - Provide example configurations for local testing

---

## Code Coverage

**Overall Coverage:** 19% (2,248 / 12,116 statements)

### Well-Covered Modules (>70%)
- `rag_factory/exceptions.py` - 100%
- `rag_factory/services/llm/base.py` - 100%
- `rag_factory/services/interfaces.py` - 100%
- `rag_factory/database/config.py` - 100%
- `rag_factory/registry/service_registry.py` - 85%
- `rag_factory/strategies/query_expansion/base.py` - 85%
- `rag_factory/services/embedding/base.py` - 81%

### Modules Needing Coverage (<20%)
- All evaluation modules - 0%
- All agentic strategy modules - 0%
- All knowledge graph modules - 0%
- All late chunking modules - 0%
- All multi-query modules - 0%
- All self-reflective modules - 0%
- Most chunking strategies - <20%
- Most indexing strategies - <20%

---

## Warnings

### Deprecation Warnings (2)

1. **Pydantic V2 Migration**
   - `rag_factory/services/llm/config.py:8` - Class-based config deprecated
   - `rag_factory/database/config.py:12` - Class-based config deprecated
   - **Action Required:** Migrate to `ConfigDict` before Pydantic V3.0

---

## Conclusion

The test suite shows **excellent overall health** with a **97.1% pass rate**. The failures are concentrated in three specific areas:

1. **Configuration issues** in migration validator tests (easily fixable)
2. **Implementation gaps** in vector embedding error handling (requires code changes)
3. **External dependencies** and schema validation (low priority)

Most skipped tests are due to missing API keys or specific service configurations, which is expected in a development environment.

**Next Steps:**
1. Fix the `PostgresqlDatabaseService.engine` property issue
2. Update `VectorEmbeddingIndexing` error handling
3. Address schema validation test expectations
4. Continue improving code coverage for untested modules
