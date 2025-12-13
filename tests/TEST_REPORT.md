# Test Suite Report

**Generated:** 2025-12-13  
**Execution Date:** 2025-12-12 08:34:50 PM -03  
**Total Duration:** 73m 47s  
**Test Runner:** File-based execution with 5-minute timeout per file

---

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Test Files** | 156 | 100% |
| **Total Tests** | 1,892 | 100% |
| ✅ **Passed Tests** | 1,616 | 85.4% |
| ❌ **Failed Tests** | 239 | 12.6% |
| ⏭️ **Skipped Tests** | 37 | 2.0% |
| | | |
| ✅ **Passed Files** | 121 | 77.6% |
| ❌ **Failed Files** | 32 | 20.5% |
| ⏭️ **Skipped Files** | 2 | 1.3% |
| ⏱️ **Timeout Files** | 1 | 0.6% |

### Key Findings

- **Overall Health:** 85.4% pass rate - strong foundation with specific fixable issues
- **Primary Blocker:** 1 timeout in embedding integration tests
- **Root Causes Identified:** Most failures stem from 5 main categories
- **Improvement Trend:** +1.1% pass rate improvement from previous run

---

## 🔴 Critical Issues by Category

### 1. Database Migration Issues 🗄️ **HIGH PRIORITY**

**Impact:** 6 test failures  
**Error Types:**
- `psycopg2.errors.UndefinedObject: index "idx_chunks_hierarchy" does not exist`
- `psycopg2.errors.UndefinedTable: relation "documents" does not exist`
- `sqlalchemy.exc.NoSuchTableError: chunks`

**Affected Test Files:**
- `tests/integration/database/test_migration_integration.py` (6 failures)

**Specific Failures:**

1. **test_real_migration_execution:**
   ```
   psycopg2.errors.UndefinedObject: index "idx_chunks_hierarchy" does not exist
   [SQL: DROP INDEX idx_chunks_hierarchy]
   ```
   - Issue: Downgrade attempting to drop non-existent index
   - Location: `migrations/versions/002_add_hierarchy_support.py:239`

2. **test_migration_with_existing_data:**
   ```
   psycopg2.errors.UndefinedTable: relation "documents" does not exist
   ```
   - Issue: Test assumes tables exist before migration
   - Needs proper test setup

3. **test_rollback_functionality:**
   ```
   sqlalchemy.exc.NoSuchTableError: chunks
   ```
   - Issue: Table doesn't exist after rollback attempt

**Recommendations:**
1. Fix migration downgrade scripts to check for existence before dropping
2. Improve test fixtures to ensure proper database state
3. Add `IF EXISTS` clauses to DROP statements
4. Review migration order and dependencies

---

### 2. Fine-Tuned Embeddings A/B Testing 🧪 **HIGH PRIORITY**

**Impact:** 4 test failures  
**Error:** Insufficient samples for A/B testing

**Affected Test Files:**
- `tests/integration/models/test_fine_tuned_embeddings_integration.py` (4 failures)

**Specific Failures:**

1. **test_ab_testing_workflow:**
   ```python
   AssertionError: assert 0 > 0
   where 0 = ABTestResult(..., model_a_samples=0, model_b_samples=49, ...)
   ```
   - Warning: `Insufficient samples: A=0, B=49 (minimum=50)`
   - Issue: Model A not generating any samples

2. **test_model_comparison_workflow:**
   ```python
   AssertionError: assert (0 + 0) == 20
   where 0 = ABTestResult(..., model_a_samples=0, model_b_samples=0, ...)
   ```
   - Warning: `Insufficient samples: A=0, B=0 (minimum=10)`
   - Issue: Neither model generating samples

**Root Cause:** Sample collection logic not working properly for A/B testing

**Recommendations:**
1. Debug sample collection in `rag_factory/models/evaluation/ab_testing.py`
2. Verify model registration and loading
3. Check test data generation
4. Add logging to track sample collection

---

### 3. ONNX Embeddings Performance ⚡ **MEDIUM PRIORITY**

**Impact:** 2 test failures  
**Error:** Performance targets not met

**Affected Test Files:**
- `tests/integration/services/test_onnx_embeddings_integration.py` (2 failures)

**Specific Failure:**
```python
test_performance_target:
  AssertionError: Average time 3069.42ms exceeds 100ms target
  assert np.float64(3.0694208592976793) < 0.1
```

**Analysis:**
- Target: 100ms per embedding
- Actual: 3,069ms per embedding (30.7x slower)
- This is a performance benchmark, not a functional failure
- 9/11 functional tests passing

**Recommendations:**
1. Review if 100ms target is realistic for ONNX embeddings
2. Consider adjusting threshold to 3-5 seconds for ONNX
3. Separate performance benchmarks from functional tests
4. Mark as `@pytest.mark.slow` or `@pytest.mark.performance`
5. Investigate ONNX optimization opportunities

---

### 4. Embedding Integration Timeout ⏱️ **HIGH PRIORITY**

**Impact:** 1 file timeout (5-minute limit exceeded)  
**Affected:** `tests/integration/services/test_embedding_integration.py`

**Status:** Test started but exceeded 5-minute timeout
- Last visible test: `test_large_batch_processing` (running when timeout occurred)
- Partial results before timeout: 1 passed, 1 failed, 1 skipped

**Specific Failure Before Timeout:**
```
test_openai_full_workflow FAILED
test_cohere_full_workflow SKIPPED
test_local_embedding_provider PASSED
test_large_batch_processing [TIMEOUT]
```

**Recommendations:**
1. Investigate `test_large_batch_processing` for infinite loops or blocking calls
2. Add timeout decorators to individual tests (`@pytest.mark.timeout(60)`)
3. Consider splitting large batch tests into smaller units
4. Add logging to identify where the hang occurs
5. Review batch processing logic for deadlocks

---

### 5. Strategy Integration Tests 🧩 **MEDIUM PRIORITY**

**Impact:** 84+ test failures across 8 files  
**Primary Cause:** Missing LLM services and API keys (expected for many)

**Affected Test Files:**
- `tests/integration/strategies/test_query_expansion_integration.py` (18 failures)
- `tests/integration/strategies/test_multi_query_integration.py` (16 failures)
- `tests/integration/strategies/test_contextual_integration.py` (12 failures)
- `tests/integration/strategies/test_late_chunking_integration.py` (12 failures)
- `tests/integration/strategies/test_base_integration.py` (10 failures)
- `tests/integration/strategies/test_hierarchical_integration.py` (8 failures)
- `tests/integration/strategies/test_knowledge_graph_integration.py` (8 failures)
- `tests/integration/strategies/test_keyword_indexing.py` (collection error)

**Common Patterns:**

1. **LM Studio Empty Responses:**
   - LM Studio connected successfully
   - Prompt tokens: 90-111 (receiving prompts)
   - Completion tokens: 150 (generating tokens)
   - But: `expanded_query` is empty string
   - Issue: LM Studio model configuration or output format

2. **API Mismatch:**
   ```python
   TypeError: MultiQueryRAGStrategy.__init__() got an unexpected keyword argument 'vector_store_service'
   ```
   - Tests using wrong parameter names
   - Need to update test code to match current API

3. **Missing Dependencies:**
   - Some tests require LLM API keys (expected)
   - Some require specific service configurations

**Recommendations:**
1. Fix LM Studio model configuration for proper output
2. Update test code to use correct API parameters
3. Add pytest markers: `@pytest.mark.requires_llm_api`
4. Implement proper mocking for CI environments
5. Document required environment variables

---

### 6. Pipeline Unit Tests 📝 **CRITICAL PRIORITY**

**Impact:** 46 test failures ⚠️  
**Affected:** `tests/unit/test_pipeline.py`

**Status:** Highest number of failures in a single file - needs immediate investigation

**Recommendations:**
1. **URGENT:** Investigate root cause of 46 failures
2. Review recent changes to pipeline code
3. Check for breaking API changes
4. Verify mock configurations
5. Run individual tests to identify patterns

---

### 7. Service Integration & Mocking Issues 🔧 **MEDIUM PRIORITY**

**Impact:** 6+ test failures  
**Errors:**
- AsyncMock cannot be awaited
- Service configuration issues
- Service instantiation errors

**Affected Test Files:**
- `tests/integration/services/test_service_implementations.py` (2 failures, 1 skipped)
- `tests/integration/services/test_service_integration.py` (2 failures)

**Specific Issues:**

1. **PostgreSQL Service Mock Error:**
   ```python
   TypeError: object AsyncMock can't be used in 'await' expression
   ```
   - Location: `test_postgresql_database_service_basic_functionality`
   - Fix: Use proper async mock configuration with `AsyncMock(return_value=...)`

2. **Service Instantiation:**
   - Configuration validation problems
   - Missing required parameters
   - Dependency injection issues

**Recommendations:**
1. Fix async mock setup in database service tests
2. Use `unittest.mock.AsyncMock` properly
3. Review service configuration requirements
4. Add better error messages for missing dependencies

---

### 8. Package Dependencies 📦 **MEDIUM PRIORITY**

**Impact:** 2 test failures  
**Affected:** `tests/integration/test_package_integration.py`

**Error:**
```
ModuleNotFoundError: No module named 'numpy'
```

**Root Cause:** Package imports `numpy` through ONNX provider, but numpy not in core dependencies

**Recommendations:**
1. Make numpy a required dependency (not optional)
2. Or: Make ONNX provider truly optional with lazy imports
3. Update `setup.py` or `pyproject.toml` to include numpy
4. Add dependency validation at import time

---

### 9. Factory & Configuration Tests ⚙️ **MEDIUM PRIORITY**

**Impact:** 38 test failures across 3 files

**Affected Test Files:**
- `tests/integration/test_factory_integration.py` (20 failures)
- `tests/integration/test_pipeline_integration.py` (14 failures)
- `tests/integration/test_config_integration.py` (4 failures)

**Likely Causes:**
- Cascading failures from missing dependencies
- Service instantiation issues
- Configuration validation problems
- API changes not reflected in tests

**Recommendations:**
1. Review factory service creation logic
2. Update configuration validation
3. Fix dependency injection
4. Update tests to match current API

---

### 10. Documentation Tests 📚 **LOW PRIORITY**

**Impact:** 12 test failures across 3 files

**Affected Test Files:**
- `tests/unit/documentation/test_links.py` (6 failures) - Broken links
- `tests/unit/documentation/test_code_examples.py` (4 failures) - Invalid examples
- `tests/unit/documentation/test_documentation_completeness.py` (2 failures) - Missing docs

**Recommendations:**
1. Fix broken documentation links
2. Update code examples to match current API
3. Add missing documentation sections
4. Set up automated link checking in CI

---

### 11. Other Unit Test Failures 🔍 **MEDIUM PRIORITY**

**Affected Files:**
- `tests/unit/strategies/agentic/test_strategy.py` (16 failures)
- `tests/unit/strategies/test_base.py` (6 failures)
- `tests/unit/cli/test_check_consistency_command.py` (2 failures)
- `tests/unit/cli/test_repl_command.py` (2 failures)
- `tests/unit/services/test_interfaces.py` (2 failures)
- `tests/unit/services/test_database_service.py` (2 failures)
- `tests/unit/repositories/test_chunk_repository.py` (failures)
- `tests/unit/services/embeddings/test_onnx_local.py` (failures)
- `tests/unit/services/embedding/test_onnx_local_provider.py` (failures)
- `tests/unit/strategies/late_chunking/test_document_embedder.py` (1 failure)
- `tests/unit/database/test_migrations.py` (failures)

**Common Issues:**
- Logic errors in strategy implementations
- Interface contract violations
- Mock configuration issues
- API mismatches

---

## 📊 Summary by Test Category

### Integration Tests (35 files)

| Category | Files | Passed | Failed | Skipped | Timeout | Key Issues |
|----------|-------|--------|--------|---------|---------|------------|
| Strategies | 8 | 0 | 8 | 0 | 0 | LLM APIs, dependencies |
| Services | 7 | 3 | 3 | 0 | 1 | Mocking, timeouts, config |
| Database | 2 | 1 | 1 | 0 | 0 | Migration issues |
| Factory/Config | 4 | 0 | 4 | 0 | 0 | Service instantiation |
| Models | 1 | 0 | 1 | 0 | 0 | A/B testing samples |
| CLI | 3 | 3 | 0 | 0 | 0 | ✅ All passing |
| Evaluation | 1 | 1 | 0 | 0 | 0 | ✅ All passing |
| Observability | 1 | 1 | 0 | 0 | 0 | ✅ All passing |
| Repositories | 1 | 1 | 0 | 0 | 0 | ✅ All passing |
| Documentation | 1 | 0 | 0 | 1 | 0 | Skipped |

### Unit Tests (118 files)

| Category | Files | Passed | Failed | Key Issues |
|----------|-------|--------|--------|------------|
| Pipeline | 1 | 0 | 1 | ⚠️ 46 failures - CRITICAL |
| Strategies | 3 | 0 | 3 | 23 failures - Logic errors |
| Documentation | 3 | 0 | 3 | 12 failures - Links, examples |
| Services | 5 | 2 | 3 | 6+ failures - Interfaces |
| Database | 2 | 1 | 1 | 4 failures - Migrations |
| CLI | 2 | 0 | 2 | 4 failures - Commands |
| Repositories | 1 | 0 | 1 | Failures - Repository logic |
| Other | 101 | 101 | 0 | ✅ All passing |

### Benchmarks (3 files)

| File | Status | Tests | Notes |
|------|--------|-------|-------|
| `test_contextual_performance.py` | ✅ Passed | 6/6 | All passing |
| `test_late_chunking_performance.py` | ✅ Passed | 8/8 | All passing |
| `test_model_comparison_performance.py` | ⏭️ Skipped | 0/6 | All skipped |

---

## 🎯 Action Plan

### Phase 1: Critical Fixes (Blocks ~70 tests) ⚡

**Priority 1: Pipeline Unit Tests**
- **File:** `tests/unit/test_pipeline.py`
- **Issue:** 46 failures - highest priority
- **Impact:** Unblocks core pipeline functionality
- **Effort:** 2-3 days
- **Priority:** 🔴 CRITICAL

**Priority 2: Fix Embedding Integration Timeout**
- **File:** `tests/integration/services/test_embedding_integration.py`
- **Test:** `test_large_batch_processing`
- **Impact:** Unblocks 1 file
- **Effort:** 1 day
- **Priority:** 🔴 CRITICAL

**Priority 3: Fix Migration Downgrade Issues**
- **File:** `migrations/versions/002_add_hierarchy_support.py`
- **Issue:** Attempting to drop non-existent indexes/tables
- **Impact:** Fixes 6 tests
- **Effort:** 0.5 days
- **Priority:** 🔴 CRITICAL

**Priority 4: Fix A/B Testing Sample Collection**
- **File:** `rag_factory/models/evaluation/ab_testing.py`
- **Issue:** Models not generating samples
- **Impact:** Fixes 4 tests
- **Effort:** 1 day
- **Priority:** 🔴 CRITICAL

---

### Phase 2: High-Impact Fixes (Blocks ~40 tests) 🔧

**1. Fix Factory Service Instantiation**
- **Files:** `tests/integration/test_factory_integration.py` (20 failures)
- **Impact:** Unblocks factory tests
- **Effort:** 2 days
- **Priority:** 🟠 HIGH

**2. Fix Pipeline Integration Tests**
- **File:** `tests/integration/test_pipeline_integration.py` (14 failures)
- **Impact:** Unblocks pipeline integration
- **Effort:** 1-2 days
- **Priority:** 🟠 HIGH

**3. Fix Async Mock in Service Tests**
- **File:** `tests/integration/services/test_service_implementations.py`
- **Impact:** Fixes 2 tests
- **Effort:** 0.5 days
- **Priority:** 🟠 HIGH

**4. Fix Package Dependencies**
- **Issue:** numpy not in core dependencies
- **Impact:** Fixes 2 tests + import issues
- **Effort:** 0.5 days
- **Priority:** 🟠 HIGH

---

### Phase 3: Strategy Integration Tests (Blocks ~84 tests) 🧩

**1. Fix LM Studio Configuration**
- **Issue:** Empty responses from LM Studio
- **Impact:** Fixes 10+ tests
- **Effort:** 1 day
- **Priority:** 🟡 MEDIUM

**2. Update Strategy Test APIs**
- **Issue:** Tests using wrong parameter names
- **Impact:** Fixes 9+ tests (MultiQueryRAGStrategy)
- **Effort:** 0.5 days
- **Priority:** 🟡 MEDIUM

**3. Add Proper Test Markers**
```python
@pytest.mark.requires_llm_api
@pytest.mark.requires_database
@pytest.mark.slow
```
- **Impact:** Better test organization
- **Effort:** 1 day
- **Priority:** 🟡 MEDIUM

---

### Phase 4: Code Quality & Documentation 📚

**1. Fix Documentation Tests**
- Broken links: 6 failures
- Code examples: 4 failures
- Completeness: 2 failures
- **Effort:** 1-2 days
- **Priority:** 🟢 LOW

**2. Fix Agentic Strategy Tests**
- 16 failures in `test_strategy.py`
- **Effort:** 1-2 days
- **Priority:** 🟢 LOW

**3. Review Performance Thresholds**
- ONNX embedding performance test
- Adjust or separate from functional tests
- **Effort:** 0.5 days
- **Priority:** 🟢 LOW

---

## 📈 Expected Outcomes

| Phase | Tests Fixed | New Pass Rate | Cumulative Effort |
|-------|-------------|---------------|-------------------|
| **Current** | - | 85.4% | - |
| **Phase 1** | ~70 | ~89% | 4-5 days |
| **Phase 2** | ~40 | ~91% | 6-9 days |
| **Phase 3** | ~84 | ~96% | 8-11 days |
| **Phase 4** | ~30 | ~98%+ | 10-14 days |

---

## 🔍 Detailed Test File Status

### ✅ Passing Files (121 files)

**Highlights:**
- All CLI integration tests (3 files, 14 tests)
- All repository integration tests (1 file, 19 tests)
- All observability tests (1 file, 11 tests)
- All evaluation tests (1 file, 7 tests)
- 101 unit test files passing
- 2 benchmark files passing

### ❌ Failed Files (32 files)

#### Integration Tests (17 files)
1. `tests/integration/database/test_migration_integration.py` - Migration issues
2. `tests/integration/models/test_fine_tuned_embeddings_integration.py` - A/B testing
3. `tests/integration/services/test_onnx_embeddings_integration.py` - Performance
4. `tests/integration/services/test_service_implementations.py` - Async mocks
5. `tests/integration/services/test_service_integration.py` - Configuration
6. `tests/integration/strategies/test_base_integration.py` - LLM dependencies
7. `tests/integration/strategies/test_contextual_integration.py` - LLM dependencies
8. `tests/integration/strategies/test_hierarchical_integration.py` - LLM dependencies
9. `tests/integration/strategies/test_keyword_indexing.py` - Collection error
10. `tests/integration/strategies/test_knowledge_graph_integration.py` - LLM dependencies
11. `tests/integration/strategies/test_late_chunking_integration.py` - Mixed issues
12. `tests/integration/strategies/test_multi_query_integration.py` - API mismatch
13. `tests/integration/strategies/test_query_expansion_integration.py` - LM Studio
14. `tests/integration/test_config_integration.py` - Config validation
15. `tests/integration/test_factory_integration.py` - Service instantiation
16. `tests/integration/test_package_integration.py` - numpy dependency
17. `tests/integration/test_pipeline_integration.py` - Pipeline config

#### Unit Tests (15 files)
18. `tests/unit/test_pipeline.py` - ⚠️ 46 failures - CRITICAL
19. `tests/unit/strategies/agentic/test_strategy.py` - 16 failures
20. `tests/unit/strategies/test_base.py` - 6 failures
21. `tests/unit/documentation/test_links.py` - 6 failures
22. `tests/unit/documentation/test_code_examples.py` - 4 failures
23. `tests/unit/cli/test_check_consistency_command.py` - 2 failures
24. `tests/unit/cli/test_repl_command.py` - 2 failures
25. `tests/unit/services/test_interfaces.py` - 2 failures
26. `tests/unit/services/test_database_service.py` - 2 failures
27. `tests/unit/documentation/test_documentation_completeness.py` - 2 failures
28. `tests/unit/database/test_migrations.py` - Multiple failures
29. `tests/unit/repositories/test_chunk_repository.py` - Multiple failures
30. `tests/unit/services/embeddings/test_onnx_local.py` - Multiple failures
31. `tests/unit/services/embedding/test_onnx_local_provider.py` - Multiple failures
32. `tests/unit/strategies/late_chunking/test_document_embedder.py` - 1 failure

### ⏱️ Timeout Files (1)
- `tests/integration/services/test_embedding_integration.py` - Exceeded 5-minute limit

### ⏭️ Skipped Files (2)
- `tests/benchmarks/test_model_comparison_performance.py` - 6 tests skipped
- `tests/integration/documentation/test_documentation_integration.py` - 6 tests skipped

---

## 💡 Recommendations

### Immediate Actions
1. ✅ **CRITICAL:** Investigate `tests/unit/test_pipeline.py` (46 failures)
2. 🔧 Fix embedding integration timeout in `test_large_batch_processing`
3. 🗄️ Fix migration downgrade scripts with `IF EXISTS` clauses
4. 🧪 Debug A/B testing sample collection logic
5. 📦 Add numpy to core dependencies

### Short-Term Improvements
1. Add pytest markers for test categorization
2. Implement proper async mocking patterns
3. Fix LM Studio model configuration
4. Update strategy test APIs to match current code
5. Fix factory service instantiation

### Long-Term Enhancements
1. Separate performance tests from functional tests
2. Improve CI/CD test organization
3. Add comprehensive test documentation
4. Implement better test fixtures and cleanup
5. Increase test coverage to 50%+
6. Create test environment templates

---

## 📝 Notes

- **Test Execution:** File-based with 5-minute timeout per file
- **Coverage:** 16% overall (needs improvement)
- **Environment:** Tests run in virtual environment
- **Database:** PostgreSQL with pgvector extension
- **LLM Services:** Most integration tests require API keys (expected)
- **Improvement:** +1.1% pass rate from previous run
- **Duration:** 3m 40s faster than previous run

---

## Test Coverage Analysis

**Current Coverage:** 16% (11,070 statements, 9,290 missed)

**Areas Needing Coverage:**
- Evaluation modules: 0% coverage
- Strategy implementations: 0-20% coverage
- Service providers: 20-40% coverage
- Agentic strategies: 0% coverage
- Knowledge graph: 0% coverage
- Multi-query: 0% coverage

**Well-Covered Areas:**
- Core capabilities: 74% coverage
- Base strategies: 76% coverage
- Reranking base: 76% coverage
- Query expansion base: 85% coverage
- Embedding base: 81% coverage

---

**Report End** | For detailed logs, see `test_results_by_file.txt` and `test_summary_by_file.txt`
