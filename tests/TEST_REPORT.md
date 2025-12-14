# Test Suite Report

**Generated:** 2025-12-14  
**Execution Date:** 2025-12-14 01:29:29 AM -03  
**Total Duration:** 71m 36s  
**Test Runner:** File-based execution with 5-minute timeout per file

---

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Test Files** | 156 | 100% |
| **Total Tests** | 1,877 | 100% |
| ✅ **Passed Tests** | 1,643 | 87.5% |
| ❌ **Failed Tests** | 197 | 10.5% |
| ⏭️ **Skipped Tests** | 37 | 2.0% |
| | | |
| ✅ **Passed Files** | 124 | 79.5% |
| ❌ **Failed Files** | 29 | 18.6% |
| ⏭️ **Skipped Files** | 2 | 1.3% |
| ⏱️ **Timeout Files** | 1 | 0.6% |

### Key Findings

- **Overall Health:** 87.5% pass rate - strong foundation with specific fixable issues
- **Improvement Trend:** +2.1% pass rate improvement from previous run (85.4% → 87.5%)
- **Primary Blocker:** 1 timeout in embedding integration tests
- **Root Causes Identified:** Most failures stem from 6 main categories
- **Duration Improvement:** 2m 11s faster than previous run (73m 47s → 71m 36s)

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
   LINE 2: INSERT INTO documents (document_id, filename...
   ```
   - Issue: Test assumes tables exist before migration
   - Needs proper test setup with database initialization

3. **test_rollback_functionality:**
   ```
   sqlalchemy.exc.NoSuchTableError: chunks
   ```
   - Issue: Table doesn't exist after rollback attempt
   - Migration state inconsistency

**Recommendations:**
1. Fix migration downgrade scripts to check for existence before dropping
2. Improve test fixtures to ensure proper database state
3. Add `IF EXISTS` clauses to DROP statements in migrations
4. Review migration order and dependencies
5. Ensure test database is properly initialized before migration tests

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
   - Model B generating 49 samples (just below minimum)

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
2. Verify model registration and loading process
3. Check test data generation and routing logic
4. Add detailed logging to track sample collection flow
5. Review model assignment logic in A/B testing framework

---

### 3. ONNX Embeddings Performance ⚡ **MEDIUM PRIORITY**

**Impact:** 2 test failures  
**Error:** Performance targets not met

**Affected Test Files:**
- `tests/integration/services/test_onnx_embeddings_integration.py` (2 failures)

**Specific Failure:**
```python
test_performance_target:
  AssertionError: Average time 2932.54ms exceeds 100ms target
  assert np.float64(2.9325386581011115) < 0.1
```

**Analysis:**
- Target: 100ms per embedding
- Actual: 2,932ms per embedding (29.3x slower)
- This is a performance benchmark, not a functional failure
- 9/11 functional tests passing
- Performance improved from previous run (3,069ms → 2,932ms)

**Recommendations:**
1. Review if 100ms target is realistic for ONNX embeddings
2. Consider adjusting threshold to 3-5 seconds for ONNX
3. Separate performance benchmarks from functional tests
4. Mark as `@pytest.mark.slow` or `@pytest.mark.performance`
5. Investigate ONNX optimization opportunities (model quantization, batch processing)

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
test_cohere_full_workflow SKIPPED (missing API key)
test_local_embedding_provider PASSED
test_large_batch_processing [TIMEOUT]
```

**Recommendations:**
1. Investigate `test_large_batch_processing` for infinite loops or blocking calls
2. Add timeout decorators to individual tests (`@pytest.mark.timeout(60)`)
3. Consider splitting large batch tests into smaller units
4. Add logging to identify where the hang occurs
5. Review batch processing logic for deadlocks or resource contention

---

### 5. Strategy Integration Tests 🧩 **MEDIUM PRIORITY**

**Impact:** 84+ test failures across 7 files  
**Primary Cause:** Missing LLM services, API configuration issues, and implementation bugs

**Affected Test Files:**
- `tests/integration/strategies/test_query_expansion_integration.py` (18 failures)
- `tests/integration/strategies/test_multi_query_integration.py` (16 failures)
- `tests/integration/strategies/test_contextual_integration.py` (12 failures)
- `tests/integration/strategies/test_late_chunking_integration.py` (12 failures)
- `tests/integration/strategies/test_hierarchical_integration.py` (8 failures)
- `tests/integration/strategies/test_knowledge_graph_integration.py` (8 failures)

**Common Error Patterns:**

#### 5.1. Contextual Strategy - Zero Chunks Indexed
```python
test_contextual_retrieval_complete_workflow:
  assert 0 == 20  # Expected 20 chunks, got 0
test_cost_tracking_accuracy:
  assert 0.0 > 0  # No cost tracked
test_large_document_processing:
  assert 0 == 100  # Expected 100 chunks, got 0
```
- **Issue:** Contextual indexing strategy not processing chunks
- **Root Cause:** LLM service not generating context or processing pipeline failure
- **Files:** All 12 failures in `test_contextual_integration.py`

#### 5.2. Hierarchical Strategy - UUID Parsing Error
```python
ValueError: badly formed hexadecimal UUID string
```
- **Issue:** UUID format mismatch in hierarchical chunk processing
- **Affected:** All 8 failures in `test_hierarchical_integration.py`
- **Root Cause:** Incorrect UUID generation or parsing in hierarchy builder

#### 5.3. Knowledge Graph - NoneType Attribute Error
```python
AttributeError: 'NoneType' object has no attribute 'search'
```
- **Issue:** Vector store service not initialized
- **Affected:** 2 failures in `test_knowledge_graph_integration.py`
- **Root Cause:** Missing service dependency injection

#### 5.4. Multi-Query Strategy - API Mismatch
```python
TypeError: MultiQueryRAGStrategy.__init__() got an unexpected keyword argument 'vector_store_service'
```
- **Issue:** Tests using deprecated parameter names
- **Affected:** Multiple failures in `test_multi_query_integration.py`
- **Root Cause:** API changes not reflected in tests

#### 5.5. Query Expansion - LM Studio Empty Responses
- **Symptom:** LM Studio connected successfully, tokens generated, but empty query results
- **Prompt tokens:** 90-111 (receiving prompts)
- **Completion tokens:** 150 (generating tokens)
- **But:** `expanded_query` is empty string
- **Issue:** LM Studio model configuration or output parsing
- **Affected:** 18 failures in `test_query_expansion_integration.py`

**Recommendations:**
1. **Contextual Strategy:** Debug chunk processing pipeline and LLM context generation
2. **Hierarchical Strategy:** Fix UUID generation/parsing in hierarchy builder
3. **Knowledge Graph:** Ensure proper service dependency injection
4. **Multi-Query:** Update test code to use correct API parameters
5. **Query Expansion:** Fix LM Studio model configuration and output parsing
6. Add pytest markers: `@pytest.mark.requires_llm_api`, `@pytest.mark.requires_database`
7. Implement proper mocking for CI environments

---

### 6. Service Integration & Mocking Issues 🔧 **MEDIUM PRIORITY**

**Impact:** 4 test failures  
**Errors:**
- AsyncMock cannot be awaited
- Coroutine protocol violations
- Service configuration issues

**Affected Test Files:**
- `tests/integration/services/test_service_implementations.py` (2 failures)
- `tests/integration/services/test_service_integration.py` (2 failures)

**Specific Issues:**

1. **PostgreSQL Service Mock Error:**
   ```python
   TypeError: object AsyncMock can't be used in 'await' expression
   ```
   - Location: `test_postgresql_database_service_basic_functionality`
   - Issue: Incorrect async mock configuration
   - Fix: Use `AsyncMock(return_value=...)` properly

2. **Database Consistency Test:**
   ```python
   TypeError: 'coroutine' object does not support the asynchronous context manager protocol
   ```
   - Location: `test_embedding_database_consistency`
   - Issue: Pool acquisition not properly mocked
   - Fix: Mock `_get_pool()` to return an async context manager

3. **LLM Service Setup Error:**
   ```python
   AttributeError: <module 'rag_factory.services.llm.service'> does not have the attribute 'AnthropicProvider'
   ```
   - Location: `test_rag_workflow` setup
   - Issue: Incorrect import path in test
   - Fix: Update import to `rag_factory.services.llm.providers.anthropic.AnthropicProvider`

**Recommendations:**
1. Fix async mock setup in database service tests
2. Use `unittest.mock.AsyncMock` properly with async context managers
3. Update service import paths in tests
4. Review service configuration requirements
5. Add better error messages for missing dependencies

---

### 7. Late Chunking Integration 📄 **MEDIUM PRIORITY**

**Impact:** 12 test failures  
**Error Types:** Mixed - embeddings, document processing, coherence analysis

**Affected Test Files:**
- `tests/integration/strategies/test_late_chunking_integration.py` (12 failures)

**Common Issues:**
- Document embedding generation failures
- Coherence score calculation errors
- Chunk boundary detection problems
- Integration with embedding services

**Recommendations:**
1. Debug document embedder initialization
2. Verify embedding service integration
3. Fix coherence analyzer logic
4. Review chunk boundary detection algorithm

---

## 📊 Summary by Test Category

### Integration Tests (35 files)

| Category | Files | Passed | Failed | Skipped | Timeout | Key Issues |
|----------|-------|--------|--------|---------|---------|------------|
| Strategies | 8 | 1 | 7 | 0 | 0 | LLM APIs, dependencies, implementation bugs |
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

**1. Fix Contextual Strategy Chunk Processing**
- **Files:** `tests/integration/strategies/test_contextual_integration.py` (12 failures)
- **Issue:** Zero chunks being indexed
- **Impact:** Unblocks contextual strategy
- **Effort:** 2 days
- **Priority:** 🟠 HIGH

**2. Fix Hierarchical Strategy UUID Handling**
- **File:** `tests/integration/strategies/test_hierarchical_integration.py` (8 failures)
- **Issue:** UUID parsing errors
- **Impact:** Unblocks hierarchical strategy
- **Effort:** 1 day
- **Priority:** 🟠 HIGH

**3. Fix Async Mock in Service Tests**
- **File:** `tests/integration/services/test_service_implementations.py`
- **Impact:** Fixes 2 tests
- **Effort:** 0.5 days
- **Priority:** 🟠 HIGH

**4. Fix Factory Service Instantiation**
- **Files:** `tests/integration/test_factory_integration.py` (20 failures)
- **Impact:** Unblocks factory tests
- **Effort:** 2 days
- **Priority:** 🟠 HIGH

---

### Phase 3: Strategy Integration Tests (Blocks ~60 tests) 🧩

**1. Fix Query Expansion LM Studio Configuration**
- **Issue:** Empty responses from LM Studio
- **Impact:** Fixes 18 tests
- **Effort:** 1-2 days
- **Priority:** 🟡 MEDIUM

**2. Update Multi-Query Strategy Test APIs**
- **Issue:** Tests using wrong parameter names
- **Impact:** Fixes 16 tests
- **Effort:** 1 day
- **Priority:** 🟡 MEDIUM

**3. Fix Late Chunking Integration**
- **Issue:** Document embedding and coherence analysis
- **Impact:** Fixes 12 tests
- **Effort:** 2 days
- **Priority:** 🟡 MEDIUM

**4. Fix Knowledge Graph Service Dependencies**
- **Issue:** NoneType attribute errors
- **Impact:** Fixes 8 tests
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
| **Current** | - | 87.5% | - |
| **Phase 1** | ~70 | ~91% | 4-5 days |
| **Phase 2** | ~40 | ~93% | 7-10 days |
| **Phase 3** | ~60 | ~96% | 10-14 days |
| **Phase 4** | ~27 | ~98%+ | 12-17 days |

---

## 🔍 Detailed Test File Status

### ✅ Passing Files (124 files)

**Highlights:**
- All CLI integration tests (3 files, 14 tests)
- All repository integration tests (1 file, 19 tests)
- All observability tests (1 file, 11 tests)
- All evaluation tests (1 file, 7 tests)
- 101 unit test files passing
- 2 benchmark files passing (14 tests)

### ❌ Failed Files (29 files)

#### Integration Tests (14 files)
1. `tests/integration/database/test_migration_integration.py` - Migration downgrade issues
2. `tests/integration/models/test_fine_tuned_embeddings_integration.py` - A/B testing sample collection
3. `tests/integration/services/test_onnx_embeddings_integration.py` - Performance targets
4. `tests/integration/services/test_service_implementations.py` - Async mocks
5. `tests/integration/services/test_service_integration.py` - Service configuration
6. `tests/integration/strategies/test_contextual_integration.py` - Zero chunks indexed
7. `tests/integration/strategies/test_hierarchical_integration.py` - UUID parsing
8. `tests/integration/strategies/test_knowledge_graph_integration.py` - Service dependencies
9. `tests/integration/strategies/test_late_chunking_integration.py` - Embedding integration
10. `tests/integration/strategies/test_multi_query_integration.py` - API mismatch
11. `tests/integration/strategies/test_query_expansion_integration.py` - LM Studio config
12. `tests/integration/test_config_integration.py` - Config validation
13. `tests/integration/test_factory_integration.py` - Service instantiation
14. `tests/integration/test_package_integration.py` - numpy dependency
15. `tests/integration/test_pipeline_integration.py` - Pipeline config

#### Unit Tests (15 files)
16. `tests/unit/test_pipeline.py` - ⚠️ 46 failures - CRITICAL
17. `tests/unit/strategies/agentic/test_strategy.py` - 16 failures
18. `tests/unit/documentation/test_links.py` - 6 failures
19. `tests/unit/documentation/test_code_examples.py` - 4 failures
20. `tests/unit/cli/test_check_consistency_command.py` - 2 failures
21. `tests/unit/cli/test_repl_command.py` - 2 failures
22. `tests/unit/services/test_interfaces.py` - 2 failures
23. `tests/unit/services/test_database_service.py` - 2 failures
24. `tests/unit/documentation/test_documentation_completeness.py` - 2 failures
25. `tests/unit/database/test_migrations.py` - Multiple failures
26. `tests/unit/repositories/test_chunk_repository.py` - Multiple failures
27. `tests/unit/services/embeddings/test_onnx_local.py` - Multiple failures
28. `tests/unit/services/embedding/test_onnx_local_provider.py` - Multiple failures
29. `tests/unit/strategies/late_chunking/test_document_embedder.py` - 1 failure

### ⏱️ Timeout Files (1)
- `tests/integration/services/test_embedding_integration.py` - Exceeded 5-minute limit

### ⏭️ Skipped Files (2)
- `tests/benchmarks/test_model_comparison_performance.py` - 6 tests skipped (missing dependencies)
- `tests/integration/documentation/test_documentation_integration.py` - 6 tests skipped

---

## 💡 Recommendations

### Immediate Actions (Configuration Issues)
1. ✅ **CRITICAL:** Investigate `tests/unit/test_pipeline.py` (46 failures)
2. 🔧 Fix embedding integration timeout in `test_large_batch_processing`
3. 🗄️ Fix migration downgrade scripts with `IF EXISTS` clauses
4. 🧪 Debug A/B testing sample collection logic
5. 📦 Verify numpy is in core dependencies

### Short-Term Improvements (Implementation Bugs)
1. Fix contextual strategy chunk processing pipeline
2. Fix hierarchical strategy UUID generation/parsing
3. Update multi-query strategy test APIs
4. Fix LM Studio model configuration and output parsing
5. Implement proper async mocking patterns
6. Fix knowledge graph service dependency injection

### Long-Term Enhancements (Test Infrastructure)
1. Add pytest markers for test categorization (`@pytest.mark.requires_llm_api`, etc.)
2. Separate performance tests from functional tests
3. Improve CI/CD test organization
4. Add comprehensive test documentation
5. Implement better test fixtures and cleanup
6. Increase test coverage to 50%+
7. Create test environment templates

---

## 📝 Notes

- **Test Execution:** File-based with 5-minute timeout per file
- **Coverage:** 16-17% overall (needs improvement)
- **Environment:** Tests run in virtual environment
- **Database:** PostgreSQL with pgvector extension
- **LLM Services:** Most integration tests require API keys or LM Studio (expected)
- **Improvement:** +2.1% pass rate from previous run (85.4% → 87.5%)
- **Duration:** 2m 11s faster than previous run (73m 47s → 71m 36s)
- **Test Count:** -15 tests from previous run (1,892 → 1,877)

---

## Test Coverage Analysis

**Current Coverage:** 16-17% (11,084 statements, ~9,200 missed)

**Areas Needing Coverage:**
- Evaluation modules: 0% coverage
- Strategy implementations: 0-20% coverage
- Service providers: 20-40% coverage
- Agentic strategies: 0% coverage
- Knowledge graph: 0% coverage
- Multi-query: 0% coverage
- Contextual: 0% coverage
- Late chunking: 0% coverage

**Well-Covered Areas:**
- Core capabilities: 74% coverage
- Base strategies: 76% coverage
- Reranking base: 76% coverage
- Query expansion base: 85% coverage
- Embedding base: 81% coverage
- LLM base: 100% coverage
- Service interfaces: 100% coverage

---

## Failing Test Files List

### Integration Test Failures (14 files)
1. `tests/integration/database/test_migration_integration.py`
2. `tests/integration/models/test_fine_tuned_embeddings_integration.py`
3. `tests/integration/services/test_onnx_embeddings_integration.py`
4. `tests/integration/services/test_service_implementations.py`
5. `tests/integration/services/test_service_integration.py`
6. `tests/integration/strategies/test_contextual_integration.py`
7. `tests/integration/strategies/test_hierarchical_integration.py`
8. `tests/integration/strategies/test_knowledge_graph_integration.py`
9. `tests/integration/strategies/test_late_chunking_integration.py`
10. `tests/integration/strategies/test_multi_query_integration.py`
11. `tests/integration/strategies/test_query_expansion_integration.py`
12. `tests/integration/test_config_integration.py`
13. `tests/integration/test_factory_integration.py`
14. `tests/integration/test_package_integration.py`
15. `tests/integration/test_pipeline_integration.py`

### Unit Test Failures (15 files)
1. `tests/unit/cli/test_check_consistency_command.py`
2. `tests/unit/cli/test_repl_command.py`
3. `tests/unit/database/test_migrations.py`
4. `tests/unit/documentation/test_code_examples.py`
5. `tests/unit/documentation/test_documentation_completeness.py`
6. `tests/unit/documentation/test_links.py`
7. `tests/unit/repositories/test_chunk_repository.py`
8. `tests/unit/services/embedding/test_onnx_local_provider.py`
9. `tests/unit/services/embeddings/test_onnx_local.py`
10. `tests/unit/services/test_database_service.py`
11. `tests/unit/services/test_interfaces.py`
12. `tests/unit/strategies/agentic/test_strategy.py`
13. `tests/unit/strategies/late_chunking/test_document_embedder.py`
14. `tests/unit/test_pipeline.py` ⚠️ **CRITICAL - 46 failures**

---

## Single Failing Tests (Detailed)

### Database Migration Tests (6 failures)
1. `tests/integration/database/test_migration_integration.py::TestMigrationIntegration::test_real_migration_execution`
2. `tests/integration/database/test_migration_integration.py::TestMigrationIntegration::test_migration_with_existing_data`
3. `tests/integration/database/test_migration_integration.py::TestMigrationIntegration::test_rollback_functionality`

### Fine-Tuned Embeddings Tests (4 failures)
4. `tests/integration/models/test_fine_tuned_embeddings_integration.py::test_ab_testing_workflow`
5. `tests/integration/models/test_fine_tuned_embeddings_integration.py::test_model_comparison_workflow`

### ONNX Embeddings Tests (2 failures)
6. `tests/integration/services/test_onnx_embeddings_integration.py::TestONNXEmbeddingsIntegration::test_performance_target`

### Service Implementation Tests (2 failures)
7. `tests/integration/services/test_service_implementations.py::TestDatabaseServices::test_postgresql_database_service_basic_functionality`

### Service Integration Tests (2 failures)
8. `tests/integration/services/test_service_integration.py::test_embedding_database_consistency`
9. `tests/integration/services/test_service_integration.py::test_rag_workflow` (ERROR)

### Contextual Strategy Tests (12 failures)
10. `tests/integration/strategies/test_contextual_integration.py::test_contextual_retrieval_complete_workflow`
11. `tests/integration/strategies/test_contextual_integration.py::test_cost_tracking_accuracy`
12. `tests/integration/strategies/test_contextual_integration.py::test_retrieval_with_different_formats`
13. `tests/integration/strategies/test_contextual_integration.py::test_error_recovery`
14. `tests/integration/strategies/test_contextual_integration.py::test_synchronous_indexing`
15. `tests/integration/strategies/test_contextual_integration.py::test_large_document_processing`

### Hierarchical Strategy Tests (8 failures)
16. `tests/integration/strategies/test_hierarchical_integration.py::TestHierarchicalIntegration::test_end_to_end_workflow`
17. `tests/integration/strategies/test_hierarchical_integration.py::TestHierarchicalIntegration::test_expansion_strategy_comparison`
18. `tests/integration/strategies/test_hierarchical_integration.py::TestHierarchicalIntegration::test_hierarchy_validation`
19. `tests/integration/strategies/test_hierarchical_integration.py::TestHierarchicalIntegration::test_multiple_documents`

### Knowledge Graph Tests (8 failures)
20. `tests/integration/strategies/test_knowledge_graph_integration.py::test_hybrid_retrieval`
21. `tests/integration/strategies/test_knowledge_graph_integration.py::test_relationship_queries`

### Late Chunking Tests (12 failures)
22-33. Multiple tests in `tests/integration/strategies/test_late_chunking_integration.py`

### Multi-Query Tests (16 failures)
34-49. Multiple tests in `tests/integration/strategies/test_multi_query_integration.py`

### Query Expansion Tests (18 failures)
50-67. Multiple tests in `tests/integration/strategies/test_query_expansion_integration.py`

### Unit Test Failures (130+ failures)
68-197. Distributed across 15 unit test files, with `tests/unit/test_pipeline.py` containing 46 failures alone

---

**Report End** | For detailed logs, see `test_results_by_file.txt` and `test_summary_by_file.txt`
