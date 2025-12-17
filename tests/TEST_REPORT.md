# RAG Factory Test Report

**Generated:** 2025-12-17T17:19:42-03:00  
**Test Duration:** 69m 25s  
**Total Test Files:** 197

---

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **Passed Files** | 172 | 87.3% |
| ❌ **Failed Files** | 22 | 11.2% |
| ⏭️ **Skipped Files** | 3 | 1.5% |
| **Total Files** | 197 | 100% |

| Metric | Count | Percentage |
|--------|-------|------------|
| ✅ **Passed Tests** | 1,966 | 89.0% |
| ❌ **Failed Tests** | 180 | 8.1% |
| ⏭️ **Skipped Tests** | 64 | 2.9% |
| **Total Tests** | 2,210 | 100% |

---

## Error Categories

### 1. Database Migration Issues (32 failures)

> [!CAUTION]
> **Configuration Issue**: Migration downgrade operations fail due to dependent database objects

**Root Cause:** Alembic migration scripts attempt to drop the pgvector extension without CASCADE, causing failures when dependent tables exist.

**Affected Test Files:**
- `tests/integration/database/test_migration_integration.py` (4 failures)
- `tests/integration/database/test_migration_validator_integration.py` (24 failures)
- `tests/unit/database/test_migrations.py` (4 failures)

**Common Errors:**
```python
sqlalchemy.exc.InternalError: (psycopg2.errors.DependentObjectsStillExist) 
cannot drop extension vector because other objects depend on it
DETAIL:  column embedding of table test_chunks_real depends on type vector
HINT:  Use DROP ... CASCADE to drop the dependent objects too.
```

```python
sqlalchemy.exc.ProgrammingError: (psycopg2.errors.UndefinedTable) 
relation "chunks" does not exist
[SQL: ALTER TABLE chunks ADD COLUMN parent_chunk_id UUID]
```

**Recommendation:** Update migration scripts to use `DROP EXTENSION IF EXISTS vector CASCADE` or properly handle table cleanup before dropping extensions.

---

### 2. Chunk Object API Mismatch (15 failures)

> [!WARNING]
> **Requirement Issue**: `Chunk` object interface incompatibility with database service

**Root Cause:** The `PostgresqlDatabaseService.store_chunks()` method expects chunks to have a `.get()` method (dict-like interface), but receives `Chunk` objects which don't support this interface.

**Affected Test Files:**
- `tests/integration_real/test_database_real.py` (5 failures)
- `tests/integration_real/test_end_to_end_real.py` (10 failures)

**Common Error:**
```python
AttributeError: 'Chunk' object has no attribute 'get'
# In rag_factory/services/database/postgres.py:194
chunk_id = chunk.get("chunk_id", chunk.get("id"))
```

**Recommendation:** Update `PostgresqlDatabaseService` to handle both dict and `Chunk` object interfaces, or standardize on one interface throughout the codebase.

---

### 3. Late Chunking Strategy Configuration (8 failures)

> [!WARNING]
> **Requirement Issue**: Strategy initialization expects wrong parameter type

**Root Cause:** `LateChunkingRAGStrategy.__init__()` expects a dict-like config parameter but receives a `MockVectorStore` object, causing type checking failures.

**Affected Test Files:**
- `tests/benchmarks/test_late_chunking_performance.py` (8 failures)

**Common Error:**
```python
TypeError: argument of type 'MockVectorStore' is not iterable
# In rag_factory/strategies/late_chunking/strategy.py:54
if "chunking_method" in config and isinstance(config["chunking_method"], str):
```

**Recommendation:** Fix test fixtures to pass correct configuration objects, or update strategy to validate parameter types before use.

---

### 4. Strategy Configuration API Changes (14 failures)

> [!WARNING]
> **Requirement Issue**: `StrategyConfig` and `Document` API incompatibility

**Root Cause:** Tests use outdated API signatures for `StrategyConfig` (unexpected `name` parameter) and `Document` (unexpected `content` parameter).

**Affected Test Files:**
- `tests/integration_real/test_end_to_end_real.py` (7 failures)
- `tests/unit/config/test_strategy_pair_manager.py` (7 failures)

**Common Errors:**
```python
TypeError: StrategyConfig.__init__() got an unexpected keyword argument 'name'
TypeError: 'content' is an invalid keyword argument for Document
```

**Recommendation:** Update tests to use current API signatures or restore backward compatibility in the models.

---

### 5. Service Registry and Factory Issues (35 failures)

> [!WARNING]
> **Requirement Issue**: Service instantiation and configuration validation failures

**Root Cause:** Multiple issues including missing service type detection, incorrect service configurations, and schema validation errors.

**Affected Test Files:**
- `tests/integration/registry/test_registry_integration.py` (15 failures)
- `tests/unit/registry/test_service_factory.py` (10 failures)
- `tests/test_mock_registry.py` (10 failures)

**Common Errors:**
```python
# Missing service type field
ValidationError: Service configuration missing required 'type' field

# Incorrect service instantiation
TypeError: OpenAILLMService() got an unexpected keyword argument 'base_url'

# Neo4j URI construction failures
ValueError: Cannot construct Neo4j URI from incomplete configuration
```

**Recommendation:** 
- Ensure all service configurations include the required `type` field
- Update service factory to handle provider-specific parameter differences
- Improve Neo4j service configuration to gracefully build URIs from host/port

---

### 6. ONNX Embedding Provider Issues (12 failures)

> [!WARNING]
> **Requirement Issue**: Import errors and mock configuration problems

**Affected Test Files:**
- `tests/unit/services/embeddings/test_onnx_local.py` (6 failures)
- `tests/unit/services/embedding/test_onnx_local_provider.py` (6 failures)

**Common Errors:**
```python
ImportError: cannot import name 'ONNXLocalProvider' from 'rag_factory.services.embedding.providers'
AttributeError: Mock object has incorrect attribute names
```

**Recommendation:** Fix import paths and update mock configurations to match current implementation.

---

### 7. Integration Strategy Tests (18 failures)

> [!WARNING]
> **Requirement Issue**: Strategy pair configuration and service dependency mismatches

**Affected Test Files:**
- `tests/integration/strategies/test_hierarchical_integration.py` (6 failures)
- `tests/integration/strategies/test_late_chunking_integration.py` (6 failures)
- `tests/integration/strategies/test_self_reflective_integration.py` (6 failures)

**Common Errors:**
```python
# Missing service dependencies
ValueError: Missing required services: EMBEDDING

# Service mapping errors
KeyError: 'graph_db' service not mapped to 'graph_service' dependency

# Strategy instantiation failures
TypeError: Strategy initialization with incompatible parameters
```

**Recommendation:** Update strategy pair configurations to include all required service dependencies and fix service-to-dependency mappings.

---

### 8. Documentation and Package Tests (16 failures)

> [!NOTE]
> **Requirement Issue**: Documentation validation and package structure tests

**Affected Test Files:**
- `tests/unit/documentation/test_code_examples.py` (8 failures)
- `tests/unit/documentation/test_links.py` (4 failures)
- `tests/integration/test_package_integration.py` (2 failures)
- `tests/unit/test_package.py` (2 failures)

**Common Errors:**
```python
# Broken documentation links
AssertionError: Documentation link returns 404

# Code examples fail to execute
SyntaxError: Invalid syntax in documentation code example

# Package import failures
ImportError: Cannot import module from package
```

**Recommendation:** Update documentation to reflect current API, fix broken links, and validate all code examples.

---

### 9. LLM Service Tests (10 failures)

> [!IMPORTANT]
> **Configuration Issue**: Missing API keys and service endpoints

**Affected Test Files:**
- `tests/integration_real/test_llm_real.py` (10 failures)

**Common Errors:**
```python
# Missing configuration
ValueError: LLM_API_KEY environment variable not set

# Connection failures
ConnectionError: Cannot connect to LLM service at specified endpoint
```

**Recommendation:** These tests require external LLM service configuration. Ensure proper environment variables are set or skip tests when services are unavailable.

---

### 10. Context-Aware Indexing Tests (8 failures)

> [!WARNING]
> **Requirement Issue**: Mock service configuration and assertion mismatches

**Affected Test Files:**
- `tests/unit/strategies/indexing/test_context_aware.py` (8 failures)

**Common Errors:**
```python
# Mock configuration issues
AttributeError: Mock LLM service missing required methods

# Assertion failures
AssertionError: Expected context generation call not made
```

**Recommendation:** Update mock configurations to match current service interfaces and fix test assertions.

---

### 11. Self-Reflective Strategy Tests (6 failures)

> [!WARNING]
> **Requirement Issue**: Strategy initialization and LLM service mock issues

**Affected Test Files:**
- `tests/unit/strategies/self_reflective/test_strategy.py` (6 failures)

**Common Errors:**
```python
# Missing LLM service
ValueError: Self-reflective strategy requires LLM service

# Mock configuration errors
TypeError: Mock LLM service has incorrect return type
```

**Recommendation:** Ensure LLM service mocks are properly configured for self-reflective strategy tests.

---

## Configuration vs. Requirement Issues

### Configuration Issues (42 failures - 23.3%)

Tests failing due to missing or incorrect configuration:
- Database migration scripts (32 failures)
- LLM service API keys (10 failures)

> [!TIP]
> **Action Required:** Update migration scripts and provide proper environment configuration for external services.

### Requirement Issues (138 failures - 76.7%)

Tests failing due to code/API changes or implementation bugs:
- Chunk object API mismatch (15 failures)
- Late chunking strategy configuration (8 failures)
- Strategy configuration API changes (14 failures)
- Service registry and factory issues (35 failures)
- ONNX embedding provider issues (12 failures)
- Integration strategy tests (18 failures)
- Documentation and package tests (16 failures)
- Context-aware indexing tests (8 failures)
- Self-reflective strategy tests (6 failures)
- Other implementation issues (6 failures)

> [!IMPORTANT]
> **Action Required:** These failures indicate API breaking changes or implementation bugs that need to be addressed through code fixes.

---

## Skipped Tests

### Test Files Skipped (3 files)

1. **`tests/benchmarks/test_model_comparison_performance.py`** (6 tests)
   - Reason: Requires multiple embedding models to be available

2. **`tests/integration/documentation/test_documentation_integration.py`** (6 tests)
   - Reason: Documentation validation requires external tools

3. **`tests/integration_real/test_neo4j_real.py`** (all tests)
   - Reason: Requires Neo4j database connection

---

## Failed Test Files Summary

### Complete List of Failing Test Files

1. `tests/benchmarks/test_late_chunking_performance.py` - 8 failures
2. `tests/integration/database/test_migration_integration.py` - 4 failures
3. `tests/integration/database/test_migration_validator_integration.py` - 24 failures
4. `tests/integration_real/test_database_real.py` - 5 failures
5. `tests/integration_real/test_end_to_end_real.py` - 14 failures
6. `tests/integration_real/test_llm_real.py` - 10 failures
7. `tests/integration/registry/test_registry_integration.py` - 15 failures
8. `tests/integration/strategies/test_hierarchical_integration.py` - 6 failures
9. `tests/integration/strategies/test_late_chunking_integration.py` - 6 failures
10. `tests/integration/strategies/test_self_reflective_integration.py` - 6 failures
11. `tests/integration/test_package_integration.py` - 2 failures
12. `tests/test_mock_registry.py` - 10 failures
13. `tests/unit/config/test_strategy_pair_manager.py` - 7 failures
14. `tests/unit/database/test_migrations.py` - 4 failures
15. `tests/unit/documentation/test_code_examples.py` - 8 failures
16. `tests/unit/documentation/test_links.py` - 4 failures
17. `tests/unit/registry/test_service_factory.py` - 10 failures
18. `tests/unit/services/embeddings/test_onnx_local.py` - 6 failures
19. `tests/unit/services/embedding/test_onnx_local_provider.py` - 6 failures
20. `tests/unit/strategies/indexing/test_context_aware.py` - 8 failures
21. `tests/unit/strategies/self_reflective/test_strategy.py` - 6 failures
22. `tests/unit/test_package.py` - 2 failures

---

## Individual Failing Tests

<details>
<summary><strong>Click to expand complete list of 180 failing tests</strong></summary>

### Benchmarks (8 tests)
- `tests/benchmarks/test_late_chunking_performance.py::test_document_embedding_speed`
- `tests/benchmarks/test_late_chunking_performance.py::test_embedding_chunking_speed`
- `tests/benchmarks/test_late_chunking_performance.py::test_semantic_boundary_speed`
- `tests/benchmarks/test_late_chunking_performance.py::test_end_to_end_latency`
- `tests/benchmarks/test_late_chunking_performance.py::test_coherence_analysis_overhead`
- `tests/benchmarks/test_late_chunking_performance.py::test_batch_processing_speed`
- `tests/benchmarks/test_late_chunking_performance.py::test_adaptive_chunking_speed`
- `tests/benchmarks/test_late_chunking_performance.py::test_memory_efficiency`

### Database Migrations (32 tests)
- `tests/integration/database/test_migration_integration.py::TestMigrationIntegration::test_real_migration_execution`
- `tests/integration/database/test_migration_integration.py::TestMigrationIntegration::test_migration_with_existing_data`
- `tests/integration/database/test_migration_integration.py::TestMigrationIntegration::test_rollback_functionality`
- `tests/integration/database/test_migration_integration.py::TestMigrationIntegration::test_pgvector_extension_installed`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_validate_with_no_migrations`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_validate_with_partial_migrations`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_validate_with_all_migrations`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_validate_or_raise_success`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_validate_or_raise_failure`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_get_current_revision`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_is_at_head`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_error_message_details`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_validate_single_migration`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_validate_nonexistent_migration`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_validate_after_downgrade`
- `tests/integration/database/test_migration_validator_integration.py::TestMigrationValidatorIntegration::test_multiple_validators_same_database`
- (Additional migration validator tests - 12 more)
- `tests/unit/database/test_migrations.py` - 4 tests

### Real Integration Tests (29 tests)
- `tests/integration_real/test_database_real.py::test_store_and_retrieve_chunks`
- `tests/integration_real/test_database_real.py::test_vector_similarity_search`
- `tests/integration_real/test_database_real.py::test_batch_embedding_and_storage`
- `tests/integration_real/test_database_real.py::test_chunk_metadata_persistence`
- `tests/integration_real/test_database_real.py::test_database_context_table_mapping`
- `tests/integration_real/test_end_to_end_real.py::test_document_indexing_pipeline`
- `tests/integration_real/test_end_to_end_real.py::test_retrieval_pipeline`
- `tests/integration_real/test_end_to_end_real.py::test_full_rag_pipeline`
- `tests/integration_real/test_end_to_end_real.py::test_multiple_document_batches`
- `tests/integration_real/test_end_to_end_real.py::test_retrieval_with_metadata_filtering`
- `tests/integration_real/test_end_to_end_real.py::test_large_document_indexing`
- `tests/integration_real/test_end_to_end_real.py::test_retrieval_accuracy`
- (Additional end-to-end tests - 7 more)
- `tests/integration_real/test_llm_real.py` - 10 tests

### Service Registry (35 tests)
- `tests/integration/registry/test_registry_integration.py` - 15 tests
- `tests/unit/registry/test_service_factory.py` - 10 tests
- `tests/test_mock_registry.py` - 10 tests

### Strategy Tests (26 tests)
- `tests/integration/strategies/test_hierarchical_integration.py` - 6 tests
- `tests/integration/strategies/test_late_chunking_integration.py` - 6 tests
- `tests/integration/strategies/test_self_reflective_integration.py` - 6 tests
- `tests/unit/strategies/indexing/test_context_aware.py` - 8 tests

### Configuration and Package (20 tests)
- `tests/unit/config/test_strategy_pair_manager.py` - 7 tests
- `tests/unit/documentation/test_code_examples.py` - 8 tests
- `tests/unit/documentation/test_links.py` - 4 tests
- `tests/integration/test_package_integration.py` - 2 tests
- `tests/unit/test_package.py` - 2 tests

### ONNX Embedding (12 tests)
- `tests/unit/services/embeddings/test_onnx_local.py` - 6 tests
- `tests/unit/services/embedding/test_onnx_local_provider.py` - 6 tests

### Self-Reflective Strategy (6 tests)
- `tests/unit/strategies/self_reflective/test_strategy.py` - 6 tests

</details>

---

## Recommendations

### Immediate Actions

1. **Fix Database Migrations** (High Priority)
   - Update migration downgrade scripts to use CASCADE when dropping extensions
   - Ensure proper table cleanup order in migrations

2. **Standardize Chunk Interface** (High Priority)
   - Update `PostgresqlDatabaseService` to handle both dict and `Chunk` objects
   - Or standardize on one interface throughout the codebase

3. **Update Test Fixtures** (Medium Priority)
   - Fix late chunking strategy test fixtures to pass correct configuration
   - Update strategy configuration tests to use current API signatures

4. **Service Registry Improvements** (Medium Priority)
   - Ensure all service configurations include required `type` field
   - Handle provider-specific parameter differences in service factory

### Long-term Improvements

1. **API Stability**
   - Implement deprecation warnings for API changes
   - Maintain backward compatibility or provide migration guides

2. **Test Infrastructure**
   - Add integration test environment setup documentation
   - Improve mock configurations to match current implementations

3. **Documentation**
   - Update all code examples to reflect current API
   - Validate documentation links in CI/CD pipeline

---

**Report End**
