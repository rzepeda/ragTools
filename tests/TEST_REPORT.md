# Test Execution Report

## Executive Summary

**Test Log:** tests/test_results_by_file.txt
**Execution Date:** Mon Dec 15 01:49:02 AM -03 2025  
**Total Duration:** 68m 17s

| Metric | Count | Percentage |
| :--- | :--- | :--- |
| **Total Test Files** | 156 | 100% |
| ✅ **Passed Files** | 142 | 91.0% |
| ❌ **Failed Files** | 12 | 7.7% |
| ⏭️ **Skipped Files** | 2 | 1.3% |
| ⏱️ **Timeout Files** | 0 | 0.0% |

| Test Results | Count | Percentage |
| :--- | :--- | :--- |
| **Total Tests** | 1821 | 100% |
| ✅ **Passed Tests** | 1725 | 94.7% |
| ❌ **Failed Tests** | 59 | 3.2% |
| ⏭️ **Skipped Tests** | 37 | 2.0% |

---

## Critical Issues by Category

### 1. Configuration & Dependency Issues
These tests failed likely due to missing configuration, dependencies, or environment setups (e.g., missing API keys, uninitialized services).

*   **`tests/integration/strategies/test_multi_query_integration.py`**
    *   `test_with_real_llm`: `ValueError: MultiQueryRAGStrategy requires services: EMBEDDING, DATABASE`. The strategy was initialized without required dependencies.

*   **`tests/integration/services/test_onnx_embeddings_integration.py`**
    *   `test_performance_target`: `AssertionError: Average time 2913.74ms exceeds 100ms target`. Performance configuration or environment slowness.

### 2. Implementation Errors (Code Bugs)
These tests failed due to unhandled exceptions, type errors, or attribute errors in the code.

*   **`tests/integration/test_pipeline_integration.py`**
    *   Multiple tests (`test_pipeline_continues_after_non_critical_failure`, `test_async_fallback_execution`, etc.) failed with `TypeError: IRAGStrategy.__init__() missing 2 required positional arguments: 'config' and 'dependencies'`. The `IRAGStrategy` or its subclass is being instantiated with an incorrect signature.

*   **`tests/integration/test_factory_integration.py`**
    *   `test_create_multiple_instances_of_same_strategy` and others: `AttributeError: 'dict' object has no attribute 'chunk_size'`. The factory is likely treating a dictionary configuration as an object.
    *   `test_register_create_use_strategy`: `AttributeError: 'DummyStrategy' object has no attribute 'initialized'`. Typos or missing methods in mock objects.

*   **`tests/integration/test_package_integration.py`**
    *   `test_full_workflow_with_installed_package`: `TypeError: Can't instantiate abstract class TestStrategy...`. Abstract methods are not fully implemented in the test class.

*   **`tests/integration/services/test_service_integration.py`**
    *   `test_embedding_database_consistency`: `TypeError: 'coroutine' object does not support the asynchronous context manager protocol`. Incorrect usage of `async with`.

*   **`tests/integration/strategies/test_knowledge_graph_integration.py`**
    *   `test_hybrid_retrieval`, `test_relationship_queries`: `AttributeError: 'NoneType' object has no attribute 'search'`. A service (likely the graph store) returned `None` instead of a valid object.

*   **`tests/integration/test_config_integration.py`**
    *   `test_config_with_factory`: `AttributeError`: Internal implementation issue with registry access.

### 3. Requirement & Logic Failures
These tests ran but failed assertions, indicating that the functionality does not meet the requirements (e.g., getting 0 results when >0 were expected).

*   **`tests/integration/models/test_fine_tuned_embeddings_integration.py`**
    *   `test_ab_testing_workflow`, `test_model_comparison_workflow`: `assert 0 > 0`. No results returned, likely due to embedding model or retrieval failure.

*   **`tests/integration/strategies/test_late_chunking_integration.py`**
    *   Multiple tests (`test_late_chunking_workflow`, `test_multiple_documents`): `assert 0 > 0`. Late chunking strategy failing to produce chunks.

*   **`tests/integration/strategies/test_multi_query_integration.py`**
    *   Multiple tests (`test_multi_query_complete_workflow`, `test_variant_diversity`): `assert 0 > 0`. Query expansion failing to produce variations or results.

*   **`tests/integration/strategies/test_contextual_integration.py`**
    *   `test_error_recovery`: `assert 7 == 10`. Retry logic or error handling not matching expected count.

*   **`tests/integration/test_package_integration.py`**
    *   `test_package_installable`: `assert 1 == 0`. Package installation check failed.

---

## Action Plan
1.  **Fix Strategy Initialization:** Update `IRAGStrategy` calls in pipeline tests to include `config` and `dependencies`.
2.  **Fix Factory Config Access:** Ensure the factory handles dictionary configurations correctly (using `['key']` or converting to object).
3.  **Resolve Dependencies:** Ensure `MultiQueryRAGStrategy` tests provide necessary service mocks.
4.  **Investigate Zero Results:** Debug the embedding and retrieval flow involved in `late_chunking` and `fine_tuned_embeddings` to see why no results are returned (check mock outputs).
5.  **Address Performance:** Investigate `ONNX` model performance or adjust the test threshold if the environment is slow.

---

## Failing Test Files List
*   `tests/integration/models/test_fine_tuned_embeddings_integration.py`
*   `tests/integration/services/test_embedding_integration.py`
*   `tests/integration/services/test_onnx_embeddings_integration.py`
*   `tests/integration/services/test_service_integration.py`
*   `tests/integration/strategies/test_contextual_integration.py`
*   `tests/integration/strategies/test_knowledge_graph_integration.py`
*   `tests/integration/strategies/test_late_chunking_integration.py`
*   `tests/integration/strategies/test_multi_query_integration.py`
*   `tests/integration/test_config_integration.py`
*   `tests/integration/test_factory_integration.py`
*   `tests/integration/test_package_integration.py`
*   `tests/integration/test_pipeline_integration.py`

## Skipped Test Files
*   `tests/benchmarks/test_model_comparison_performance.py`
*   `tests/integration/documentation/test_documentation_integration.py`

## Single Failing Tests List
1. `tests/integration/models/test_fine_tuned_embeddings_integration.py::test_ab_testing_workflow`
2. `tests/integration/models/test_fine_tuned_embeddings_integration.py::test_model_comparison_workflow`
3. `tests/integration/services/test_onnx_embeddings_integration.py::TestONNXEmbeddingsIntegration::test_performance_target`
4. `tests/integration/services/test_service_integration.py::test_embedding_database_consistency`
5. `tests/integration/strategies/test_contextual_integration.py::test_error_recovery`
6. `tests/integration/strategies/test_knowledge_graph_integration.py::test_hybrid_retrieval`
7. `tests/integration/strategies/test_knowledge_graph_integration.py::test_relationship_queries`
8. `tests/integration/strategies/test_late_chunking_integration.py::test_late_chunking_workflow`
9. `tests/integration/strategies/test_late_chunking_integration.py::test_multiple_documents`
10. `tests/integration/strategies/test_late_chunking_integration.py::test_short_document`
11. `tests/integration/strategies/test_multi_query_integration.py::TestMultiQueryIntegration::test_multi_query_complete_workflow`
12. `tests/integration/strategies/test_multi_query_integration.py::TestMultiQueryIntegration::test_multi_query_async_workflow`
13. `tests/integration/strategies/test_multi_query_integration.py::TestMultiQueryIntegration::test_variant_diversity`
14. `tests/integration/strategies/test_multi_query_integration.py::TestMultiQueryIntegration::test_performance_requirements`
15. `tests/integration/strategies/test_multi_query_integration.py::TestMultiQueryIntegration::test_fallback_on_failure`
16. `tests/integration/strategies/test_multi_query_integration.py::TestMultiQueryIntegration::test_ranking_strategy_comparison`
17. `tests/integration/strategies/test_multi_query_integration.py::TestMultiQueryWithLMStudio::test_with_real_llm`
18. `tests/integration/test_config_integration.py::test_config_with_factory`
19. `tests/integration/test_factory_integration.py::test_register_create_use_strategy`
20. `tests/integration/test_factory_integration.py::test_create_multiple_instances_of_same_strategy`
21. `tests/integration/test_factory_integration.py::test_config_file_with_yaml`
22. `tests/integration/test_factory_integration.py::test_config_file_with_json`
23. `tests/integration/test_factory_integration.py::test_factory_error_recovery`
24. `tests/integration/test_factory_integration.py::test_factory_state_after_failed_creation`
25. `tests/integration/test_package_integration.py::TestPackageInstallation::test_package_installable`
26. `tests/integration/test_package_integration.py::TestFullWorkflow::test_full_workflow_with_installed_package`
27. `tests/integration/test_pipeline_integration.py::test_pipeline_continues_after_non_critical_failure`
28. `tests/integration/test_pipeline_integration.py::test_async_fallback_execution`
29. `tests/integration/test_pipeline_integration.py::test_parallel_execution_with_failures`
