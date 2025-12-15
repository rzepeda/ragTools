# Test Report
**Date:** Mon Dec 15 2025
**Total Duration:** 69m 7s

## Executive Summary

The test suite execution resulted in a high pass rate, with most core functionalities verifying correctly. However, there is a specific recurring failure related to ONNX embedding providers and document embedders, pointing to a shared implementation or mocking issue.

| Metric | Count | Percentage |
| :--- | :--- | :--- |
| **Total Tests** | 1806 | 100% |
| **Passed** | 1754 | 97.1% |
| **Failed** | 14 | 0.8% |
| **Skipped** | 38 | 2.1% |

## Critical Issues by Category

### Implementation Issues
The primary source of failure is a `TypeError: 'Mock' object is not iterable` occurring in tests that interact with the ONNX runtime session.

*   **Issue:** The `self.session.get_inputs()` method in mocked ONNX sessions is returning a `Mock` object instead of an iterable.
*   **Affected Components:**
    *   `ONNXLocalProvider` (`tests/unit/services/embedding/test_onnx_local_provider.py`)
    *   `DocumentEmbedder` (`tests/unit/strategies/late_chunking/test_document_embedder.py`)
*   **Root Cause:** The mock setup for `ort.InferenceSession` likely needs to be updated to return a list of mock inputs when `get_inputs()` is called to correctly simulate the ONNX runtime behavior.

### Configuration / Environment Issues
Two test files were entirely skipped. While not failures, these represent unchecked areas potentially due to environment configuration.

*   `tests/benchmarks/test_model_comparison_performance.py`: Likely skipped due to missing heavy models or performance constraints in the test environment.
*   `tests/integration/documentation/test_documentation_integration.py`: Likely skipped due to missing documentation generation tools or dependencies.

### Requirement Issues
No specific requirement-based failures were identified in this run.

## Detailed Failure Analysis

### 1. ONNX Local Provider & Document Embedder
**Error:** `TypeError: 'Mock' object is not iterable`
**Context:**
```python
# rag_factory/services/embedding/providers/onnx_local.py or document_embedder.py
input_names = [inp.name for inp in self.session.get_inputs()]
```
**Diagnosis:** The code expects `self.session.get_inputs()` to return a list of input objects (implicitly, objects with a `.name` attribute). The test mock is returning a specific `Mock` object which is not iterable, causing the list comprehension to fail.

## List of Failing Test Files
1.  `tests/unit/services/embedding/test_onnx_local_provider.py`
2.  `tests/unit/strategies/late_chunking/test_document_embedder.py`

## List of Single Failing Tests

**tests/unit/services/embedding/test_onnx_local_provider.py**
*   `test_get_embeddings`

**tests/unit/strategies/late_chunking/test_document_embedder.py**
*   `test_document_embedding_basic`
*   `test_token_embeddings_extracted`
*   `test_long_document_truncation`
*   `test_batch_processing`
*   `test_embedding_dimensions_consistent`
*   `test_model_name_stored`
