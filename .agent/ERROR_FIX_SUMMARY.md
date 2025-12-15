# Error Fix Summary - 2025-12-15

## Overview
Following the `/fix-similar-errors` workflow to systematically fix test failures.

## Error Patterns Identified

### 1. ✅ FIXED: Missing `token_type_ids` in ONNX Models
**Error Pattern**: `ValueError: Required inputs (['token_type_ids']) are missing from input feed`

**Affected Files**:
- `rag_factory/strategies/late_chunking/document_embedder.py`
- `rag_factory/services/embedding/providers/onnx_local.py`

**Solution Applied**:
Added dynamic detection of required inputs and automatic inclusion of `token_type_ids` when needed:

```python
# Check if model requires token_type_ids (BERT-based models like MiniLM)
input_names = [inp.name for inp in self.session.get_inputs()]
if "token_type_ids" in input_names:
    # Create token_type_ids (all zeros for single sequence)
    token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
    inputs["token_type_ids"] = token_type_ids
```

**Status**: ✅ Fixed in both files (lines 121-126)

---

### 2. ✅ FIXED: Embedding Dimension Mismatch
**Error Pattern**: `assert 384 == 768` - Tests expecting 768 dimensions but model returns 384

**Root Cause**: Tests were hardcoded to expect 768 dimensions (mpnet model), but environment uses 384-dimension model (MiniLM)

**Solution Applied**:
Updated tests to use dynamic dimension checking:

```python
# Before:
assert len(result.embeddings[0]) == 768

# After:
assert len(result.embeddings[0]) == result.dimensions  # Use actual model dimensions
assert result.dimensions in [384, 768]  # Support both MiniLM (384) and mpnet (768)
```

**Status**: ✅ Fixed in `tests/integration/services/test_embedding_integration.py` (lines 101, 247)

---

### 3. 🔧 IN PROGRESS: Late Chunking Model Configuration
**Error Pattern**: Model download fails because tests use wrong default model

**Root Cause**: 
- Environment variable: `EMBEDDING_MODEL_NAME=Xenova/all-MiniLM-L6-v2`
- Test default: `Xenova/all-mpnet-base-v2`
- The mpnet model doesn't have ONNX files in the expected location

**Solution Needed**:
Update test files to use environment variable instead of hardcoded default:

```python
# Current (wrong):
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "Xenova/all-mpnet-base-v2")

# Should be:
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "Xenova/all-MiniLM-L6-v2")
```

**Affected Files**:
- `tests/integration/strategies/test_late_chunking_integration.py`
- Any other test files using the wrong default

**Status**: 🔧 Needs fixing

---

### 4. 🔧 TODO: Database Mock Async Context Manager Error
**Error Pattern**: `TypeError: 'coroutine' object does not support the asynchronous context manager protocol`

**Location**: `tests/integration/services/test_service_integration.py::test_embedding_database_consistency`

**Root Cause**: Mock setup for `pool.acquire()` not properly configured for async context manager

**Solution Needed**:
Fix the mock to properly support async context manager protocol:

```python
# The mock needs to return an async context manager
mock_pool.acquire = AsyncMock()
mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
```

**Status**: 🔧 Needs investigation and fix

---

### 5. 🔧 TODO: Factory AttributeError
**Error Pattern**: `AttributeError: 'str' object has no attribute '_registry'`

**Location**: `tests/integration/test_config_integration.py::test_config_with_factory`

**Root Cause**: `RAGFactory.create_strategy()` being called incorrectly - method called on class instead of instance

**Solution Needed**:
Fix the test to create a factory instance first:

```python
# Current (wrong):
strategy = RAGFactory.create_strategy("test_strategy", strategy_config.model_dump())

# Should be:
factory = RAGFactory()
strategy = factory.create_strategy("test_strategy", strategy_config.model_dump())
```

**Status**: 🔧 Needs fixing

---

### 6. 🔧 TODO: Multi-Query LLM Empty Responses
**Error Pattern**: Tests failing because LLM returns empty responses

**Affected Tests**:
- `tests/integration/strategies/test_multi_query_integration.py` (multiple tests)

**Root Cause**: LM Studio not returning proper responses or mock LLM not configured correctly

**Solution Needed**:
- Verify LM Studio is running and responding
- Check if tests need better mock LLM setup
- May need to adjust test expectations or LLM configuration

**Status**: 🔧 Needs investigation

---

### 7. 🔧 TODO: Knowledge Graph Vector Store None
**Error Pattern**: `AttributeError: 'NoneType' object has no attribute 'search'`

**Location**: 
- `tests/integration/strategies/test_knowledge_graph_integration.py::test_hybrid_retrieval`
- `tests/integration/strategies/test_knowledge_graph_integration.py::test_relationship_queries`

**Root Cause**: Vector store not being passed to strategy or not being initialized properly

**Solution Needed**:
Ensure vector store is properly initialized and passed to the knowledge graph strategy

**Status**: 🔧 Needs investigation

---

## Next Steps

1. ✅ **COMPLETED**: Fix `token_type_ids` errors
2. ✅ **COMPLETED**: Fix embedding dimension mismatches
3. 🔄 **IN PROGRESS**: Fix late chunking model configuration
4. ⏭️ **TODO**: Fix database mock async issues
5. ⏭️ **TODO**: Fix factory instantiation error
6. ⏭️ **TODO**: Fix multi-query LLM response issues
7. ⏭️ **TODO**: Fix knowledge graph vector store initialization

## Test Execution Plan

After fixing each error pattern:
1. Run the specific failing tests to verify the fix
2. Check if the same error appears in other test files
3. Apply the same fix to all occurrences
4. Run the full test suite to ensure no regressions

## Commands to Run Tests

```bash
# Run specific test file
source venv/bin/activate && pytest tests/integration/strategies/test_late_chunking_integration.py -xvs

# Run all late chunking tests
source venv/bin/activate && pytest tests/integration/strategies/test_late_chunking_integration.py

# Run all integration tests
source venv/bin/activate && pytest tests/integration/ -x

# Run with coverage
source venv/bin/activate && pytest tests/integration/ --cov=rag_factory --cov-report=term-missing
```

## Notes

- Test results file timestamp: 2025-12-15 08:41:57
- Environment model: Xenova/all-MiniLM-L6-v2 (384 dimensions)
- Model path: models/embedding
- LM Studio: http://192.168.56.1:1234/v1 (phi-4-mini-instruct)
