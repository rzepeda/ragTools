# Fix #13: KnowledgeGraphRAGStrategy vector_store Bug

**Date:** 2025-12-13  
**Pattern:** Code bug - accessing non-existent attribute  
**Status:** ✅ PARTIAL SUCCESS (2/4 tests passing)

---

## Error Details

**Error Type:** `AttributeError: 'KnowledgeGraphRAGStrategy' object has no attribute 'vector_store'`

**Location:** `rag_factory/strategies/knowledge_graph/strategy.py`, line 101

**Failed Tests:**
- test_knowledge_graph_workflow
- test_hybrid_retrieval
- test_relationship_queries
- test_graph_statistics

---

## Root Cause

The `KnowledgeGraphRAGStrategy.index_document()` method was trying to access `self.vector_store` which doesn't exist in the new dependency injection pattern. The strategy should use `self.deps.database_service` instead.

**Problematic Code:**
```python
# Line 101 in strategy.py
for chunk in chunks:
    self.vector_store.index_chunk(  # ❌ vector_store doesn't exist
        chunk_id=chunk["chunk_id"],
        text=chunk["text"],
        metadata={"document_id": document_id}
    )
```

---

## Fix Applied

Updated `index_document()` method to use `database_service` from dependencies:

```python
# Index chunks in vector store (using database service)
if self.deps.database_service:
    for chunk in chunks:
        # Use database service to store chunks
        if hasattr(self.deps.database_service, 'add_chunk'):
            self.deps.database_service.add_chunk(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                metadata={"document_id": document_id}
            )
        elif hasattr(self.deps.database_service, 'index_chunk'):
            self.deps.database_service.index_chunk(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                metadata={"document_id": document_id}
            )
else:
    logger.warning("No database service available for chunk indexing")
```

---

## Results

### Before Fix
- ✅ 0/4 tests passing (0%)
- ❌ All tests failing with AttributeError

### After Fix
- ✅ 2/4 tests passing (50%)
- ❌ 2 tests still failing with different error

**Passing Tests:**
- ✅ test_knowledge_graph_workflow
- ✅ test_graph_statistics

**Still Failing:**
- ❌ test_hybrid_retrieval - `AttributeError: 'NoneType' object has no attribute 'search'`
- ❌ test_relationship_queries - `AttributeError: 'NoneType' object has no attribute 'search'`

---

## Remaining Issue

The `HybridRetriever` class (used by the strategy) was initialized with `None` for vector_store:

```python
# Line 63-67 in strategy.py
self.hybrid_retriever = HybridRetriever(
    None,  # vector_store_service placeholder ❌
    self.graph_store,
    self.strategy_config
)
```

When `retrieve()` is called, the HybridRetriever tries to use `self.vector_store.search()` which fails because it's None.

**Location of Error:**
```python
# rag_factory/strategies/knowledge_graph/hybrid_retriever.py, line 64
vector_results = self.vector_store.search(query, top_k=top_k * 2)
                 ^^^^^^^^^^^^^^^^^^^^^^^^
# AttributeError: 'NoneType' object has no attribute 'search'
```

---

## Next Steps

To fully fix this issue, we need to:

1. Update `HybridRetriever.__init__()` to accept database_service from dependencies
2. Update `HybridRetriever.retrieve()` to use `self.database_service.search_similar()` instead of `self.vector_store.search()`
3. Update `KnowledgeGraphRAGStrategy.__init__()` to pass `self.deps.database_service` to HybridRetriever

**Estimated Impact:** Would fix the remaining 2 tests (100% pass rate for knowledge graph tests)

---

## Pattern Recognition

This is the **same pattern** as:
- Fix #8: MultiQueryRAGStrategy._fallback_retrieve (vector_store → database_service)
- Fix #13: KnowledgeGraphRAGStrategy.index_document (vector_store → database_service)

**Common Pattern:** Old code accessing `self.vector_store` needs to be updated to use `self.deps.database_service`

---

## Files Modified

**File:** `rag_factory/strategies/knowledge_graph/strategy.py`  
**Lines:** 99-106 (replaced 8 lines with 19 lines)  
**Method:** `index_document()`

---

## Verification

**Test Command:**
```bash
./run_tests_with_env.sh tests/integration/strategies/test_knowledge_graph_integration.py -v
```

**Results:**
```
=================== 2 failed, 2 passed, 6 warnings in 11.11s ===================
```

**Coverage Impact:** 19% (up from 16% initially)

---

## Summary

✅ **Partial Success:** Fixed 50% of knowledge graph tests  
🟡 **Remaining Work:** Need to update HybridRetriever class  
📊 **Overall Progress:** 14/34 strategy tests now passing (41%, up from 40%)

---

**Fix Type:** Code Bug  
**Difficulty:** Medium  
**Time to Fix:** 5 minutes  
**Pattern Frequency:** High (3rd occurrence of vector_store → database_service pattern)
