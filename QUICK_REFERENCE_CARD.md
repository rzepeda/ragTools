# Quick Reference Card - Test Fixing Patterns

**Last Updated:** 2025-12-13 23:36  
**Status:** 14 Fixes Complete, 48+ Tests Fixed

---

## 🚀 Quick Start

```bash
# Run all strategy tests
./run_tests_with_env.sh tests/integration/strategies/ -v

# Run specific file
./run_tests_with_env.sh tests/integration/strategies/test_[name].py -v

# Check results
cat test_results.txt | grep -E "(PASSED|FAILED)"
```

---

## ✅ Current Status

**Fully Passing (100%):**
- test_base_integration.py (5/5)
- test_keyword_indexing.py (6/6)
- test_config_integration.py (1/1)
- test_base.py unit tests (2/2)

**Partially Passing:**
- test_knowledge_graph_integration.py (2/4 - 50%)
- test_multi_query_integration.py (1+/8 - 12.5%+)

**Need Work:**
- test_contextual_integration.py (0/6)

---

## 🔧 Top 4 Fix Patterns

### 1. Add `requires_services()` ⭐ MOST COMMON
```python
def requires_services(self):
    from rag_factory.services.dependencies import ServiceDependency
    return set()  # or {ServiceDependency.LLM, ServiceDependency.DATABASE}
```
**When:** `TypeError: Can't instantiate abstract class... 'requires_services'`

### 2. Use StrategyDependencies ⭐ VERY COMMON
```python
from rag_factory.services.dependencies import StrategyDependencies

dependencies = StrategyDependencies(
    llm_service=mock_llm,
    embedding_service=mock_embedding,
    database_service=mock_db
)
strategy = Strategy(config, dependencies)
```
**When:** `TypeError: __init__() got unexpected keyword argument 'vector_store_service'`

### 3. Fix vector_store → database_service ⭐ COMMON
```python
# Change from:
self.vector_store.search(...)

# To:
if self.deps.database_service:
    self.deps.database_service.search_similar(...)
```
**When:** `AttributeError: ... has no attribute 'vector_store'`

### 4. Use AsyncMock for async methods ⭐ CRITICAL
```python
from unittest.mock import AsyncMock

mock_service.async_method = AsyncMock(return_value=expected_value)
```
**When:** `TypeError: object Mock can't be used in 'await' expression`

---

## 📋 Systematic Approach (5 Steps)

1. **Identify** - Find first error in test output
2. **Fix** - Implement solution once
3. **Check** - Search if pattern repeats
4. **Apply** - Fix all instances
5. **Verify** - Run tests and document

---

## 🎯 Next 3 Fixes (Prioritized)

### Fix #15: HybridRetriever (EASY - 10 min)
**File:** `rag_factory/strategies/knowledge_graph/hybrid_retriever.py`  
**Change:** `self.vector_store.search()` → `self.database_service.search_similar()`  
**Impact:** +2 tests (knowledge graph to 100%)

### Fix #16: MultiQuery Mock Data (MEDIUM - 20 min)
**File:** `tests/integration/strategies/test_multi_query_integration.py`  
**Change:** Add realistic return values to mocks  
**Impact:** +7 tests (multi-query to 100%)

### Fix #17: Contextual Business Logic (HARD - 30+ min)
**File:** `tests/integration/strategies/test_contextual_integration.py`  
**Change:** Debug why getting 0 results  
**Impact:** +6 tests (contextual to 100%)

---

## 📖 Documentation Files

**Read These First:**
- `QUICK_START_CONTINUE_FIXING.md` - How to continue
- `STRATEGY_TEST_FIX_PATTERNS.md` - Pattern reference
- `FINAL_SESSION_SUMMARY_14_FIXES.md` - Complete overview

**Detailed Info:**
- `COMPLETE_SESSION_SUMMARY.md` - Full methodology
- `FIX_13_KNOWLEDGE_GRAPH_VECTOR_STORE.md` - Example fix
- `PROGRESS_UPDATE.md` - Latest status

---

## 🆘 Troubleshooting

**Error:** Can't instantiate abstract class  
**Fix:** Add `requires_services()` method

**Error:** Unexpected keyword argument  
**Fix:** Use StrategyDependencies pattern

**Error:** No attribute 'vector_store'  
**Fix:** Use `self.deps.database_service`

**Error:** Mock can't be awaited  
**Fix:** Use AsyncMock for async methods

---

## 📊 Success Metrics

- ✅ 14 patterns fixed
- ✅ 48+ tests repaired
- ✅ 44%+ pass rate (from ~0%)
- ✅ 0 API errors (from ~30)
- ✅ 0 async errors (from ~7)

---

## 🎓 Key Learnings

1. **One pattern at a time** - Don't try to fix everything
2. **Test frequently** - Verify each fix
3. **Document everything** - Future you will thank you
4. **Look for patterns** - Same error often repeats
5. **AsyncMock matters** - Critical for async tests

---

**Good luck! The foundation is solid! 🚀**
