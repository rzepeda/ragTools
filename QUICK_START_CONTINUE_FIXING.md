# Quick Start: Continue Fixing Tests

This guide helps you continue the systematic test-fixing work.

---

## 🚀 Quick Commands

### Run All Strategy Tests
```bash
./run_tests_with_env.sh tests/integration/strategies/ -v
```

### Run Specific Test File
```bash
./run_tests_with_env.sh tests/integration/strategies/test_[name].py -v
```

### Run Single Test
```bash
./run_tests_with_env.sh tests/integration/strategies/test_[name].py::test_function_name -v
```

---

## 📋 Current Status (As of 2025-12-13)

### ✅ Fully Passing (100%)
- `test_base_integration.py` - 5/5 tests
- `test_keyword_indexing.py` - 6/6 tests

### 🟡 Partially Passing
- `test_knowledge_graph_integration.py` - 2/4 tests (50%)
- `test_multi_query_integration.py` - 1/8 tests (12.5%)

### 🔴 Not Passing
- `test_contextual_integration.py` - 0/6 tests (business logic issues)

---

## 🎯 Next 3 Fixes (Prioritized)

### Fix #14: HybridRetriever vector_store (EASY)
**Impact:** 2 tests  
**File:** `rag_factory/strategies/knowledge_graph/hybrid_retriever.py`  
**Pattern:** Same as Fix #8 and #13

**What to do:**
1. Find `self.vector_store.search()` in hybrid_retriever.py
2. Replace with `self.database_service.search_similar()`
3. Update `__init__` to accept database_service

**Expected Result:** test_knowledge_graph_integration.py goes to 4/4 (100%)

---

### Fix #15: MultiQuery Async Mocks (MEDIUM)
**Impact:** 7 tests  
**File:** `tests/integration/strategies/test_multi_query_integration.py`  
**Pattern:** Mock setup issue

**What to do:**
```python
from unittest.mock import AsyncMock

# In test fixtures, change:
mock_db_service = Mock()

# To:
mock_db_service = Mock()
mock_db_service.asearch_similar = AsyncMock(return_value=[])
```

**Expected Result:** test_multi_query_integration.py goes to 8/8 (100%)

---

### Fix #16: Contextual Business Logic (HARD)
**Impact:** 6 tests  
**File:** `tests/integration/strategies/test_contextual_integration.py`  
**Pattern:** Business logic - tests getting 0 results

**What to do:**
1. Check if LM Studio is returning empty responses
2. Verify mock services are properly configured
3. Ensure database operations are working
4. Debug why `total_chunks` is 0

**Expected Result:** test_contextual_integration.py goes to 6/6 (100%)

---

## 📖 Reference Documents

### Pattern Guides
- **STRATEGY_TEST_FIX_PATTERNS.md** - Quick reference for common patterns
- **COMPLETE_SESSION_SUMMARY.md** - Full session overview

### Detailed Fix Documentation
- **STRATEGY_TEST_FIXES_SUMMARY.md** - All 13 fixes detailed
- **FIX_13_KNOWLEDGE_GRAPH_VECTOR_STORE.md** - Latest fix example

---

## 🔍 How to Find Next Error

### Step 1: Run Tests
```bash
./run_tests_with_env.sh tests/integration/strategies/ -v --tb=short
```

### Step 2: Check First Error
```bash
head -100 test_results.txt
```

### Step 3: Search for Pattern
```bash
grep -c "error_message" tests/test_results_by_file.txt
```

### Step 4: Fix Pattern
1. Fix the first occurrence
2. Check if it repeats
3. Apply to all instances
4. Test and verify
5. Document the fix

---

## 🛠️ Common Fix Patterns

### Pattern A: Add requires_services()
```python
def requires_services(self):
    from rag_factory.services.dependencies import ServiceDependency
    return set()  # or {ServiceDependency.LLM, ...}
```

### Pattern B: Use StrategyDependencies
```python
dependencies = StrategyDependencies(
    llm_service=mock_llm,
    database_service=mock_db
)
strategy = Strategy(config, dependencies)
```

### Pattern C: Fix vector_store → database_service
```python
# Change from:
self.vector_store.search(...)

# To:
self.deps.database_service.search_similar(...)
```

### Pattern D: Use AsyncMock
```python
from unittest.mock import AsyncMock

mock_service.async_method = AsyncMock(return_value=expected_value)
```

---

## ✅ Verification Checklist

After each fix:
- [ ] Test runs without errors
- [ ] Check if pattern repeats elsewhere
- [ ] Apply fix to all instances
- [ ] Run tests again to verify
- [ ] Document the fix
- [ ] Update summary documents

---

## 📊 Progress Tracking

Update these files after each fix:
1. **COMPLETE_SESSION_SUMMARY.md** - Add new fix to list
2. **TEST_RESULTS_AFTER_FIXES.md** - Update test counts
3. Create new **FIX_XX_[NAME].md** for complex fixes

---

## 🎯 Goal

**Target:** 100% pass rate for all strategy integration tests

**Current:** 14/34 tests passing (41%)  
**Remaining:** 20 tests to fix  
**Estimated:** 3-5 more fix patterns needed

---

## 💡 Tips

1. **One pattern at a time** - Don't try to fix everything at once
2. **Test frequently** - Verify each fix before moving on
3. **Document everything** - Future you will thank you
4. **Look for patterns** - Same error often appears multiple times
5. **Use the guides** - Reference STRATEGY_TEST_FIX_PATTERNS.md

---

## 🆘 If Stuck

1. Check **STRATEGY_TEST_FIX_PATTERNS.md** for similar patterns
2. Look at previous fixes in **STRATEGY_TEST_FIXES_SUMMARY.md**
3. Search for the error message in test_results_by_file.txt
4. Check if it's a known pattern (vector_store, requires_services, etc.)

---

## 🚀 Let's Continue!

**Next Command to Run:**
```bash
./run_tests_with_env.sh tests/integration/strategies/test_knowledge_graph_integration.py -v
```

**Expected:** 2/4 passing (need to fix HybridRetriever)

**After that:**
```bash
./run_tests_with_env.sh tests/integration/strategies/test_multi_query_integration.py -v
```

**Expected:** 1/8 passing (need to fix async mocks)

---

**Good luck! The foundation is solid, and the patterns are clear. You've got this! 🎉**
