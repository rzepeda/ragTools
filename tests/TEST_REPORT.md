# Test Suite Report

**Generated:** 2025-12-11  
**Execution Time:** 70 minutes 10 seconds  
**Test Runner:** File-based execution with 5-minute timeout per file

---

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Test Files** | 156 | 100% |
| **Total Tests** | 1,940 | 100% |
| ✅ **Passed Tests** | 1,605 | 82.7% |
| ❌ **Failed Tests** | 302 | 15.6% |
| ⏭️ **Skipped Tests** | 33 | 1.7% |
| | | |
| ✅ **Passed Files** | 120 | 76.9% |
| ❌ **Failed Files** | 35 | 22.4% |
| ⏭️ **Skipped Files** | 1 | 0.6% |
| ⏱️ **Timeout Files** | 0 | 0.0% |

### Key Findings

- **Overall Health:** 82.7% pass rate - good foundation with specific fixable issues
- **No Timeouts:** All tests completed within timeout limits
- **Root Causes Identified:** Most failures stem from 3 main issues
- **Quick Wins Available:** ~50+ failures can be fixed with 2 dependency installations

---

## 🔴 Critical Root Causes (Affects 50+ Tests)

### 1. Missing `sentence-transformers` Dependency ⚠️ **HIGH PRIORITY**

**Impact:** 32+ test failures across multiple files  
**Error:** `ModuleNotFoundError: No module named 'sentence_transformers'`

**Affected Test Files:**
- `tests/benchmarks/test_late_chunking_performance.py` (12 failures)
- `tests/benchmarks/test_model_comparison_performance.py` (12 failures)
- `tests/integration/models/test_fine_tuned_embeddings_integration.py` (2 failures)
- `tests/integration/services/test_embedding_integration.py` (multiple failures)
- `tests/integration/strategies/test_late_chunking_integration.py` (multiple failures)

**Fix:**
```bash
pip install sentence-transformers
```

**Why This Matters:** This library is required for:
- Late chunking strategy (semantic boundary detection)
- Model comparison benchmarks
- Fine-tuned embedding models
- Sentence transformer-based embeddings

---

### 2. Invalid Provider Name: `onnx-local` ⚠️ **HIGH PRIORITY**

**Impact:** 24+ test failures  
**Error:** `ValueError: Invalid provider: onnx-local. Must be one of ['openai', 'cohere', 'local']`

**Affected Test Files:**
- `tests/integration/services/test_embedding_integration.py`
- `tests/integration/services/test_onnx_embeddings_integration.py`
- `tests/unit/services/embeddings/test_onnx_local.py`
- `tests/unit/services/embedding/test_onnx_local_provider.py`

**Root Cause:** Provider validation code doesn't recognize `'onnx-local'` as valid

**Fix Options:**
1. **Update provider validation** to include `'onnx-local'` in allowed providers
2. **Rename provider** from `'onnx-local'` to `'local'` in tests and code
3. **Add provider alias** mapping `'onnx-local'` → `'local'`

**Recommended:** Option 1 - Add `'onnx-local'` to the allowed providers list in the embedding service validation.

---

### 3. Test Fixture Issues 🔧 **MEDIUM PRIORITY**

**Impact:** 4+ test failures  
**Error:** `fixture 'document' not found`

**Affected Tests:**
- `test_vector_search_with_metadata_filter`
- `test_vector_search_identical_embedding`

**Root Cause:** Tests reference `document` fixture but only `document_with_chunks` is available

**Fix:** Update test signatures to use correct fixture name:
```python
# Change from:
def test_vector_search_with_metadata_filter(self, db_session, document):

# To:
def test_vector_search_with_metadata_filter(self, db_session, document_with_chunks):
```

---

## Detailed Failure Analysis by Category

### Integration Tests - Strategies (12 files, ~120 failures)

| File | Passed | Failed | Primary Issue |
|------|--------|--------|---------------|
| `test_query_expansion_integration.py` | 2 | 18 | Missing LLM service/API keys |
| `test_multi_query_integration.py` | 0 | 16 | Missing LLM service/API keys |
| `test_contextual_integration.py` | 0 | 12 | Missing LLM service/API keys |
| `test_late_chunking_integration.py` | 3 | 12 | sentence-transformers missing |
| `test_base_integration.py` | 0 | 10 | Service integration issues |
| `test_hierarchical_integration.py` | 0 | 8 | Missing dependencies |
| `test_knowledge_graph_integration.py` | 0 | 8 | Missing LLM service |
| `test_keyword_indexing.py` | 0 | 0 | Collection error |

**Common Pattern:** Most failures require:
- LLM API keys (OpenAI, Anthropic)
- sentence-transformers library
- Proper service mocking for CI

---

### Integration Tests - Services (7 files, ~40 failures)

| File | Passed | Failed | Primary Issue |
|------|--------|--------|---------------|
| `test_embedding_integration.py` | 0 | 14 | onnx-local provider + sentence-transformers |
| `test_repository_integration.py` | 10 | 14 | Fixture issues + assertion errors |
| `test_onnx_embeddings_integration.py` | 9 | 2 | onnx-local provider validation |
| `test_service_implementations.py` | 17 | 2 | Minor integration issues |
| `test_llm_integration.py` | 0 | 2 | Missing API keys (7 skipped) |
| `test_service_integration.py` | 0 | 2 | Service configuration |

**Repository Integration Specific Errors:**
- `AssertionError: assert 6 == 5` - Pagination test counting issue
- `AssertionError: assert 2 == 1` - Status filtering test
- `ValueError: The truth value of an array with more than one element is ambiguous` - NumPy array comparison issue

---

### Benchmarks (2 files, 24 failures)

| File | Passed | Failed | Primary Issue |
|------|--------|--------|---------------|
| `test_late_chunking_performance.py` | 2 | 12 | sentence-transformers missing |
| `test_model_comparison_performance.py` | 0 | 12 | sentence-transformers missing |

**Fix:** Install sentence-transformers - will resolve all 24 failures

---

### Integration Tests - Factory & Config (3 files, ~38 failures)

| File | Passed | Failed | Primary Issue |
|------|--------|--------|---------------|
| `test_factory_integration.py` | 1 | 20 | Service instantiation failures |
| `test_pipeline_integration.py` | 0 | 14 | Pipeline configuration |
| `test_config_integration.py` | 6 | 4 | Config validation |

**Likely Causes:**
- Missing service dependencies (sentence-transformers, API keys)
- Provider validation issues (onnx-local)
- Service registration problems

---

### Unit Tests (8 files, ~80 failures)

| File | Passed | Failed | Primary Issue |
|------|--------|--------|---------------|
| `test_pipeline.py` | 1 | 46 | Pipeline logic errors |
| `test_strategy.py` (agentic) | 3 | 16 | Strategy implementation |
| `test_base.py` (strategies) | 13 | 6 | Base strategy issues |
| `test_links.py` | 0 | 6 | Broken documentation links |
| `test_onnx_local.py` (embeddings) | 17 | 4 | onnx-local provider |
| `test_code_examples.py` | 2 | 4 | Code example validation |
| `test_onnx_local_provider.py` | 9 | 2 | Provider validation |
| `test_interfaces.py` | 20 | 2 | Interface contracts |

**Priority:** `test_pipeline.py` with 46 failures needs investigation

---

### Database Tests (2 files, 4 failures)

| File | Passed | Failed | Issue |
|------|--------|--------|-------|
| `test_connection.py` | 13 | 2 | Connection configuration |
| `test_models.py` | 20 | 2 | Model validation |

**Likely Cause:** TEST_DATABASE_URL configuration or schema migration issues

---

### Documentation Tests (3 files, 12 failures)

| File | Passed | Failed | Issue |
|------|--------|--------|-------|
| `test_links.py` | 0 | 6 | Broken links |
| `test_code_examples.py` | 2 | 4 | Invalid examples |
| `test_documentation_completeness.py` | 7 | 2 | Missing docs |

---

### CLI Tests (2 files, 4 failures)

| File | Passed | Failed | Issue |
|------|--------|--------|-------|
| `test_check_consistency_command.py` | 10 | 2 | Command execution |
| `test_repl_command.py` | 3 | 2 | REPL interaction |

---

## 🎯 Immediate Action Plan

### Phase 1: Quick Wins (Fixes ~50+ tests) ⚡

**1. Install sentence-transformers**
```bash
pip install sentence-transformers
```
**Impact:** Fixes 32+ test failures immediately

**2. Fix onnx-local provider validation**

Locate provider validation code and add `'onnx-local'` to allowed providers:

```python
# Find this code (likely in embedding service)
VALID_PROVIDERS = ['openai', 'cohere', 'local']

# Change to:
VALID_PROVIDERS = ['openai', 'cohere', 'local', 'onnx-local']
```
**Impact:** Fixes 24+ test failures

**3. Fix test fixtures**

Update these test methods in `test_repository_integration.py`:
```python
# Line ~391 and ~421
def test_vector_search_with_metadata_filter(self, db_session, document_with_chunks):
def test_vector_search_identical_embedding(self, db_session, document_with_chunks):
```
**Impact:** Fixes 2 test errors

**Total Phase 1 Impact:** ~58 failures fixed (19% of all failures)

---

### Phase 2: Repository Integration Fixes

**Fix assertion errors in `test_repository_integration.py`:**

1. **Pagination test** - Check why 6 documents returned instead of 5
2. **Status filtering** - Verify filter logic returns correct count
3. **NumPy array comparison** - Use proper array comparison:
   ```python
   # Instead of: if embedding == expected
   # Use: if np.array_equal(embedding, expected)
   ```

**Impact:** Fixes 6+ failures

---

### Phase 3: Environment Configuration

**Create `.env.test` with required variables:**

```bash
# Database
TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/ragtools_test

# LLM Services (for integration tests)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional
OLLAMA_BASE_URL=http://localhost:11434
```

**Or:** Add pytest markers to skip tests requiring external services in CI

**Impact:** Enables ~100+ integration tests to run properly

---

### Phase 4: Code Quality

1. **Fix `test_pipeline.py`** (46 failures) - requires investigation
2. **Fix documentation links** (6 failures)
3. **Validate code examples** (4 failures)
4. **Review agentic strategy** (16 failures)

---

## Environment Setup

### Required Dependencies

```bash
# Core (already installed)
pip install pytest pytest-asyncio pytest-cov

# Missing dependency causing failures
pip install sentence-transformers

# Optional (for full test coverage)
pip install torch  # For some embedding models
pip install sklearn  # For some strategies
```

### Environment Variables

```bash
# Minimal for database tests
export TEST_DATABASE_URL="postgresql://user:pass@localhost:5432/ragtools_test"

# For LLM integration tests
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Test Execution Strategy

### For Local Development

```bash
# Run tests that don't require external dependencies
pytest tests/unit/cli -v
pytest tests/unit/evaluation -v
pytest tests/unit/observability -v
pytest tests/unit/repositories -v

# After installing sentence-transformers
pytest tests/benchmarks -v
pytest tests/integration/models -v
```

### For CI/CD

**Recommended approach:**

1. **Add pytest markers:**
```python
@pytest.mark.requires_sentence_transformers
@pytest.mark.requires_api_key
@pytest.mark.requires_database
```

2. **Configure CI to skip external dependencies:**
```bash
pytest -m "not requires_api_key and not requires_database"
```

3. **Separate test suites:**
- Unit tests (no external deps) - Run always
- Integration tests (mocked services) - Run always
- E2E tests (real services) - Run on demand

---

## Summary Statistics

### Failures by Root Cause

| Root Cause | Count | % of Failures | Fix Difficulty |
|------------|-------|---------------|----------------|
| Missing sentence-transformers | 32+ | 10.6% | ⚡ Easy |
| Invalid onnx-local provider | 24+ | 7.9% | ⚡ Easy |
| Missing API keys/LLM services | 100+ | 33.1% | 🔧 Config |
| Test fixture issues | 4 | 1.3% | ⚡ Easy |
| Pipeline logic errors | 46 | 15.2% | 🔨 Medium |
| Repository assertions | 6 | 2.0% | 🔧 Medium |
| Documentation issues | 12 | 4.0% | 🔧 Medium |
| Other integration issues | 78 | 25.8% | 🔨 Medium |

### Quick Win Potential

- **Easy fixes (1-2 commands):** ~60 failures (20%)
- **Config fixes (.env setup):** ~100 failures (33%)
- **Code fixes (investigation needed):** ~142 failures (47%)

---

## Conclusion

The test suite has an **82.7% pass rate** with **302 failures** that can be systematically addressed:

### ✅ Immediate Actions (Fixes ~20% of failures)
1. `pip install sentence-transformers`
2. Add `'onnx-local'` to valid providers
3. Fix test fixture references

### 🔧 Short Term (Fixes ~33% of failures)
4. Configure test environment variables
5. Add pytest markers for external dependencies
6. Fix repository integration assertions

### 🔨 Medium Term (Fixes remaining ~47%)
7. Investigate pipeline test failures (46 tests)
8. Fix agentic strategy tests (16 tests)
9. Update documentation and links
10. Review remaining integration test issues

**Expected Outcome:** With Phase 1 & 2 complete, pass rate should improve to **90%+**

---

## Detailed Failed Test Files

See `test_summary_by_file.txt` for complete list of 35 failed test files.

**Highest Priority Files:**
1. ✅ Install sentence-transformers → Fixes benchmarks + late chunking
2. ✅ Fix onnx-local provider → Fixes embedding tests
3. 🔨 Investigate `test_pipeline.py` → 46 failures
4. 🔨 Investigate `test_strategy.py` (agentic) → 16 failures
