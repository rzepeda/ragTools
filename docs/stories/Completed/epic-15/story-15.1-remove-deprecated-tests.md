# Story 15.1: Remove Deprecated Tests

**Story ID:** 15.1  
**Epic:** Epic 15 - Test Coverage Improvements  
**Story Points:** 2  
**Priority:** Critical  
**Dependencies:** None

---

## User Story

**As a** developer  
**I want** to remove tests for deprecated interfaces  
**So that** the test suite only covers current architecture and doesn't confuse contributors

---

## Detailed Requirements

### Functional Requirements

1. **Identify Deprecated Tests**
   - Locate all test files that test the old `IRAGStrategy` interface
   - Verify these tests are truly deprecated (interface no longer exists in codebase)
   - Document which tests are being removed and why

2. **Remove Test Files**
   - Delete `tests/unit/strategies/test_base.py` (229 lines)
   - Delete `tests/unit/strategies/test_base_strategy_di.py` (253 lines)
   - Total: 482 lines of deprecated test code

3. **Verify Test Suite**
   - Run full test suite after removal
   - Ensure no other tests depend on the removed files
   - Verify all remaining tests pass

4. **Update Documentation**
   - Update any test documentation that references removed tests
   - Add note to changelog about deprecated test removal

### Non-Functional Requirements

1. **Safety**
   - Only remove tests that are confirmed deprecated
   - Do not remove any tests for current interfaces

2. **Verification**
   - All remaining tests must pass after removal
   - No import errors or missing dependencies

---

## Acceptance Criteria

### AC1: Deprecated Test Identification
- [ ] `tests/unit/strategies/test_base.py` confirmed to test deprecated `IRAGStrategy`
- [ ] `tests/unit/strategies/test_base_strategy_di.py` confirmed to test deprecated DI pattern
- [ ] No other files import or depend on these test files

### AC2: File Removal
- [ ] `tests/unit/strategies/test_base.py` deleted from repository
- [ ] `tests/unit/strategies/test_base_strategy_di.py` deleted from repository
- [ ] Git commit message clearly explains removal reason

### AC3: Test Suite Verification
- [ ] Full test suite runs successfully after removal
- [ ] No import errors related to removed files
- [ ] Test count decreased by expected amount
- [ ] All remaining tests pass (100% success rate)

### AC4: Documentation Updates
- [ ] Changelog updated with deprecation note
- [ ] Any test documentation referencing removed files updated
- [ ] PR description clearly explains what was removed and why

---

## Technical Specifications

### Files to Remove

1. **`tests/unit/strategies/test_base.py`**
   - **Reason**: Tests deprecated `IRAGStrategy` interface
   - **Replacement**: Current architecture uses `IIndexingStrategy` and `IRetrievalStrategy` (tested in `test_indexing_interface.py` and `test_retrieval_interface.py`)
   - **Lines**: 229
   - **Tests Removed**: 
     - Interface definition tests
     - Configuration dataclass tests
     - Chunk, PreparedData, QueryResult dataclass tests
     - Concrete implementation tests
     - Type hint validation tests

2. **`tests/unit/strategies/test_base_strategy_di.py`**
   - **Reason**: Tests DI for deprecated `IRAGStrategy` interface
   - **Replacement**: New interfaces have their own DI validation
   - **Lines**: 253
   - **Tests Removed**:
     - Strategy initialization with services
     - Missing service validation
     - Error message validation
     - Service requirement combinations

### Verification Commands

```bash
# Before removal - count tests
pytest tests/ --collect-only | grep "test session starts"

# Remove files
rm tests/unit/strategies/test_base.py
rm tests/unit/strategies/test_base_strategy_di.py

# After removal - verify tests still pass
pytest tests/ -v

# Verify no imports of removed files
grep -r "from.*test_base import" tests/
grep -r "from.*test_base_strategy_di import" tests/
```

### Expected Impact

- **Tests Removed**: ~20-25 test functions
- **Lines Removed**: 482 lines
- **Coverage Impact**: Minimal (deprecated code already removed from main codebase)
- **Build Time**: Slightly faster (fewer tests to run)

---

## Testing Strategy

### Pre-Removal Verification
1. Confirm files test deprecated interfaces
2. Search for any imports of these test files
3. Check if any fixtures or utilities are shared

### Post-Removal Verification
1. Run full test suite: `pytest tests/ -v`
2. Check for import errors: `pytest tests/ --collect-only`
3. Verify test count decreased appropriately
4. Run linting: `pylint tests/`
5. Run type checking: `mypy tests/`

---

## Definition of Done

- [ ] Both deprecated test files removed from repository
- [ ] Full test suite passes (100% success rate)
- [ ] No import errors or missing dependencies
- [ ] Changelog updated
- [ ] PR approved and merged
- [ ] CI/CD pipeline passes

---

## Notes

- This is a **critical** priority because deprecated tests can confuse new contributors
- The removal is **safe** because the interfaces being tested no longer exist in the codebase
- This cleanup was identified in the test coverage analysis performed on 2025-12-08
