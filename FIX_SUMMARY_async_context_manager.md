# Fix Summary: Async Context Manager Mock Error

## Date: 2025-12-15

## Error Pattern Fixed
```
TypeError: 'coroutine' object does not support the asynchronous context manager protocol
```

## Root Cause
The mock for `pool.acquire()` was incorrectly configured. When using `AsyncMock` for the `acquire()` method, it returns a coroutine that needs to be awaited. However, the actual code uses `async with pool.acquire()` which expects `acquire()` to return an async context manager directly, not a coroutine.

## Solution
Changed the mock configuration to use `MagicMock` for `pool.acquire()` so it returns an async context manager directly without being a coroutine. This matches the actual behavior of `asyncpg.Pool.acquire()`.

## Files Modified

### 1. `/mnt/MCPProyects/ragTools/tests/integration/services/test_service_integration.py`

**Lines 75-93**: Fixed database_service fixture
```python
# Before:
pool.acquire.return_value.__aenter__.return_value = conn

# After:
acquire_cm = AsyncMock()
acquire_cm.__aenter__.return_value = conn
acquire_cm.__aexit__.return_value = None
# Make acquire() return the context manager directly (not as a coroutine)
pool.acquire = MagicMock(return_value=acquire_cm)
```

## Technical Details

The issue was subtle:
1. `AsyncMock().method()` returns a coroutine that must be awaited
2. But `async with pool.acquire()` expects `acquire()` to return an object that supports the async context manager protocol directly
3. The fix uses `MagicMock` for `acquire()` which returns the async context manager synchronously

## Test Results

### Before Fix
- **1 test failing** with async context manager error:
  - `test_embedding_database_consistency`

### After Fix
- **1 test passing** ✅

### Test Command Used
```bash
source venv/bin/activate && pytest tests/integration/services/test_service_integration.py::test_embedding_database_consistency -xvs
```

## Impact
- ✅ **100% resolution** of async context manager mock errors
- ✅ **Better mock configuration** - properly mimics asyncpg.Pool behavior
- ✅ **Improved test reliability** - tests now correctly mock async database operations

## Related Code Pattern
This same pattern should be used whenever mocking async context managers:
```python
# Create async context manager
async_cm = AsyncMock()
async_cm.__aenter__.return_value = mock_object
async_cm.__aexit__.return_value = None

# Make the method return it directly (not as a coroutine)
mock.method = MagicMock(return_value=async_cm)
```

## Verification
The test now correctly:
1. Mocks the PostgreSQL connection pool
2. Simulates async context manager behavior
3. Verifies database operations work correctly with mocked connections

This fix ensures that database integration tests can run without requiring an actual PostgreSQL database.
