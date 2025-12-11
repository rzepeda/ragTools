# Test Runner - By File

This test runner executes tests by file instead of individually, making it much faster.

## Quick Start

```bash
cd /mnt/MCPProyects/ragTools/tests
./run_tests_by_file.sh
```

**Note:** The script automatically loads environment variables from `/mnt/MCPProyects/ragTools/.env` if it exists.

## Features

- ✅ Runs all test files with 5-minute timeout per file
- 📊 Tracks both file-level and test-level statistics
- 🚀 Much faster than individual test execution (~10-20 minutes vs 30-60 minutes)
- 💾 Saves results incrementally
- 🔍 Detailed output for failed tests

## Output Files

All files are created in the `tests/` directory:

- **`test_summary_by_file.txt`** - Summary with statistics
- **`test_results_by_file.txt`** - Detailed results for all test files
- **`failed_test_files.txt`** - List of files with failed tests
- **`skipped_test_files.txt`** - List of files with all tests skipped
- **`timeout_test_files.txt`** - List of files that timed out

## Monitoring Progress

In another terminal:

```bash
cd /mnt/MCPProyects/ragTools/tests
./monitor_file_tests.sh
```

Or check manually:

```bash
# See current progress
tail -20 test_results_by_file.txt

# Count completed files
grep -c "^\[" test_results_by_file.txt
```

## Understanding Results

### File Status

- **✅ PASSED** - All tests in file passed (some may be skipped)
- **❌ FAILED** - All tests in file failed or had errors
- **⚠️ SOME_FAILED** - Mix of passed and failed tests
- **⏭️ ALL_SKIPPED** - All tests in file were skipped
- **⏱️ TIMEOUT** - File execution exceeded 5-minute timeout

### Test Counts

Each file result shows:
- **P** - Passed tests
- **F** - Failed tests
- **S** - Skipped tests

## Example Output

```
[1/150] Running: tests/unit/test_example.py
  ✅ PASSED (25 passed, 3 skipped)

[2/150] Running: tests/integration/test_api.py
  ⚠️  SOME FAILED (10 passed, 2 failed, 1 skipped)
```

## After Completion

Once complete, review:

1. `test_summary_by_file.txt` - Overall statistics
2. `failed_test_files.txt` - Files needing attention
3. `test_results_by_file.txt` - Detailed failure information
