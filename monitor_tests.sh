#!/bin/bash
# Monitor the progress of individual test execution

RESULTS_FILE="individual_test_results.txt"
SUMMARY_FILE="test_summary.txt"

echo "Test Execution Monitor"
echo "======================"
echo ""

# Check if results file exists
if [ ! -f "$RESULTS_FILE" ]; then
    echo "No results file found yet. Tests may still be collecting..."
    exit 0
fi

# Count current progress
TOTAL_LINES=$(wc -l < "$RESULTS_FILE" 2>/dev/null || echo "0")
PASSED=$(grep -c "PASSED:" "$RESULTS_FILE" 2>/dev/null || echo "0")
FAILED=$(grep -c "FAILED:" "$RESULTS_FILE" 2>/dev/null || echo "0")
SKIPPED=$(grep -c "SKIPPED:" "$RESULTS_FILE" 2>/dev/null || echo "0")
TIMEOUT=$(grep -c "TIMEOUT:" "$RESULTS_FILE" 2>/dev/null || echo "0")

COMPLETED=$((PASSED + FAILED + SKIPPED + TIMEOUT))

echo "Progress: $COMPLETED tests completed"
echo ""
echo "Current Stats:"
echo "  ✅ PASSED : $PASSED"
echo "  ❌ FAILED : $FAILED"
echo "  ⏭️  SKIPPED: $SKIPPED"
echo "  ⏱️  TIMEOUT: $TIMEOUT"
echo ""

# Show last few tests
echo "Last 5 test results:"
tail -n 10 "$RESULTS_FILE" | grep -E "PASSED:|FAILED:|SKIPPED:|TIMEOUT:" | tail -n 5

# Check if summary exists (means tests are done)
if [ -f "$SUMMARY_FILE" ]; then
    echo ""
    echo "✓ Test execution completed!"
    echo "See $SUMMARY_FILE for full summary"
fi
