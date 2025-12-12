#!/bin/bash
# Monitor the progress of file-based test execution

RESULTS_FILE="test_results_by_file.txt"
SUMMARY_FILE="test_summary_by_file.txt"

echo "Test Execution Monitor (By File)"
echo "================================="
echo ""

# Check if results file exists
if [ ! -f "$RESULTS_FILE" ]; then
    echo "No results file found yet. Tests may still be collecting..."
    exit 0
fi

# Count current progress
TOTAL_LINES=$(grep -c "^\[" "$RESULTS_FILE" 2>/dev/null | tr -d '\n' || echo "0")
PASSED=$(grep -c "PASSED:" "$RESULTS_FILE" 2>/dev/null | tr -d '\n' || echo "0")
FAILED=$(grep -c "FAILED:" "$RESULTS_FILE" 2>/dev/null | tr -d '\n' || echo "0")
SOME_FAILED=$(grep -c "SOME_FAILED:" "$RESULTS_FILE" 2>/dev/null | tr -d '\n' || echo "0")
SKIPPED=$(grep -c "ALL_SKIPPED:" "$RESULTS_FILE" 2>/dev/null | tr -d '\n' || echo "0")
TIMEOUT=$(grep -c "TIMEOUT:" "$RESULTS_FILE" 2>/dev/null | tr -d '\n' || echo "0")

# Clean up values
TOTAL_LINES=$(echo "$TOTAL_LINES" | tr -d ' \n\r')
PASSED=$(echo "$PASSED" | tr -d ' \n\r')
FAILED=$(echo "$FAILED" | tr -d ' \n\r')
SOME_FAILED=$(echo "$SOME_FAILED" | tr -d ' \n\r')
SKIPPED=$(echo "$SKIPPED" | tr -d ' \n\r')
TIMEOUT=$(echo "$TIMEOUT" | tr -d ' \n\r')

# Default to 0 if empty
TOTAL_LINES=${TOTAL_LINES:-0}
PASSED=${PASSED:-0}
FAILED=${FAILED:-0}
SOME_FAILED=${SOME_FAILED:-0}
SKIPPED=${SKIPPED:-0}
TIMEOUT=${TIMEOUT:-0}

TOTAL_FAILED=$((FAILED + SOME_FAILED))
COMPLETED=$((PASSED + TOTAL_FAILED + SKIPPED + TIMEOUT))

echo "Progress: $COMPLETED test files completed"
echo ""
echo "Current Stats:"
echo "  ✅ PASSED FILES    : $PASSED"
echo "  ❌ FAILED FILES    : $TOTAL_FAILED"
echo "  ⏭️  SKIPPED FILES   : $SKIPPED"
echo "  ⏱️  TIMEOUT FILES   : $TIMEOUT"
echo ""

# Show last few test files
echo "Last 5 test file results:"
grep "^\[" "$RESULTS_FILE" | tail -n 5

# Check if summary exists (means tests are done)
if [ -f "$SUMMARY_FILE" ]; then
    echo ""
    echo "✓ Test execution completed!"
    echo "See $SUMMARY_FILE for full summary"
fi
