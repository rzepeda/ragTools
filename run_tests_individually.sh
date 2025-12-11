#!/bin/bash
# Script to run each test individually and save results
# This helps identify failing or hanging tests

TIMEOUT=60  # seconds per test
RESULTS_FILE="individual_test_results.txt"
SUMMARY_FILE="test_summary.txt"
FAILED_FILE="failed_tests.txt"
SKIPPED_FILE="skipped_tests.txt"
TIMEOUT_FILE="timeout_tests.txt"

# Initialize counters
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0
TIMEOUT_COUNT=0
ERROR=0

# Clear previous results
> "$RESULTS_FILE"
> "$FAILED_FILE"
> "$SKIPPED_FILE"
> "$TIMEOUT_FILE"

echo "================================================================================"
echo "Individual Test Runner"
echo "================================================================================"
echo ""

# Activate virtual environment
source venv/bin/activate

# Get test list from pytest
echo "Collecting tests..."
pytest --collect-only --quiet --quiet 2>&1 | grep -E '^tests/' > test_list.txt
TOTAL=$(wc -l < test_list.txt)
echo "Found $TOTAL tests"
echo ""

# Run each test
INDEX=0
START_TIME=$(date +%s)

while IFS= read -r test_name; do
    INDEX=$((INDEX + 1))
    echo "[$INDEX/$TOTAL] Running: $test_name"
    
    # Run test with timeout
    timeout $TIMEOUT pytest "$test_name" -v --tb=short > test_output.tmp 2>&1
    EXIT_CODE=$?
    
    # Determine status
    if [ $EXIT_CODE -eq 124 ]; then
        # Timeout
        STATUS="TIMEOUT"
        TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
        echo "  ⏱️  TIMEOUT after ${TIMEOUT}s"
        echo "$test_name" >> "$TIMEOUT_FILE"
    elif [ $EXIT_CODE -eq 0 ]; then
        if grep -q "PASSED" test_output.tmp; then
            STATUS="PASSED"
            PASSED=$((PASSED + 1))
            echo "  ✅ PASSED"
        elif grep -q "SKIPPED" test_output.tmp; then
            STATUS="SKIPPED"
            SKIPPED=$((SKIPPED + 1))
            echo "  ⏭️  SKIPPED"
            echo "$test_name" >> "$SKIPPED_FILE"
        else
            STATUS="UNKNOWN"
            ERROR=$((ERROR + 1))
            echo "  ❓ UNKNOWN"
        fi
    else
        STATUS="FAILED"
        FAILED=$((FAILED + 1))
        echo "  ❌ FAILED"
        echo "$test_name" >> "$FAILED_FILE"
    fi
    
    # Save result
    echo "[$INDEX/$TOTAL] $STATUS: $test_name" >> "$RESULTS_FILE"
    
    # Save detailed output for failed/timeout tests
    if [ "$STATUS" = "FAILED" ] || [ "$STATUS" = "TIMEOUT" ]; then
        echo "----------------------------------------" >> "$RESULTS_FILE"
        cat test_output.tmp >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
    fi
    
done < test_list.txt

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))

# Clean up
rm -f test_output.tmp test_list.txt

# Print summary
echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo "Total tests: $TOTAL"
echo "Total duration: ${DURATION}s (${MINUTES} minutes)"
echo ""
echo "  PASSED    : $PASSED ($(awk "BEGIN {printf \"%.1f\", $PASSED/$TOTAL*100}")%)"
echo "  FAILED    : $FAILED ($(awk "BEGIN {printf \"%.1f\", $FAILED/$TOTAL*100}")%)"
echo "  SKIPPED   : $SKIPPED ($(awk "BEGIN {printf \"%.1f\", $SKIPPED/$TOTAL*100}")%)"
echo "  TIMEOUT   : $TIMEOUT_COUNT ($(awk "BEGIN {printf \"%.1f\", $TIMEOUT_COUNT/$TOTAL*100}")%)"
echo "  ERROR     : $ERROR ($(awk "BEGIN {printf \"%.1f\", $ERROR/$TOTAL*100}")%)"
echo ""

# Save summary
{
    echo "================================================================================"
    echo "TEST RESULTS SUMMARY"
    echo "================================================================================"
    echo ""
    echo "Total tests: $TOTAL"
    echo "Total duration: ${DURATION}s (${MINUTES} minutes)"
    echo ""
    echo "  PASSED    : $PASSED ($(awk "BEGIN {printf \"%.1f\", $PASSED/$TOTAL*100}")%)"
    echo "  FAILED    : $FAILED ($(awk "BEGIN {printf \"%.1f\", $FAILED/$TOTAL*100}")%)"
    echo "  SKIPPED   : $SKIPPED ($(awk "BEGIN {printf \"%.1f\", $SKIPPED/$TOTAL*100}")%)"
    echo "  TIMEOUT   : $TIMEOUT_COUNT ($(awk "BEGIN {printf \"%.1f\", $TIMEOUT_COUNT/$TOTAL*100}")%)"
    echo "  ERROR     : $ERROR ($(awk "BEGIN {printf \"%.1f\", $ERROR/$TOTAL*100}")%)"
    echo ""
    echo "================================================================================"
    echo "FAILED TESTS ($FAILED)"
    echo "================================================================================"
    if [ -f "$FAILED_FILE" ]; then
        cat "$FAILED_FILE"
    fi
    echo ""
    echo "================================================================================"
    echo "TIMEOUT TESTS ($TIMEOUT_COUNT)"
    echo "================================================================================"
    if [ -f "$TIMEOUT_FILE" ]; then
        cat "$TIMEOUT_FILE"
    fi
    echo ""
    echo "================================================================================"
    echo "SKIPPED TESTS ($SKIPPED)"
    echo "================================================================================"
    if [ -f "$SKIPPED_FILE" ]; then
        cat "$SKIPPED_FILE"
    fi
} > "$SUMMARY_FILE"

echo "Results saved to: $RESULTS_FILE"
echo "Summary saved to: $SUMMARY_FILE"
echo "Failed tests: $FAILED_FILE"
echo "Skipped tests: $SKIPPED_FILE"
echo "Timeout tests: $TIMEOUT_FILE"
