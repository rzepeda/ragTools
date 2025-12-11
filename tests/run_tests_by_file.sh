#!/bin/bash
# Script to run tests by file instead of individually
# This is much faster than running each test separately

TIMEOUT=300  # 5 minutes per file (increased for files with many tests)
RESULTS_FILE="test_results_by_file.txt"
SUMMARY_FILE="test_summary_by_file.txt"
FAILED_FILE="failed_test_files.txt"
SKIPPED_FILE="skipped_test_files.txt"
TIMEOUT_FILE="timeout_test_files.txt"

# Initialize counters
TOTAL_FILES=0
PASSED_FILES=0
FAILED_FILES=0
SKIPPED_FILES=0
TIMEOUT_COUNT=0
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Clear previous results
> "$RESULTS_FILE"
> "$FAILED_FILE"
> "$SKIPPED_FILE"
> "$TIMEOUT_FILE"

echo "================================================================================"
echo "Test Runner - By File"
echo "================================================================================"
echo ""

# Activate virtual environment
cd /mnt/MCPProyects/ragTools
source venv/bin/activate

# Load .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    set -a  # automatically export all variables
    source .env
    set +a
fi

# Get list of all test files
echo "Collecting test files..."
find tests -name "test_*.py" -type f | sort > test_files_list.txt
TOTAL_FILES=$(wc -l < test_files_list.txt)
echo "Found $TOTAL_FILES test files"
echo ""

# Run each test file
INDEX=0
START_TIME=$(date +%s)

while IFS= read -r test_file; do
    INDEX=$((INDEX + 1))
    echo "[$INDEX/$TOTAL_FILES] Running: $test_file"
    
    # Run test file with timeout
    timeout $TIMEOUT pytest "$test_file" -v --tb=short 2>&1 > test_output.tmp
    EXIT_CODE=$?
    
    # Count tests in this file
    FILE_PASSED=$(grep -c "PASSED" test_output.tmp 2>/dev/null | tr -d '\n' || echo "0")
    FILE_FAILED=$(grep -c "FAILED" test_output.tmp 2>/dev/null | tr -d '\n' || echo "0")
    FILE_SKIPPED=$(grep -c "SKIPPED" test_output.tmp 2>/dev/null | tr -d '\n' || echo "0")
    
    # Ensure we have valid numbers (remove any whitespace)
    FILE_PASSED=$(echo "$FILE_PASSED" | tr -d ' \n\r')
    FILE_FAILED=$(echo "$FILE_FAILED" | tr -d ' \n\r')
    FILE_SKIPPED=$(echo "$FILE_SKIPPED" | tr -d ' \n\r')
    
    # Default to 0 if empty
    FILE_PASSED=${FILE_PASSED:-0}
    FILE_FAILED=${FILE_FAILED:-0}
    FILE_SKIPPED=${FILE_SKIPPED:-0}
    
    FILE_TOTAL=$((FILE_PASSED + FILE_FAILED + FILE_SKIPPED))
    
    # Update total counters
    TOTAL_TESTS=$((TOTAL_TESTS + FILE_TOTAL))
    PASSED_TESTS=$((PASSED_TESTS + FILE_PASSED))
    FAILED_TESTS=$((FAILED_TESTS + FILE_FAILED))
    SKIPPED_TESTS=$((SKIPPED_TESTS + FILE_SKIPPED))


    
    # Determine file status
    if [ $EXIT_CODE -eq 124 ]; then
        # Timeout
        STATUS="TIMEOUT"
        TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
        echo "  ⏱️  TIMEOUT after ${TIMEOUT}s"
        echo "$test_file" >> "$TIMEOUT_FILE"
    elif [ $EXIT_CODE -eq 0 ]; then
        if [ $FILE_FAILED -eq 0 ]; then
            if [ $FILE_SKIPPED -eq $FILE_TOTAL ]; then
                STATUS="ALL_SKIPPED"
                SKIPPED_FILES=$((SKIPPED_FILES + 1))
                echo "  ⏭️  ALL SKIPPED ($FILE_SKIPPED tests)"
                echo "$test_file" >> "$SKIPPED_FILE"
            else
                STATUS="PASSED"
                PASSED_FILES=$((PASSED_FILES + 1))
                echo "  ✅ PASSED ($FILE_PASSED passed, $FILE_SKIPPED skipped)"
            fi
        else
            STATUS="SOME_FAILED"
            FAILED_FILES=$((FAILED_FILES + 1))
            echo "  ⚠️  SOME FAILED ($FILE_PASSED passed, $FILE_FAILED failed, $FILE_SKIPPED skipped)"
            echo "$test_file" >> "$FAILED_FILE"
        fi
    else
        STATUS="FAILED"
        FAILED_FILES=$((FAILED_FILES + 1))
        echo "  ❌ FAILED ($FILE_PASSED passed, $FILE_FAILED failed, $FILE_SKIPPED skipped)"
        echo "$test_file" >> "$FAILED_FILE"
    fi
    
    # Save result
    echo "[$INDEX/$TOTAL_FILES] $STATUS: $test_file" >> "$RESULTS_FILE"
    echo "  Tests: $FILE_TOTAL (P:$FILE_PASSED F:$FILE_FAILED S:$FILE_SKIPPED)" >> "$RESULTS_FILE"
    
    # Save detailed output for failed/timeout files
    if [ "$STATUS" = "FAILED" ] || [ "$STATUS" = "SOME_FAILED" ] || [ "$STATUS" = "TIMEOUT" ]; then
        echo "----------------------------------------" >> "$RESULTS_FILE"
        cat test_output.tmp >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
    fi
    
done < test_files_list.txt

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

# Clean up
rm -f test_output.tmp test_files_list.txt

# Calculate percentages safely
if [ $TOTAL_FILES -gt 0 ]; then
    PASSED_FILES_PCT=$(awk "BEGIN {printf \"%.1f\", $PASSED_FILES/$TOTAL_FILES*100}")
    FAILED_FILES_PCT=$(awk "BEGIN {printf \"%.1f\", $FAILED_FILES/$TOTAL_FILES*100}")
    SKIPPED_FILES_PCT=$(awk "BEGIN {printf \"%.1f\", $SKIPPED_FILES/$TOTAL_FILES*100}")
    TIMEOUT_PCT=$(awk "BEGIN {printf \"%.1f\", $TIMEOUT_COUNT/$TOTAL_FILES*100}")
else
    PASSED_FILES_PCT="0.0"
    FAILED_FILES_PCT="0.0"
    SKIPPED_FILES_PCT="0.0"
    TIMEOUT_PCT="0.0"
fi

if [ $TOTAL_TESTS -gt 0 ]; then
    PASSED_TESTS_PCT=$(awk "BEGIN {printf \"%.1f\", $PASSED_TESTS/$TOTAL_TESTS*100}")
    FAILED_TESTS_PCT=$(awk "BEGIN {printf \"%.1f\", $FAILED_TESTS/$TOTAL_TESTS*100}")
    SKIPPED_TESTS_PCT=$(awk "BEGIN {printf \"%.1f\", $SKIPPED_TESTS/$TOTAL_TESTS*100}")
else
    PASSED_TESTS_PCT="0.0"
    FAILED_TESTS_PCT="0.0"
    SKIPPED_TESTS_PCT="0.0"
fi

# Print summary
echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo "Total test files: $TOTAL_FILES"
echo "Total duration: ${MINUTES}m ${SECONDS}s"
echo ""
echo "FILE RESULTS:"
echo "  ✅ PASSED FILES    : $PASSED_FILES ($PASSED_FILES_PCT%)"
echo "  ❌ FAILED FILES    : $FAILED_FILES ($FAILED_FILES_PCT%)"
echo "  ⏭️  SKIPPED FILES   : $SKIPPED_FILES ($SKIPPED_FILES_PCT%)"
echo "  ⏱️  TIMEOUT FILES   : $TIMEOUT_COUNT ($TIMEOUT_PCT%)"
echo ""
echo "TEST RESULTS:"
echo "  ✅ PASSED TESTS    : $PASSED_TESTS ($PASSED_TESTS_PCT%)"
echo "  ❌ FAILED TESTS    : $FAILED_TESTS ($FAILED_TESTS_PCT%)"
echo "  ⏭️  SKIPPED TESTS   : $SKIPPED_TESTS ($SKIPPED_TESTS_PCT%)"
echo "  📊 TOTAL TESTS     : $TOTAL_TESTS"
echo ""

# Save summary
{
    echo "================================================================================"
    echo "TEST RESULTS SUMMARY - BY FILE"
    echo "================================================================================"
    echo ""
    echo "Execution Date: $(date)"
    echo "Total test files: $TOTAL_FILES"
    echo "Total duration: ${MINUTES}m ${SECONDS}s"
    echo ""
    echo "FILE RESULTS:"
    echo "  ✅ PASSED FILES    : $PASSED_FILES ($PASSED_FILES_PCT%)"
    echo "  ❌ FAILED FILES    : $FAILED_FILES ($FAILED_FILES_PCT%)"
    echo "  ⏭️  SKIPPED FILES   : $SKIPPED_FILES ($SKIPPED_FILES_PCT%)"
    echo "  ⏱️  TIMEOUT FILES   : $TIMEOUT_COUNT ($TIMEOUT_PCT%)"
    echo ""
    echo "TEST RESULTS:"
    echo "  ✅ PASSED TESTS    : $PASSED_TESTS ($PASSED_TESTS_PCT%)"
    echo "  ❌ FAILED TESTS    : $FAILED_TESTS ($FAILED_TESTS_PCT%)"
    echo "  ⏭️  SKIPPED TESTS   : $SKIPPED_TESTS ($SKIPPED_TESTS_PCT%)"
    echo "  📊 TOTAL TESTS     : $TOTAL_TESTS"
    echo ""
    echo "================================================================================"
    echo "FAILED TEST FILES ($FAILED_FILES)"
    echo "================================================================================"
    if [ -f "$FAILED_FILE" ]; then
        cat "$FAILED_FILE"
    fi
    echo ""
    echo "================================================================================"
    echo "TIMEOUT TEST FILES ($TIMEOUT_COUNT)"
    echo "================================================================================"
    if [ -f "$TIMEOUT_FILE" ]; then
        cat "$TIMEOUT_FILE"
    fi
    echo ""
    echo "================================================================================"
    echo "SKIPPED TEST FILES ($SKIPPED_FILES)"
    echo "================================================================================"
    if [ -f "$SKIPPED_FILE" ]; then
        cat "$SKIPPED_FILE"
    fi
} > "$SUMMARY_FILE"

echo "Results saved to: $RESULTS_FILE"
echo "Summary saved to: $SUMMARY_FILE"
echo "Failed test files: $FAILED_FILE"
echo "Skipped test files: $SKIPPED_FILE"
echo "Timeout test files: $TIMEOUT_FILE"
