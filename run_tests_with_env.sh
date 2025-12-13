#!/bin/bash
# run_tests_with_env.sh - Run tests with .env configuration

set -e  # Exit on error

# Output file
OUTPUT_FILE="test_results.txt"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "✗ Virtual environment not found at venv/bin/activate"
    exit 1
fi

# Load .env file
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    echo "✓ Environment variables loaded from .env"
    echo "  EMBEDDING_MODEL_NAME: $EMBEDDING_MODEL_NAME"
    echo "  EMBEDDING_MODEL_PATH: $EMBEDDING_MODEL_PATH"
else
    echo "⚠ .env file not found, using defaults"
fi

# Create header for output file
{
    echo "========================================"
    echo "Test Results - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo "Environment Configuration:"
    echo "  EMBEDDING_MODEL_NAME: $EMBEDDING_MODEL_NAME"
    echo "  EMBEDDING_MODEL_PATH: $EMBEDDING_MODEL_PATH"
    echo "  LM_STUDIO_BASE_URL: $LM_STUDIO_BASE_URL"
    echo "  LM_STUDIO_MODEL: $LM_STUDIO_MODEL"
    echo ""
    echo "Command: pytest $@"
    echo "========================================"
    echo ""
} > "$OUTPUT_FILE"

# Run tests and output to both terminal and file
echo ""
echo "Running tests..."
echo "Output will be saved to: $OUTPUT_FILE"
echo ""

pytest "$@" 2>&1 | tee -a "$OUTPUT_FILE"

# Add footer
{
    echo ""
    echo "========================================"
    echo "Test run completed at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
} >> "$OUTPUT_FILE"

echo ""
echo "✓ Results saved to: $OUTPUT_FILE"
