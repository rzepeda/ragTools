#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting End-to-End CLI Validation${NC}"

# Ensure sample docs exist
if [ ! -d "sample-docs" ]; then
    echo -e "${RED}Error: sample-docs directory missing${NC}"
    exit 1
fi

# Ensure cli-config.yaml exists
if [ ! -f "cli-config.yaml" ]; then
    echo -e "${RED}Error: cli-config.yaml missing${NC}"
    exit 1
fi

# Run Validation
echo "Running rag-factory validate-e2e..."
PYTHON_CMD="python3"
if [ -d "venv" ]; then
    PYTHON_CMD="./venv/bin/python"
fi
$PYTHON_CMD -m rag_factory.cli.main validate-e2e --config cli-config.yaml

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Validation Successful!${NC}"
else
    echo -e "${RED}Validation Failed!${NC}"
    exit 1
fi
