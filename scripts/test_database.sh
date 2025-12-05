#!/bin/bash
# Test database setup script
# Creates a separate test database for running tests

set -e

echo "======================================"
echo "RAG Factory Test Database Setup"
echo "======================================"
echo ""

# Configuration
DB_NAME="${DB_TEST_NAME:-rag_factory_test}"
DB_USER="${DB_USER:-postgres}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if PostgreSQL is running
if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" &> /dev/null; then
    print_error "PostgreSQL is not running on $DB_HOST:$DB_PORT"
    exit 1
fi

print_success "PostgreSQL is running"

# Drop existing test database if it exists
print_info "Dropping existing test database if present"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "DROP DATABASE IF EXISTS $DB_NAME;" 2>/dev/null || true

# Create test database
print_info "Creating test database: $DB_NAME"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "CREATE DATABASE $DB_NAME;"
print_success "Test database created: $DB_NAME"

# Install pgvector extension
print_info "Installing pgvector extension"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;"
print_success "pgvector extension installed"

# Set environment variable
export DB_DATABASE_URL="postgresql://$DB_USER@$DB_HOST:$DB_PORT/$DB_NAME"

# Run migrations
print_info "Running migrations"
alembic upgrade head
print_success "Migrations complete"

echo ""
echo "======================================"
echo "Test Database Setup Complete!"
echo "======================================"
echo ""
echo "Database URL: $DB_DATABASE_URL"
echo ""
echo "Run tests with:"
echo "  export DB_DATABASE_URL=\"$DB_DATABASE_URL\""
echo "  pytest tests/"
echo ""
