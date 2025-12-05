#!/bin/bash
# Database setup script for local development
# This script sets up PostgreSQL with pgvector extension for RAG Factory

set -e

echo "======================================"
echo "RAG Factory Database Setup"
echo "======================================"
echo ""

# Configuration
DB_NAME="${DB_NAME:-rag_factory_dev}"
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

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    print_error "PostgreSQL is not installed"
    echo ""
    echo "Install PostgreSQL:"
    echo "  Ubuntu/Debian: sudo apt-get install postgresql-15 postgresql-contrib"
    echo "  macOS:         brew install postgresql@15"
    exit 1
fi

print_success "PostgreSQL is installed"

# Check if PostgreSQL is running
if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" &> /dev/null; then
    print_error "PostgreSQL is not running on $DB_HOST:$DB_PORT"
    echo ""
    echo "Start PostgreSQL:"
    echo "  Ubuntu/Debian: sudo systemctl start postgresql"
    echo "  macOS:         brew services start postgresql@15"
    exit 1
fi

print_success "PostgreSQL is running"

# Create database if it doesn't exist
print_info "Creating database: $DB_NAME"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1 || \
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "CREATE DATABASE $DB_NAME;"
print_success "Database created: $DB_NAME"

# Install pgvector extension
print_info "Installing pgvector extension"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;"
print_success "pgvector extension installed"

# Set environment variable
export DB_DATABASE_URL="postgresql://$DB_USER@$DB_HOST:$DB_PORT/$DB_NAME"

echo ""
echo "======================================"
echo "Database Setup Complete!"
echo "======================================"
echo ""
echo "Database URL: $DB_DATABASE_URL"
echo ""
echo "Next steps:"
echo "  1. Set environment variable:"
echo "     export DB_DATABASE_URL=\"$DB_DATABASE_URL\""
echo ""
echo "  2. Run migrations:"
echo "     alembic upgrade head"
echo ""
echo "  3. Verify setup:"
echo "     python -c 'from rag_factory.database import DatabaseConnection; db = DatabaseConnection(); print(db.health_check())'"
echo ""
