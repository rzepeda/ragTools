#!/usr/bin/env python3
"""Setup test database for integration tests.

This script creates the test database if it doesn't exist.
Run this before running integration tests.
"""

import os
import sys
from urllib.parse import urlparse
from sqlalchemy import create_engine, text


def setup_test_database():
    """Create test database if it doesn't exist."""
    # Get test database URL from environment
    test_db_url = os.getenv("DB_TEST_DATABASE_URL")
    if not test_db_url:
        print("ERROR: DB_TEST_DATABASE_URL not set")
        print("\nPlease set the DB_TEST_DATABASE_URL environment variable:")
        print('  export DB_TEST_DATABASE_URL="postgresql://user:password@localhost:5432/rag_test"')
        sys.exit(1)

    # Parse database URL
    parsed = urlparse(test_db_url)
    db_name = parsed.path.lstrip('/')

    if not db_name:
        print("ERROR: Database name not found in DB_TEST_DATABASE_URL")
        print(f"URL: {test_db_url}")
        sys.exit(1)

    # Create URL for postgres database (to create test db)
    postgres_url = test_db_url.replace(f"/{db_name}", "/postgres")
    
    print(f"Setting up test database: {db_name}")
    print(f"Host: {parsed.hostname}")
    print(f"Port: {parsed.port or 5432}")
    print()
    
    try:
        # Connect to postgres database
        print("Connecting to postgres database...")
        engine = create_engine(postgres_url, isolation_level="AUTOCOMMIT")
        
        with engine.connect() as conn:
            # Drop existing test database if exists
            print(f"Dropping existing database '{db_name}' (if exists)...")
            conn.execute(text(f"DROP DATABASE IF EXISTS {db_name}"))
            
            # Create test database
            print(f"Creating database '{db_name}'...")
            conn.execute(text(f"CREATE DATABASE {db_name}"))
        
        engine.dispose()
        print("✓ Database created successfully")
        print()
        
        # Connect to test database and install extensions
        print("Installing extensions...")
        test_engine = create_engine(test_db_url)
        
        with test_engine.connect() as conn:
            # Install pgvector extension
            print("  - Installing pgvector extension...")
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            print("  ✓ pgvector extension installed")
        
        test_engine.dispose()
        
        print()
        print("=" * 60)
        print("✅ Test database setup complete!")
        print("=" * 60)
        print()
        print("You can now run database tests:")
        print("  pytest tests/unit/database/ -v")
        print("  pytest tests/integration/database/ -v")
        print()
        
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ Error setting up test database")
        print("=" * 60)
        print("\nTest database setup failed!")
        print("Please check:")
        print("1. PostgreSQL is running")
        print("2. Check connection parameters in DB_TEST_DATABASE_URL")
        print("3. User has permission to create databases")
        print("4. Ensure pgvector extension is installed on PostgreSQL")
        print()
        sys.exit(1)


if __name__ == "__main__":
    setup_test_database()
