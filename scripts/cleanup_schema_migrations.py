#!/usr/bin/env python3
"""
Script to clean up deprecated schema_migrations table.

This table was used by the old MigrationManager system.
The project now uses Alembic, which uses the alembic_version table.

Usage:
    python scripts/cleanup_schema_migrations.py
"""

import os
import sys
from sqlalchemy import create_engine, text


def cleanup_schema_migrations():
    """Drop deprecated schema_migrations table."""
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL") or os.getenv("DB_DATABASE_URL")
    if not database_url:
        print("❌ ERROR: DATABASE_URL or DB_DATABASE_URL not set")
        print("\nPlease set one of these environment variables:")
        print("  export DATABASE_URL='postgresql://user:pass@host:port/database'")
        print("  export DB_DATABASE_URL='postgresql://user:pass@host:port/database'")
        sys.exit(1)
    
    print("⚠️  This will drop the 'schema_migrations' table")
    print("   This table is no longer used (replaced by 'alembic_version')")
    print(f"\n   Database: {database_url.split('@')[-1]}")  # Show only host/db part
    
    response = input("\nContinue? (yes/no): ")
    if response.lower() != "yes":
        print("❌ Cancelled")
        sys.exit(0)
    
    # Connect and drop table
    try:
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            # Check if table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = 'schema_migrations'
                )
            """))
            exists = result.scalar()
            
            if not exists:
                print("✅ Table 'schema_migrations' does not exist (already clean)")
                return
            
            # Drop table
            print("🗑️  Dropping 'schema_migrations' table...")
            conn.execute(text("DROP TABLE schema_migrations"))
            conn.commit()
            
            print("✅ Successfully dropped 'schema_migrations' table")
            print("   Migration tracking now uses 'alembic_version' table only")
        
        engine.dispose()
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cleanup_schema_migrations()
