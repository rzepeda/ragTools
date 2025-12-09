"""Database migration management.

This module provides a simple migration system for managing database schema
changes, versioning, and rollbacks.
"""

import logging
import os
from typing import List, Optional
from datetime import datetime
from rag_factory.services.interfaces import IDatabaseService

logger = logging.getLogger(__name__)

class MigrationManager:
    """Manager for database migrations.
    
    Handles discovery and execution of SQL migration scripts, tracking
    schema versions, and managing rollbacks.
    """
    
    def __init__(self, db_service: IDatabaseService, migrations_dir: str = "migrations"):
        """Initialize migration manager.
        
        Args:
            db_service: Database service instance
            migrations_dir: Directory containing migration scripts
        """
        self.db_service = db_service
        self.migrations_dir = migrations_dir
        
    async def init_migration_table(self) -> None:
        """Initialize the schema_migrations table."""
        pool = await self.db_service._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version VARCHAR(50) PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions.
        
        Returns:
            List of version strings
        """
        await self.init_migration_table()
        pool = await self.db_service._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
        return [row['version'] for row in rows]
        
    async def run_migrations(self) -> List[str]:
        """Run pending migrations.
        
        Returns:
            List of executed migration versions
        """
        await self.init_migration_table()
        applied = set(await self.get_applied_migrations())
        
        # Discover migration files
        if not os.path.exists(self.migrations_dir):
            return []
            
        files = sorted([f for f in os.listdir(self.migrations_dir) if f.endswith('.sql')])
        executed = []
        
        pool = await self.db_service._get_pool()
        
        for file in files:
            version = file.split('_')[0]
            if version in applied:
                continue
                
            path = os.path.join(self.migrations_dir, file)
            with open(path, 'r') as f:
                sql = f.read()
                
            logger.info(f"Applying migration {file}...")
            
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(sql)
                    await conn.execute(
                        "INSERT INTO schema_migrations (version) VALUES ($1)",
                        version
                    )
            
            executed.append(file)
            logger.info(f"Applied migration {file}")
            
        return executed

    async def get_current_version(self) -> Optional[str]:
        """Get the current schema version.
        
        Returns:
            Current version string or None if no migrations applied
        """
        applied = await self.get_applied_migrations()
        return applied[-1] if applied else None
