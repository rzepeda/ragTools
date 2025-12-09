"""Integration tests for migrations."""
import pytest
import os
import tempfile
from rag_factory.services.database.postgres import PostgresqlDatabaseService
from rag_factory.database.migrations import MigrationManager

# Skip if no DB connection available
DB_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DB_URL,
    reason="TEST_DATABASE_URL not set"
)

class TestMigrationIntegration:
    """Integration tests for migration system."""
    
    @pytest.fixture
    async def db_service(self):
        service = PostgresqlDatabaseService(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5432)),
            database=os.getenv("DB_NAME", "test_db"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "")
        )
        yield service
        await service.close()

    @pytest.mark.asyncio
    async def test_real_migration_execution(self, db_service):
        """Test running migrations against real DB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy migration
            with open(os.path.join(tmpdir, "001_test.sql"), "w") as f:
                f.write("CREATE TABLE IF NOT EXISTS test_migration (id SERIAL PRIMARY KEY);")
                
            manager = MigrationManager(db_service, migrations_dir=tmpdir)
            
            # Run migration
            executed = await manager.run_migrations()
            assert "001_test.sql" in executed
            
            # Verify version
            version = await manager.get_current_version()
            assert version == "001"
            
            # Verify table exists
            pool = await db_service._get_pool()
            async with pool.acquire() as conn:
                # Check if table exists
                row = await conn.fetchrow(
                    "SELECT to_regclass('public.test_migration')"
                )
                assert row[0] is not None
                
                # Clean up
                await conn.execute("DROP TABLE test_migration")
                await conn.execute("DELETE FROM schema_migrations WHERE version = '001'")
