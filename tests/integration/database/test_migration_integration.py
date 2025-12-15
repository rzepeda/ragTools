"""Integration tests for Alembic migrations with real database."""

import pytest
from pathlib import Path
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text, inspect


@pytest.mark.integration
class TestMigrationIntegration:
    """Integration tests for migration system."""

    @pytest.fixture
    def alembic_config(self, test_db_url: str) -> Config:
        """Create Alembic configuration for testing."""
        project_root = Path(__file__).parent.parent.parent.parent
        alembic_ini = project_root / "alembic.ini"

        config = Config(str(alembic_ini))
        config.set_main_option("sqlalchemy.url", test_db_url)

        return config

    @pytest.mark.asyncio
    async def test_real_migration_execution(self, alembic_config: Config, test_db_url: str) -> None:
        """Test running migrations against real database."""
        # Start from clean state
        command.downgrade(alembic_config, "base")

        # Upgrade to head
        command.upgrade(alembic_config, "head")

        # Verify tables exist
        engine = create_engine(test_db_url)
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        assert "documents" in tables
        assert "chunks" in tables

        engine.dispose()

    @pytest.mark.asyncio
    async def test_migration_with_existing_data(self, alembic_config: Config, test_db_url: str) -> None:
        """Test migration with existing data in database."""
        # Start from clean state
        command.downgrade(alembic_config, "base")
        
        # Upgrade to initial schema
        command.upgrade(alembic_config, "001")

        # Insert test data with all required fields
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO documents (document_id, filename, source_path, content_hash, total_chunks, metadata, status)
                VALUES (gen_random_uuid(), 'test.txt', '/test', 'abc123', 0, '{}', 'pending')
            """))
            conn.commit()
        engine.dispose()

        # Upgrade to head (should preserve data)
        command.upgrade(alembic_config, "head")

        # Verify data still exists
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM documents"))
            count = result.scalar()
        engine.dispose()

        assert count == 1

    @pytest.mark.asyncio
    async def test_rollback_functionality(self, alembic_config: Config, test_db_url: str) -> None:
        """Test rolling back migrations."""
        # Start from clean state
        command.downgrade(alembic_config, "base")
        
        # Upgrade to head
        command.upgrade(alembic_config, "head")

        # Verify hierarchy columns exist (from migration 002)
        engine = create_engine(test_db_url)
        inspector = inspect(engine)
        
        # Check if chunks table exists
        tables = inspector.get_table_names()
        assert "chunks" in tables, "chunks table should exist after upgrade to head"
        
        columns = [col['name'] for col in inspector.get_columns("chunks")]
        assert "parent_chunk_id" in columns, "parent_chunk_id should exist in migration 002"

        # Downgrade to 001
        command.downgrade(alembic_config, "001")

        # Verify hierarchy columns removed
        inspector = inspect(engine)
        tables_after = inspector.get_table_names()
        
        # Table should still exist after downgrade to 001
        assert "chunks" in tables_after, "chunks table should still exist after downgrade to 001"
        
        columns_after = [col['name'] for col in inspector.get_columns("chunks")]
        assert "parent_chunk_id" not in columns_after, "parent_chunk_id should be removed after downgrade"

        engine.dispose()

    @pytest.mark.asyncio
    async def test_pgvector_extension_installed(self, alembic_config: Config, test_db_url: str) -> None:
        """Test that pgvector extension is installed by migrations."""
        # Upgrade to head
        command.upgrade(alembic_config, "head")

        # Check pgvector extension
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                )
            """))
            has_pgvector = result.scalar()
        engine.dispose()

        assert has_pgvector is True
