"""Unit tests for Alembic database migrations."""

import pytest
from pathlib import Path
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect


class TestAlembicMigrations:
    """Test suite for Alembic migration system."""

    @pytest.fixture
    def alembic_config(self, test_db_url: str) -> Config:
        """
        Create Alembic configuration for testing.

        Args:
            test_db_url: Test database URL from fixture

        Returns:
            Alembic Config instance
        """
        # Get path to alembic.ini
        project_root = Path(__file__).parent.parent.parent.parent
        alembic_ini = project_root / "alembic.ini"

        # Create config
        config = Config(str(alembic_ini))
        config.set_main_option("sqlalchemy.url", test_db_url)

        return config

    def test_migration_upgrade_to_head(self, alembic_config: Config, test_db_url: str) -> None:
        """Test upgrading migrations to head."""
        # Upgrade to head
        command.upgrade(alembic_config, "head")

        # Verify we're at head
        script = ScriptDirectory.from_config(alembic_config)
        head_revision = script.get_current_head()

        # Check database version
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            current = context.get_current_revision()

        assert current == head_revision, f"Expected {head_revision}, got {current}"
        engine.dispose()

    def test_migration_downgrade(self, alembic_config: Config, test_db_url: str) -> None:
        """Test downgrading migrations."""
        # Upgrade to head first
        command.upgrade(alembic_config, "head")

        # Get current version
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            before_downgrade = context.get_current_revision()
        engine.dispose()

        # Downgrade one version
        command.downgrade(alembic_config, "-1")

        # Verify version changed
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            after_downgrade = context.get_current_revision()
        engine.dispose()

        assert after_downgrade != before_downgrade

    def test_migration_history(self, alembic_config: Config) -> None:
        """Test retrieving migration history."""
        script = ScriptDirectory.from_config(alembic_config)

        # Get all revisions
        revisions = list(script.walk_revisions())

        # Should have at least 2 migrations (001 and 002)
        assert len(revisions) >= 2

        # Verify revision IDs (filter out None values for type safety)
        revision_ids = [rev.revision for rev in revisions if rev.revision is not None]
        assert "001" in revision_ids
        assert "002" in revision_ids

    def test_migration_idempotency(self, alembic_config: Config, test_db_url: str) -> None:
        """Test that running migrations twice doesn't cause errors."""
        # Upgrade to head
        command.upgrade(alembic_config, "head")

        # Get version
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            version_1 = context.get_current_revision()
        engine.dispose()

        # Upgrade again (should be no-op)
        command.upgrade(alembic_config, "head")

        # Verify version unchanged
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            version_2 = context.get_current_revision()
        engine.dispose()

        assert version_1 == version_2

    def test_get_current_version(self, alembic_config: Config, test_db_url: str) -> None:
        """Test retrieving current schema version."""
        # Upgrade to head
        command.upgrade(alembic_config, "head")

        # Get current version
        engine = create_engine(test_db_url)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            current = context.get_current_revision()
        engine.dispose()

        assert current is not None
        assert len(current) > 0

    def test_migration_creates_tables(self, alembic_config: Config, test_db_url: str) -> None:
        """Test that migrations create expected tables."""
        # Upgrade to head
        command.upgrade(alembic_config, "head")

        # Check tables exist
        engine = create_engine(test_db_url)
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        # Should have documents and chunks tables
        assert "documents" in tables
        assert "chunks" in tables
        assert "alembic_version" in tables

        engine.dispose()

    def test_migration_creates_indexes(self, alembic_config: Config, test_db_url: str) -> None:
        """Test that migrations create expected indexes."""
        # Upgrade to head
        command.upgrade(alembic_config, "head")

        # Check indexes exist
        engine = create_engine(test_db_url)
        inspector = inspect(engine)

        # Check chunks table indexes
        indexes = inspector.get_indexes("chunks")
        # Filter out None names for type safety
        index_names: list[str] = []
        for idx in indexes:
            name = idx.get('name')
            if name is not None:
                index_names.append(name)

        # Should have vector index
        assert any("embedding" in name for name in index_names)

        engine.dispose()
