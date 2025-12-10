"""Helper utilities for Alembic testing.

This module provides utility functions for working with Alembic migrations
in tests, making it easier to run migrations, check versions, and verify
migration state.
"""

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine
from pathlib import Path
from typing import Optional


def get_alembic_config(database_url: str) -> Config:
    """
    Get Alembic configuration for testing.

    Args:
        database_url: Database URL

    Returns:
        Alembic Config instance
    """
    project_root = Path(__file__).parent.parent.parent
    alembic_ini = project_root / "alembic.ini"

    config = Config(str(alembic_ini))
    config.set_main_option("sqlalchemy.url", database_url)

    return config


def get_current_revision(database_url: str) -> Optional[str]:
    """
    Get current migration revision from database.

    Args:
        database_url: Database URL

    Returns:
        Current revision ID or None
    """
    engine = create_engine(database_url)
    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        current = context.get_current_revision()
    engine.dispose()

    return current


def get_head_revision(config: Config) -> Optional[str]:
    """
    Get head revision from migration scripts.

    Args:
        config: Alembic configuration

    Returns:
        Head revision ID or None if no migrations exist
    """
    script = ScriptDirectory.from_config(config)
    return script.get_current_head()


def upgrade_to_head(database_url: str) -> None:
    """
    Upgrade database to head revision.

    Args:
        database_url: Database URL
    """
    config = get_alembic_config(database_url)
    command.upgrade(config, "head")


def downgrade_to_base(database_url: str) -> None:
    """
    Downgrade database to base (no migrations).

    Args:
        database_url: Database URL
    """
    config = get_alembic_config(database_url)
    command.downgrade(config, "base")


def verify_migration_applied(database_url: str, revision: str) -> bool:
    """
    Verify a specific migration is applied.

    Args:
        database_url: Database URL
        revision: Migration revision ID

    Returns:
        True if migration is applied
    """
    current = get_current_revision(database_url)

    if current == revision:
        return True

    # Check if current is descendant of revision
    config = get_alembic_config(database_url)
    script = ScriptDirectory.from_config(config)

    # Walk from current to base
    if current:
        for rev in script.walk_revisions(current, "base"):
            if rev.revision == revision:
                return True

    return False
