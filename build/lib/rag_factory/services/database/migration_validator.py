"""Migration validator for Alembic migrations.

This module provides validation functionality to ensure required Alembic migrations
are applied before running strategy pairs on a database.
"""

from typing import Optional
from pathlib import Path
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from sqlalchemy import text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError

from rag_factory.services.database.postgres import PostgresqlDatabaseService


class MigrationValidationError(Exception):
    """Raised when migration validation fails."""
    
    def __init__(self, message: str, missing_revisions: list[str]):
        """Initialize validation error.
        
        Args:
            message: Error message
            missing_revisions: List of missing revision IDs
        """
        super().__init__(message)
        self.missing_revisions = missing_revisions


class MigrationValidator:
    """Validates Alembic migrations against strategy pair requirements.
    
    This class checks if required database migrations are applied before
    allowing strategy pairs to run. It integrates with Alembic to query
    the migration history and validate required revisions.
    
    Example:
        >>> from rag_factory.services.database import PostgresqlDatabaseService
        >>> db_service = PostgresqlDatabaseService(...)
        >>> validator = MigrationValidator(db_service)
        >>> is_valid, missing = validator.validate(["001", "002"])
        >>> if not is_valid:
        ...     print(f"Missing migrations: {missing}")
    """
    
    def __init__(
        self,
        db_service: PostgresqlDatabaseService,
        alembic_config_path: Optional[str] = None
    ):
        """Initialize migration validator.
        
        Args:
            db_service: Database service instance
            alembic_config_path: Path to alembic.ini file. If None, searches
                                for alembic.ini in project root.
        """
        self.db_service = db_service
        self.engine = db_service.engine
        
        # Find alembic.ini if not provided
        if alembic_config_path is None:
            alembic_config_path = self._find_alembic_config()
        
        self.alembic_cfg = Config(alembic_config_path)
        self.script_dir = ScriptDirectory.from_config(self.alembic_cfg)
    
    def _find_alembic_config(self) -> str:
        """Find alembic.ini in project root.
        
        Returns:
            Path to alembic.ini
            
        Raises:
            FileNotFoundError: If alembic.ini cannot be found
        """
        # Start from this file and go up to project root
        current = Path(__file__).parent
        for _ in range(5):  # Search up to 5 levels
            alembic_ini = current / "alembic.ini"
            if alembic_ini.exists():
                return str(alembic_ini)
            current = current.parent
        
        raise FileNotFoundError(
            "Could not find alembic.ini. Please provide alembic_config_path."
        )
    
    def validate(self, required_revisions: list[str]) -> tuple[bool, list[str]]:
        """Check if required migrations are applied.
        
        Args:
            required_revisions: List of required revision IDs (e.g., ["001", "002"])
        
        Returns:
            Tuple of (is_valid, missing_revisions)
            - is_valid: True if all required revisions are applied
            - missing_revisions: List of missing revision IDs
            
        Example:
            >>> is_valid, missing = validator.validate(["001", "002"])
            >>> if not is_valid:
            ...     print(f"Please run: alembic upgrade {missing[0]}")
        """
        current_revision = self._get_current_revision()
        
        # If no migrations applied at all, all are missing
        if not current_revision:
            return (False, required_revisions)
        
        # Get all revisions between base and current
        applied_revisions = self._get_applied_revisions(current_revision)
        
        # Check which required revisions are missing
        missing = [r for r in required_revisions if r not in applied_revisions]
        
        return (len(missing) == 0, missing)
    
    def validate_or_raise(self, required_revisions: list[str]) -> None:
        """Validate migrations and raise exception if any are missing.
        
        Args:
            required_revisions: List of required revision IDs
            
        Raises:
            MigrationValidationError: If any required migrations are missing
            
        Example:
            >>> try:
            ...     validator.validate_or_raise(["001", "002"])
            ... except MigrationValidationError as e:
            ...     print(f"Missing: {e.missing_revisions}")
        """
        is_valid, missing = self.validate(required_revisions)
        
        if not is_valid:
            error_msg = self._build_error_message(missing)
            raise MigrationValidationError(error_msg, missing)
    
    def _build_error_message(self, missing_revisions: list[str]) -> str:
        """Build detailed error message for missing migrations.
        
        Args:
            missing_revisions: List of missing revision IDs
            
        Returns:
            Formatted error message with upgrade suggestions
        """
        if not missing_revisions:
            return "No missing migrations"
        
        lines = [
            "Required database migrations are not applied:",
            "",
            "Missing revisions:",
        ]
        
        # List each missing revision with details
        for rev_id in missing_revisions:
            try:
                revision = self.script_dir.get_revision(rev_id)
                if revision:
                    lines.append(f"  - {rev_id}: {revision.doc or 'No description'}")
                else:
                    lines.append(f"  - {rev_id}: (revision not found in scripts)")
            except Exception:
                lines.append(f"  - {rev_id}: (unable to get details)")
        
        lines.extend([
            "",
            "To apply missing migrations, run:",
            f"  alembic upgrade {missing_revisions[-1]}",
            "",
            "Or to apply all migrations:",
            "  alembic upgrade head",
        ])
        
        return "\n".join(lines)
    
    def _get_current_revision(self) -> Optional[str]:
        """Get current revision from alembic_version table.
        
        Returns:
            Current revision ID or None if no migrations applied
            
        Note:
            Returns None if alembic_version table doesn't exist
        """
        try:
            with self.engine.connect() as conn:
                # Check if alembic_version table exists
                inspector = inspect(conn)
                if "alembic_version" not in inspector.get_table_names():
                    return None
                
                # Get current revision
                context = MigrationContext.configure(conn)
                return context.get_current_revision()
        except ProgrammingError:
            # Table doesn't exist
            return None
    
    def _get_applied_revisions(self, current_revision: str) -> set[str]:
        """Get all revisions up to current.
        
        Args:
            current_revision: Current revision ID from database
            
        Returns:
            Set of all applied revision IDs
        """
        revisions = set()
        
        # Iterate from current back to base
        for rev in self.script_dir.iterate_revisions(current_revision, "base"):
            if rev.revision:
                revisions.add(rev.revision)
        
        return revisions
    
    def get_current_revision(self) -> Optional[str]:
        """Get the current database revision.
        
        Returns:
            Current revision ID or None if no migrations applied
            
        Example:
            >>> current = validator.get_current_revision()
            >>> print(f"Database is at revision: {current}")
        """
        return self._get_current_revision()
    
    def get_all_revisions(self) -> list[str]:
        """Get all available migration revisions.
        
        Returns:
            List of all revision IDs in chronological order
            
        Example:
            >>> all_revs = validator.get_all_revisions()
            >>> print(f"Available migrations: {all_revs}")
        """
        revisions = []
        
        # Walk all revisions from head to base
        head = self.script_dir.get_current_head()
        if head:
            for rev in self.script_dir.iterate_revisions(head, "base"):
                if rev.revision:
                    revisions.append(rev.revision)
        
        # Reverse to get chronological order (base to head)
        return list(reversed(revisions))
    
    def is_at_head(self) -> bool:
        """Check if database is at the latest migration.
        
        Returns:
            True if database is at head revision
            
        Example:
            >>> if not validator.is_at_head():
            ...     print("Database needs upgrade")
        """
        current = self._get_current_revision()
        head = self.script_dir.get_current_head()
        
        return current == head if current and head else False
