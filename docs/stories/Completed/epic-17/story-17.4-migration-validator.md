# Story 17.4: Implement Migration Validator (Alembic Integration)

**As a** system  
**I want** to validate that required Alembic migrations are applied  
**So that** strategy pairs only run on properly configured databases

## Acceptance Criteria
- Query `alembic_version` table from Epic 16's Alembic setup
- Check if required revision IDs are in migration history
- Provide clear error messages listing missing migrations
- Suggest `alembic upgrade` commands for missing revisions
- Handle case where alembic_version table doesn't exist
- Integration tests with test Alembic migrations

## Implementation
```python
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
import sqlalchemy as sa
from rag_factory.services.database import DatabaseService

class MigrationValidator:
    """Validates Alembic migrations against strategy pair requirements"""
    
    def __init__(self, db_service: DatabaseService, alembic_config_path: str = "alembic.ini"):
        self.db = db_service
        self.alembic_cfg = Config(alembic_config_path)
        self.script_dir = ScriptDirectory.from_config(self.alembic_cfg)
    
    def validate(self, required_revisions: list[str]) -> tuple[bool, list[str]]:
        """
        Check if required migrations are applied.
        
        Returns:
            (is_valid, missing_revisions)
        """
        current_revision = self._get_current_revision()
        
        if not current_revision:
            return (False, required_revisions)
        
        # Get all revisions between base and current
        applied_revisions = self._get_applied_revisions(current_revision)
        
        # Check which required revisions are missing
        missing = [r for r in required_revisions if r not in applied_revisions]
        
        return (len(missing) == 0, missing)
    
    def _get_current_revision(self) -> str:
        """Get current revision from alembic_version table"""
        with self.db.engine.connect() as conn:
            context = MigrationContext.configure(conn)
            return context.get_current_revision()
    
    def _get_applied_revisions(self, current_revision: str) -> set[str]:
        """Get all revisions up to current"""
        revisions = set()
        for rev in self.script_dir.iterate_revisions(current_revision, "base"):
            revisions.add(rev.revision)
        return revisions
```

## Story Points
5
