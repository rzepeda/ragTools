"""Integration tests for MigrationValidator with real database."""

import pytest
from pathlib import Path
from alembic import command
from alembic.config import Config

from rag_factory.services.database import PostgresqlDatabaseService
from rag_factory.services.database.migration_validator import (
    MigrationValidator,
    MigrationValidationError,
)


@pytest.mark.integration
class TestMigrationValidatorIntegration:
    """Integration tests for MigrationValidator with real Alembic migrations."""
    
    @pytest.fixture
    def alembic_config(self, test_db_url: str) -> Config:
        """Create Alembic configuration for testing."""
        project_root = Path(__file__).parent.parent.parent.parent
        alembic_ini = project_root / "alembic.ini"
        
        config = Config(str(alembic_ini))
        config.set_main_option("sqlalchemy.url", test_db_url)
        
        return config
    
    @pytest.fixture
    def db_service(self, test_db_url: str) -> PostgresqlDatabaseService:
        """Create database service for testing."""
        service = PostgresqlDatabaseService(
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_pass"
        )
        # Override connection string with test URL
        service._connection_string = test_db_url
        
        # Recreate engine with test URL
        from sqlalchemy import create_engine
        service.engine = create_engine(test_db_url)
        
        yield service
        
        # Cleanup
        service.close()
    
    @pytest.fixture
    def validator(self, db_service: PostgresqlDatabaseService) -> MigrationValidator:
        """Create MigrationValidator for testing."""
        project_root = Path(__file__).parent.parent.parent.parent
        alembic_ini = project_root / "alembic.ini"
        
        return MigrationValidator(db_service, str(alembic_ini))
    
    @pytest.mark.asyncio
    async def test_validate_with_no_migrations(
        self,
        validator: MigrationValidator,
        alembic_config: Config
    ) -> None:
        """Test validation when no migrations are applied."""
        # Start from clean state
        command.downgrade(alembic_config, "base")
        
        # Validate - should fail
        is_valid, missing = validator.validate(["001", "002"])
        
        assert is_valid is False
        assert "001" in missing
        assert "002" in missing
    
    @pytest.mark.asyncio
    async def test_validate_with_partial_migrations(
        self,
        validator: MigrationValidator,
        alembic_config: Config
    ) -> None:
        """Test validation when only some migrations are applied."""
        # Start from clean state
        command.downgrade(alembic_config, "base")
        
        # Apply only first migration
        command.upgrade(alembic_config, "001")
        
        # Validate - should fail for 002
        is_valid, missing = validator.validate(["001", "002"])
        
        assert is_valid is False
        assert "001" not in missing
        assert "002" in missing
    
    @pytest.mark.asyncio
    async def test_validate_with_all_migrations(
        self,
        validator: MigrationValidator,
        alembic_config: Config
    ) -> None:
        """Test validation when all required migrations are applied."""
        # Apply all migrations
        command.upgrade(alembic_config, "head")
        
        # Validate - should succeed
        is_valid, missing = validator.validate(["001", "002"])
        
        assert is_valid is True
        assert missing == []
    
    @pytest.mark.asyncio
    async def test_validate_or_raise_success(
        self,
        validator: MigrationValidator,
        alembic_config: Config
    ) -> None:
        """Test validate_or_raise when all migrations are applied."""
        # Apply all migrations
        command.upgrade(alembic_config, "head")
        
        # Should not raise
        validator.validate_or_raise(["001", "002"])
    
    @pytest.mark.asyncio
    async def test_validate_or_raise_failure(
        self,
        validator: MigrationValidator,
        alembic_config: Config
    ) -> None:
        """Test validate_or_raise when migrations are missing."""
        # Start from clean state
        command.downgrade(alembic_config, "base")
        
        # Should raise with helpful message
        with pytest.raises(MigrationValidationError) as exc_info:
            validator.validate_or_raise(["001", "002"])
        
        error = exc_info.value
        assert error.missing_revisions == ["001", "002"]
        assert "Required database migrations are not applied" in str(error)
        assert "alembic upgrade" in str(error)
    
    @pytest.mark.asyncio
    async def test_get_current_revision(
        self,
        validator: MigrationValidator,
        alembic_config: Config
    ) -> None:
        """Test getting current revision from database."""
        # Start from clean state
        command.downgrade(alembic_config, "base")
        
        # No migrations applied
        current = validator.get_current_revision()
        assert current is None
        
        # Apply first migration
        command.upgrade(alembic_config, "001")
        current = validator.get_current_revision()
        assert current == "001"
        
        # Apply all migrations
        command.upgrade(alembic_config, "head")
        current = validator.get_current_revision()
        assert current is not None
        assert current != "001"  # Should be at a later revision
    
    @pytest.mark.asyncio
    async def test_get_all_revisions(self, validator: MigrationValidator) -> None:
        """Test getting all available revisions."""
        all_revisions = validator.get_all_revisions()
        
        # Should have at least our two test migrations
        assert len(all_revisions) >= 2
        assert "001" in all_revisions
        assert "002" in all_revisions
        
        # Should be in chronological order
        idx_001 = all_revisions.index("001")
        idx_002 = all_revisions.index("002")
        assert idx_001 < idx_002
    
    @pytest.mark.asyncio
    async def test_is_at_head(
        self,
        validator: MigrationValidator,
        alembic_config: Config
    ) -> None:
        """Test checking if database is at head revision."""
        # Start from clean state
        command.downgrade(alembic_config, "base")
        
        # Not at head
        assert validator.is_at_head() is False
        
        # Apply only first migration
        command.upgrade(alembic_config, "001")
        assert validator.is_at_head() is False
        
        # Apply all migrations
        command.upgrade(alembic_config, "head")
        assert validator.is_at_head() is True
    
    @pytest.mark.asyncio
    async def test_error_message_details(
        self,
        validator: MigrationValidator,
        alembic_config: Config
    ) -> None:
        """Test that error messages include migration details."""
        # Start from clean state
        command.downgrade(alembic_config, "base")
        
        try:
            validator.validate_or_raise(["001", "002"])
            pytest.fail("Should have raised MigrationValidationError")
        except MigrationValidationError as e:
            error_msg = str(e)
            
            # Should include revision IDs
            assert "001" in error_msg
            assert "002" in error_msg
            
            # Should include upgrade command
            assert "alembic upgrade" in error_msg
            
            # Should include migration descriptions
            assert "Initial" in error_msg or "schema" in error_msg.lower()
    
    @pytest.mark.asyncio
    async def test_validate_single_migration(
        self,
        validator: MigrationValidator,
        alembic_config: Config
    ) -> None:
        """Test validating a single migration requirement."""
        # Apply all migrations
        command.upgrade(alembic_config, "head")
        
        # Validate just one migration
        is_valid, missing = validator.validate(["001"])
        
        assert is_valid is True
        assert missing == []
    
    @pytest.mark.asyncio
    async def test_validate_nonexistent_migration(
        self,
        validator: MigrationValidator,
        alembic_config: Config
    ) -> None:
        """Test validating a migration that doesn't exist."""
        # Apply all migrations
        command.upgrade(alembic_config, "head")
        
        # Validate nonexistent migration
        is_valid, missing = validator.validate(["999"])
        
        assert is_valid is False
        assert "999" in missing
    
    @pytest.mark.asyncio
    async def test_validate_after_downgrade(
        self,
        validator: MigrationValidator,
        alembic_config: Config
    ) -> None:
        """Test validation after downgrading migrations."""
        # Apply all migrations
        command.upgrade(alembic_config, "head")
        
        # Verify all are applied
        is_valid, _ = validator.validate(["001", "002"])
        assert is_valid is True
        
        # Downgrade to 001
        command.downgrade(alembic_config, "001")
        
        # Now 002 should be missing
        is_valid, missing = validator.validate(["001", "002"])
        assert is_valid is False
        assert "002" in missing
        assert "001" not in missing
    
    @pytest.mark.asyncio
    async def test_multiple_validators_same_database(
        self,
        db_service: PostgresqlDatabaseService,
        alembic_config: Config
    ) -> None:
        """Test multiple validators on the same database."""
        # Apply migrations
        command.upgrade(alembic_config, "head")
        
        # Create two validators
        project_root = Path(__file__).parent.parent.parent.parent
        alembic_ini = project_root / "alembic.ini"
        
        validator1 = MigrationValidator(db_service, str(alembic_ini))
        validator2 = MigrationValidator(db_service, str(alembic_ini))
        
        # Both should see the same state
        rev1 = validator1.get_current_revision()
        rev2 = validator2.get_current_revision()
        
        assert rev1 == rev2
        
        is_valid1, _ = validator1.validate(["001", "002"])
        is_valid2, _ = validator2.validate(["001", "002"])
        
        assert is_valid1 == is_valid2 == True


@pytest.mark.integration
class TestMigrationValidatorEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def db_service(self, test_db_url: str) -> PostgresqlDatabaseService:
        """Create database service for testing."""
        service = PostgresqlDatabaseService(
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_pass"
        )
        service._connection_string = test_db_url
        
        from sqlalchemy import create_engine
        service.engine = create_engine(test_db_url)
        
        yield service
        service.close()
    
    @pytest.mark.asyncio
    async def test_validator_with_auto_discovered_config(
        self,
        db_service: PostgresqlDatabaseService,
        test_db_url: str
    ) -> None:
        """Test validator with auto-discovered alembic.ini."""
        # Create validator without explicit config path
        validator = MigrationValidator(db_service)
        
        # Should still work
        all_revisions = validator.get_all_revisions()
        assert len(all_revisions) >= 2
    
    @pytest.mark.asyncio
    async def test_validate_empty_requirements(
        self,
        db_service: PostgresqlDatabaseService,
        test_db_url: str
    ) -> None:
        """Test validation with empty requirements list."""
        validator = MigrationValidator(db_service)
        
        # Empty requirements should always be valid
        is_valid, missing = validator.validate([])
        
        assert is_valid is True
        assert missing == []
