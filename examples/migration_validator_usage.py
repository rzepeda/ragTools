"""Example usage of MigrationValidator.

This example demonstrates how to use the MigrationValidator to ensure
required database migrations are applied before running strategy pairs.
"""

from rag_factory.services.database import (
    PostgresqlDatabaseService,
    MigrationValidator,
    MigrationValidationError,
)


def example_basic_validation():
    """Basic migration validation example."""
    print("=== Basic Migration Validation ===\n")
    
    # Create database service
    db_service = PostgresqlDatabaseService(
        host="localhost",
        port=5432,
        database="rag_db",
        user="rag_user",
        password="rag_pass"
    )
    
    try:
        # Create validator (auto-discovers alembic.ini)
        validator = MigrationValidator(db_service)
        
        # Check if required migrations are applied
        required_migrations = ["001", "002"]
        is_valid, missing = validator.validate(required_migrations)
        
        if is_valid:
            print("✅ All required migrations are applied!")
        else:
            print(f"❌ Missing migrations: {missing}")
            print(f"\nTo fix, run:")
            print(f"  alembic upgrade {missing[-1]}")
    
    finally:
        db_service.close()


def example_validate_or_raise():
    """Example using validate_or_raise for strict validation."""
    print("\n=== Strict Validation with Exception ===\n")
    
    db_service = PostgresqlDatabaseService(
        host="localhost",
        port=5432,
        database="rag_db",
        user="rag_user",
        password="rag_pass"
    )
    
    try:
        validator = MigrationValidator(db_service)
        
        # This will raise MigrationValidationError if migrations are missing
        try:
            validator.validate_or_raise(["001", "002"])
            print("✅ All migrations validated successfully!")
            
        except MigrationValidationError as e:
            print(f"❌ Migration validation failed!")
            print(f"\nError details:")
            print(str(e))
            print(f"\nMissing revisions: {e.missing_revisions}")
    
    finally:
        db_service.close()


def example_check_current_state():
    """Example checking current migration state."""
    print("\n=== Check Current Migration State ===\n")
    
    db_service = PostgresqlDatabaseService(
        host="localhost",
        port=5432,
        database="rag_db",
        user="rag_user",
        password="rag_pass"
    )
    
    try:
        validator = MigrationValidator(db_service)
        
        # Get current revision
        current = validator.get_current_revision()
        print(f"Current database revision: {current or 'No migrations applied'}")
        
        # Get all available revisions
        all_revisions = validator.get_all_revisions()
        print(f"\nAvailable migrations: {all_revisions}")
        
        # Check if at head
        at_head = validator.is_at_head()
        if at_head:
            print("\n✅ Database is at the latest migration")
        else:
            print("\n⚠️  Database is not at the latest migration")
            print("   Run 'alembic upgrade head' to update")
    
    finally:
        db_service.close()


def example_strategy_pair_validation():
    """Example validating migrations for a strategy pair."""
    print("\n=== Strategy Pair Migration Validation ===\n")
    
    # Strategy pair configuration
    strategy_config = {
        "name": "hybrid_rag",
        "required_migrations": ["001", "002"],  # Requires base schema + hierarchy
        "database": {
            "host": "localhost",
            "port": 5432,
            "database": "rag_db",
            "user": "rag_user",
            "password": "rag_pass"
        }
    }
    
    # Create database service
    db_service = PostgresqlDatabaseService(**strategy_config["database"])
    
    try:
        # Validate migrations before initializing strategy
        validator = MigrationValidator(db_service)
        
        print(f"Validating migrations for strategy: {strategy_config['name']}")
        print(f"Required migrations: {strategy_config['required_migrations']}")
        
        try:
            validator.validate_or_raise(strategy_config["required_migrations"])
            print("\n✅ Migration validation passed!")
            print("   Strategy can be safely initialized.")
            
            # Now safe to initialize strategy...
            # strategy = initialize_strategy(db_service, strategy_config)
            
        except MigrationValidationError as e:
            print("\n❌ Migration validation failed!")
            print("   Strategy cannot be initialized until migrations are applied.")
            print(f"\n{e}")
            
            # Don't initialize strategy
            return
    
    finally:
        db_service.close()


def example_with_explicit_config():
    """Example with explicit alembic.ini path."""
    print("\n=== Validation with Explicit Config Path ===\n")
    
    db_service = PostgresqlDatabaseService(
        host="localhost",
        port=5432,
        database="rag_db",
        user="rag_user",
        password="rag_pass"
    )
    
    try:
        # Specify explicit path to alembic.ini
        validator = MigrationValidator(
            db_service,
            alembic_config_path="/path/to/alembic.ini"
        )
        
        is_valid, missing = validator.validate(["001"])
        
        if is_valid:
            print("✅ Migration 001 is applied")
        else:
            print(f"❌ Migration 001 is missing")
    
    finally:
        db_service.close()


def example_progressive_validation():
    """Example validating migrations progressively."""
    print("\n=== Progressive Migration Validation ===\n")
    
    db_service = PostgresqlDatabaseService(
        host="localhost",
        port=5432,
        database="rag_db",
        user="rag_user",
        password="rag_pass"
    )
    
    try:
        validator = MigrationValidator(db_service)
        
        # Check migrations one by one
        migrations_to_check = ["001", "002"]
        
        for migration in migrations_to_check:
            is_valid, missing = validator.validate([migration])
            
            if is_valid:
                print(f"✅ Migration {migration}: Applied")
            else:
                print(f"❌ Migration {migration}: Missing")
                print(f"   Run: alembic upgrade {migration}")
                break  # Stop at first missing migration
    
    finally:
        db_service.close()


if __name__ == "__main__":
    """Run all examples.
    
    Note: These examples require a running PostgreSQL database.
    Update the connection parameters to match your setup.
    """
    
    print("MigrationValidator Usage Examples")
    print("=" * 50)
    
    try:
        example_basic_validation()
        example_validate_or_raise()
        example_check_current_state()
        example_strategy_pair_validation()
        example_with_explicit_config()
        example_progressive_validation()
        
        print("\n" + "=" * 50)
        print("Examples completed!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nMake sure PostgreSQL is running and connection parameters are correct.")
