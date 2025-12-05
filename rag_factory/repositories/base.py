"""Base repository with common CRUD operations.

This module defines the abstract base repository class that provides
common database operations and transaction management for all repositories.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional
from sqlalchemy.orm import Session
from contextlib import contextmanager

from .exceptions import RepositoryError


T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """Abstract base repository for database operations.

    This class provides a common interface for all repository implementations,
    including transaction management and basic CRUD operations. Concrete
    repositories should inherit from this class and implement the abstract methods.

    Type Parameters:
        T: The entity type this repository manages

    Attributes:
        session: SQLAlchemy session for database operations
    """

    def __init__(self, session: Session):
        """Initialize the repository with a database session.

        Args:
            session: SQLAlchemy session for database operations
        """
        self.session = session

    @abstractmethod
    def get_by_id(self, entity_id) -> Optional[T]:
        """Retrieve an entity by its unique identifier.

        Args:
            entity_id: Unique identifier of the entity

        Returns:
            The entity if found, None otherwise

        Raises:
            DatabaseConnectionError: If database query fails
        """
        pass

    @abstractmethod
    def create(self, *args, **kwargs) -> T:
        """Create a new entity.

        Args:
            *args: Positional arguments for entity creation
            **kwargs: Keyword arguments for entity creation

        Returns:
            The created entity

        Raises:
            DuplicateEntityError: If entity violates uniqueness constraints
            DatabaseConnectionError: If database operation fails
        """
        pass

    @abstractmethod
    def update(self, entity_id, **updates) -> T:
        """Update an existing entity.

        Args:
            entity_id: Unique identifier of the entity to update
            **updates: Fields to update

        Returns:
            The updated entity

        Raises:
            EntityNotFoundError: If entity does not exist
            DatabaseConnectionError: If database operation fails
        """
        pass

    @abstractmethod
    def delete(self, entity_id) -> bool:
        """Delete an entity by its identifier.

        Args:
            entity_id: Unique identifier of the entity to delete

        Returns:
            True if deletion was successful

        Raises:
            EntityNotFoundError: If entity does not exist
            DatabaseConnectionError: If database operation fails
        """
        pass

    def commit(self):
        """Commit the current transaction.

        This method commits all pending changes to the database.
        If the commit fails, the transaction is rolled back automatically.

        Raises:
            RepositoryError: If commit fails
        """
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise RepositoryError(f"Commit failed: {str(e)}")

    def rollback(self):
        """Rollback the current transaction.

        This method discards all pending changes since the last commit.
        """
        self.session.rollback()

    def flush(self):
        """Flush pending changes to the database without committing.

        This is useful for getting auto-generated IDs or ensuring
        constraint violations are detected before commit.

        Raises:
            RepositoryError: If flush fails
        """
        try:
            self.session.flush()
        except Exception as e:
            raise RepositoryError(f"Flush failed: {str(e)}")

    @contextmanager
    def transaction(self):
        """Context manager for transaction handling.

        This provides automatic commit on success and rollback on failure.

        Usage:
            with repository.transaction():
                repository.create(...)
                repository.update(...)
            # Auto-commits here if no exception

        Yields:
            None

        Raises:
            RepositoryError: If transaction fails
        """
        try:
            yield
            self.commit()
        except Exception as e:
            self.rollback()
            raise RepositoryError(f"Transaction failed: {str(e)}")
