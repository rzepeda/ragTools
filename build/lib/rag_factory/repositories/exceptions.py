"""Custom exception classes for repository operations.

This module defines the exception hierarchy for handling errors in repository operations,
including entity not found, duplicate entities, and database connectivity issues.
"""

from typing import Any


class RepositoryError(Exception):
    """Base exception for all repository-related errors.

    All repository-specific exceptions should inherit from this class
    to allow for unified exception handling at the application level.
    """
    pass


class EntityNotFoundError(RepositoryError):
    """Raised when a requested entity does not exist in the database.

    Attributes:
        entity_type: The type/class name of the entity that was not found
        entity_id: The identifier that was used to search for the entity
    """

    def __init__(self, entity_type: str, entity_id: Any):
        """Initialize EntityNotFoundError.

        Args:
            entity_type: Type/class name of the entity (e.g., "Document", "Chunk")
            entity_id: The ID that was searched for
        """
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} with id {entity_id} not found")


class DuplicateEntityError(RepositoryError):
    """Raised when attempting to create an entity that violates a uniqueness constraint.

    Attributes:
        entity_type: The type/class name of the entity
        field: The field name that has the uniqueness constraint
        value: The duplicate value that caused the violation
    """

    def __init__(self, entity_type: str, field: str, value: Any):
        """Initialize DuplicateEntityError.

        Args:
            entity_type: Type/class name of the entity
            field: Field name with the uniqueness constraint
            value: The duplicate value
        """
        self.entity_type = entity_type
        self.field = field
        self.value = value
        super().__init__(f"{entity_type} with {field}={value} already exists")


class DatabaseConnectionError(RepositoryError):
    """Raised when database connection or query execution fails.

    This exception wraps lower-level SQLAlchemy exceptions to provide
    a cleaner interface for application code.
    """
    pass


class InvalidQueryError(RepositoryError):
    """Raised when query parameters are invalid or malformed.

    Examples include empty embedding vectors, invalid top_k values,
    or malformed filter criteria.
    """
    pass
