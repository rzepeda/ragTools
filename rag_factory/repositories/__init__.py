"""Repositories package for RAG Factory.

This package contains repository implementations for database operations,
following the Repository pattern for clean separation between business
logic and data access.

Available repositories:
- DocumentRepository: CRUD operations for documents
- ChunkRepository: CRUD operations and vector search for chunks
- BaseRepository: Abstract base class for all repositories

Available exceptions:
- RepositoryError: Base exception for all repository errors
- EntityNotFoundError: Raised when entity doesn't exist
- DuplicateEntityError: Raised when uniqueness constraint violated
- DatabaseConnectionError: Raised when database operations fail
- InvalidQueryError: Raised when query parameters are invalid
"""

from .base import BaseRepository
from .document import DocumentRepository
from .chunk import ChunkRepository
from .exceptions import (
    RepositoryError,
    EntityNotFoundError,
    DuplicateEntityError,
    DatabaseConnectionError,
    InvalidQueryError
)

__all__ = [
    # Base classes
    "BaseRepository",

    # Repository implementations
    "DocumentRepository",
    "ChunkRepository",

    # Exceptions
    "RepositoryError",
    "EntityNotFoundError",
    "DuplicateEntityError",
    "DatabaseConnectionError",
    "InvalidQueryError",
]
