"""Unit tests for repository exceptions."""

import pytest
from rag_factory.repositories.exceptions import (
    RepositoryError,
    EntityNotFoundError,
    DuplicateEntityError,
    DatabaseConnectionError,
    InvalidQueryError
)


class TestRepositoryError:
    """Tests for base RepositoryError exception."""

    def test_repository_error_message(self):
        """Test RepositoryError with custom message."""
        error = RepositoryError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_repository_error_inheritance(self):
        """Test that RepositoryError inherits from Exception."""
        error = RepositoryError("test")
        assert isinstance(error, Exception)


class TestEntityNotFoundError:
    """Tests for EntityNotFoundError exception."""

    def test_entity_not_found_attributes(self):
        """Test EntityNotFoundError stores entity type and ID."""
        error = EntityNotFoundError("Document", "123-456")
        assert error.entity_type == "Document"
        assert error.entity_id == "123-456"

    def test_entity_not_found_message(self):
        """Test EntityNotFoundError generates correct message."""
        error = EntityNotFoundError("Chunk", "abc-def")
        assert str(error) == "Chunk with id abc-def not found"

    def test_entity_not_found_inheritance(self):
        """Test EntityNotFoundError inherits from RepositoryError."""
        error = EntityNotFoundError("Document", "123")
        assert isinstance(error, RepositoryError)


class TestDuplicateEntityError:
    """Tests for DuplicateEntityError exception."""

    def test_duplicate_entity_attributes(self):
        """Test DuplicateEntityError stores entity details."""
        error = DuplicateEntityError("Document", "content_hash", "abc123")
        assert error.entity_type == "Document"
        assert error.field == "content_hash"
        assert error.value == "abc123"

    def test_duplicate_entity_message(self):
        """Test DuplicateEntityError generates correct message."""
        error = DuplicateEntityError("Document", "content_hash", "xyz789")
        assert str(error) == "Document with content_hash=xyz789 already exists"

    def test_duplicate_entity_inheritance(self):
        """Test DuplicateEntityError inherits from RepositoryError."""
        error = DuplicateEntityError("Document", "hash", "123")
        assert isinstance(error, RepositoryError)


class TestDatabaseConnectionError:
    """Tests for DatabaseConnectionError exception."""

    def test_database_connection_error_message(self):
        """Test DatabaseConnectionError with custom message."""
        error = DatabaseConnectionError("Connection timeout")
        assert str(error) == "Connection timeout"

    def test_database_connection_error_inheritance(self):
        """Test DatabaseConnectionError inherits from RepositoryError."""
        error = DatabaseConnectionError("test")
        assert isinstance(error, RepositoryError)


class TestInvalidQueryError:
    """Tests for InvalidQueryError exception."""

    def test_invalid_query_error_message(self):
        """Test InvalidQueryError with custom message."""
        error = InvalidQueryError("Embedding cannot be empty")
        assert str(error) == "Embedding cannot be empty"

    def test_invalid_query_error_inheritance(self):
        """Test InvalidQueryError inherits from RepositoryError."""
        error = InvalidQueryError("test")
        assert isinstance(error, RepositoryError)
