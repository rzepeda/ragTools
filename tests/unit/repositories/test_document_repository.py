"""Unit tests for DocumentRepository."""

import pytest
from uuid import uuid4
from unittest.mock import Mock, MagicMock
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from rag_factory.repositories.document import DocumentRepository
from rag_factory.repositories.exceptions import (
    EntityNotFoundError,
    DuplicateEntityError,
    DatabaseConnectionError
)
from rag_factory.database.models import Document


@pytest.fixture
def mock_session():
    """Create a mock SQLAlchemy session."""
    session = Mock()
    session.query = Mock()
    session.add = Mock()
    session.delete = Mock()
    session.flush = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    return session


@pytest.fixture
def doc_repo(mock_session):
    """Create a DocumentRepository with mock session."""
    return DocumentRepository(mock_session)


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    doc = Document(
        document_id=uuid4(),
        filename="test.txt",
        source_path="/path/test.txt",
        content_hash="abc123",
        metadata_={},
        status="pending"
    )
    return doc


class TestDocumentRepositoryCreate:
    """Tests for document creation."""

    def test_create_document_success(self, doc_repo, mock_session):
        """Test successfully creating a new document."""
        # Mock that no duplicate exists
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_session.query.return_value = mock_query

        # Create document
        doc = doc_repo.create(
            filename="test.txt",
            source_path="/path/test.txt",
            content_hash="hash123"
        )

        # Verify document was added to session
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

        # Verify document attributes
        assert doc.filename == "test.txt"
        assert doc.source_path == "/path/test.txt"
        assert doc.content_hash == "hash123"
        assert doc.status == "pending"

    def test_create_document_with_metadata(self, doc_repo, mock_session):
        """Test creating document with custom metadata."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_session.query.return_value = mock_query

        metadata = {"author": "John Doe", "category": "research"}
        doc = doc_repo.create(
            filename="test.txt",
            source_path="/path/test.txt",
            content_hash="hash123",
            metadata=metadata
        )

        assert doc.metadata_ == metadata

    def test_create_document_duplicate_raises_error(self, doc_repo, mock_session, sample_document):
        """Test creating duplicate document raises DuplicateEntityError."""
        # Mock that duplicate exists
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = sample_document
        mock_session.query.return_value = mock_query

        with pytest.raises(DuplicateEntityError) as exc_info:
            doc_repo.create("test.txt", "/path/test.txt", "abc123")

        assert exc_info.value.entity_type == "Document"
        assert exc_info.value.field == "content_hash"
        assert exc_info.value.value == "abc123"

    def test_create_document_database_error(self, doc_repo, mock_session):
        """Test database error during creation."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_session.query.return_value = mock_query
        mock_session.add.side_effect = SQLAlchemyError("Connection lost")

        with pytest.raises(DatabaseConnectionError):
            doc_repo.create("test.txt", "/path/test.txt", "hash123")

        mock_session.rollback.assert_called_once()


class TestDocumentRepositoryRead:
    """Tests for reading documents."""

    def test_get_by_id_found(self, doc_repo, mock_session, sample_document):
        """Test retrieving document by ID when it exists."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = sample_document
        mock_session.query.return_value = mock_query

        result = doc_repo.get_by_id(sample_document.document_id)

        assert result == sample_document
        mock_session.query.assert_called_with(Document)

    def test_get_by_id_not_found(self, doc_repo, mock_session):
        """Test retrieving non-existent document returns None."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_session.query.return_value = mock_query

        result = doc_repo.get_by_id(uuid4())

        assert result is None

    def test_get_by_id_database_error(self, doc_repo, mock_session):
        """Test database error during retrieval."""
        mock_session.query.side_effect = SQLAlchemyError("Connection lost")

        with pytest.raises(DatabaseConnectionError):
            doc_repo.get_by_id(uuid4())

    def test_get_by_content_hash_found(self, doc_repo, mock_session, sample_document):
        """Test retrieving document by content hash."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = sample_document
        mock_session.query.return_value = mock_query

        result = doc_repo.get_by_content_hash("abc123")

        assert result == sample_document

    def test_list_all_with_pagination(self, doc_repo, mock_session):
        """Test listing documents with pagination."""
        mock_docs = [Mock() for _ in range(5)]
        mock_query = Mock()
        mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = mock_docs
        mock_session.query.return_value = mock_query

        results = doc_repo.list_all(skip=10, limit=5)

        assert len(results) == 5
        mock_query.order_by.return_value.offset.assert_called_with(10)
        mock_query.order_by.return_value.offset.return_value.limit.assert_called_with(5)

    def test_count_documents(self, doc_repo, mock_session):
        """Test counting total documents."""
        mock_query = Mock()
        mock_query.scalar.return_value = 42
        mock_session.query.return_value = mock_query

        count = doc_repo.count()

        assert count == 42

    def test_get_by_status(self, doc_repo, mock_session):
        """Test filtering documents by status."""
        mock_docs = [Mock(), Mock()]
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = mock_docs
        mock_session.query.return_value = mock_query

        results = doc_repo.get_by_status("completed")

        assert len(results) == 2


class TestDocumentRepositoryUpdate:
    """Tests for updating documents."""

    def test_update_document_success(self, doc_repo, mock_session, sample_document):
        """Test successfully updating document fields."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = sample_document
        mock_session.query.return_value = mock_query

        updated = doc_repo.update(sample_document.document_id, filename="updated.txt")

        assert updated.filename == "updated.txt"
        mock_session.flush.assert_called_once()

    def test_update_document_not_found(self, doc_repo, mock_session):
        """Test updating non-existent document raises error."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_session.query.return_value = mock_query

        with pytest.raises(EntityNotFoundError):
            doc_repo.update(uuid4(), filename="test.txt")

    def test_update_status(self, doc_repo, mock_session, sample_document):
        """Test updating document status."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = sample_document
        mock_session.query.return_value = mock_query

        updated = doc_repo.update_status(sample_document.document_id, "completed")

        assert updated.status == "completed"

    def test_update_metadata(self, doc_repo, mock_session, sample_document):
        """Test updating document metadata."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = sample_document
        mock_session.query.return_value = mock_query

        new_metadata = {"key": "value"}
        updated = doc_repo.update(sample_document.document_id, metadata=new_metadata)

        assert updated.metadata_ == new_metadata

    def test_update_database_error(self, doc_repo, mock_session, sample_document):
        """Test database error during update."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = sample_document
        mock_session.query.return_value = mock_query
        mock_session.flush.side_effect = SQLAlchemyError("Connection lost")

        with pytest.raises(DatabaseConnectionError):
            doc_repo.update(sample_document.document_id, filename="new.txt")

        mock_session.rollback.assert_called_once()


class TestDocumentRepositoryDelete:
    """Tests for deleting documents."""

    def test_delete_document_success(self, doc_repo, mock_session, sample_document):
        """Test successfully deleting a document."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = sample_document
        mock_session.query.return_value = mock_query

        result = doc_repo.delete(sample_document.document_id)

        assert result is True
        mock_session.delete.assert_called_once_with(sample_document)
        mock_session.flush.assert_called_once()

    def test_delete_document_not_found(self, doc_repo, mock_session):
        """Test deleting non-existent document raises error."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_session.query.return_value = mock_query

        with pytest.raises(EntityNotFoundError):
            doc_repo.delete(uuid4())

    def test_delete_database_error(self, doc_repo, mock_session, sample_document):
        """Test database error during deletion."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = sample_document
        mock_session.query.return_value = mock_query
        mock_session.delete.side_effect = SQLAlchemyError("Connection lost")

        with pytest.raises(DatabaseConnectionError):
            doc_repo.delete(sample_document.document_id)

        mock_session.rollback.assert_called_once()

    def test_bulk_delete_documents(self, doc_repo, mock_session):
        """Test bulk deleting multiple documents."""
        doc_ids = [uuid4(), uuid4(), uuid4()]
        mock_query = Mock()
        mock_query.filter.return_value.delete.return_value = 3
        mock_session.query.return_value = mock_query

        count = doc_repo.bulk_delete(doc_ids)

        assert count == 3
        mock_session.flush.assert_called_once()


class TestDocumentRepositoryBulkOperations:
    """Tests for bulk operations."""

    def test_bulk_create_documents(self, doc_repo, mock_session):
        """Test bulk creating multiple documents."""
        documents = [
            {"filename": f"test{i}.txt", "source_path": f"/path/test{i}.txt",
             "content_hash": f"hash{i}"}
            for i in range(3)
        ]

        created = doc_repo.bulk_create(documents)

        assert len(created) == 3
        mock_session.bulk_save_objects.assert_called_once()
        mock_session.flush.assert_called_once()

    def test_bulk_create_database_error(self, doc_repo, mock_session):
        """Test database error during bulk create."""
        mock_session.bulk_save_objects.side_effect = SQLAlchemyError("Connection lost")
        documents = [{"filename": "test.txt", "source_path": "/path", "content_hash": "hash"}]

        with pytest.raises(DatabaseConnectionError):
            doc_repo.bulk_create(documents)

        mock_session.rollback.assert_called_once()


class TestDocumentRepositoryTransactions:
    """Tests for transaction management."""

    def test_commit_success(self, doc_repo, mock_session):
        """Test successful commit."""
        doc_repo.commit()
        mock_session.commit.assert_called_once()

    def test_commit_failure_triggers_rollback(self, doc_repo, mock_session):
        """Test commit failure triggers rollback."""
        from rag_factory.repositories.exceptions import RepositoryError
        mock_session.commit.side_effect = Exception("Commit failed")

        with pytest.raises(RepositoryError):
            doc_repo.commit()

        mock_session.rollback.assert_called_once()

    def test_rollback(self, doc_repo, mock_session):
        """Test rollback."""
        doc_repo.rollback()
        mock_session.rollback.assert_called_once()
