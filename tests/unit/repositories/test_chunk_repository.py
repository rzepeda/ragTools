"""Unit tests for ChunkRepository."""

import pytest
from uuid import uuid4
from unittest.mock import Mock, MagicMock, patch
from sqlalchemy.exc import SQLAlchemyError
import numpy as np

from rag_factory.repositories.chunk import ChunkRepository
from rag_factory.repositories.exceptions import (
    EntityNotFoundError,
    DatabaseConnectionError,
    InvalidQueryError
)
from rag_factory.database.models import Chunk


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
    session.execute = Mock()
    session.merge = Mock()
    return session


@pytest.fixture
def chunk_repo(mock_session):
    """Create a ChunkRepository with mock session."""
    return ChunkRepository(mock_session)


@pytest.fixture
def sample_chunk():
    """Create a sample chunk for testing."""
    chunk = Chunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        text="Sample chunk text",
        embedding=None,
        metadata_={}
    )
    return chunk


@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector."""
    return np.random.rand(1536).tolist()


class TestChunkRepositoryCreate:
    """Tests for chunk creation."""

    def test_create_chunk_success(self, chunk_repo, mock_session):
        """Test successfully creating a new chunk."""
        doc_id = uuid4()
        chunk = chunk_repo.create(
            document_id=doc_id,
            chunk_index=0,
            text="Test chunk"
        )

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
        assert chunk.document_id == doc_id
        assert chunk.chunk_index == 0
        assert chunk.text == "Test chunk"

    def test_create_chunk_with_embedding(self, chunk_repo, mock_session, sample_embedding):
        """Test creating chunk with embedding."""
        doc_id = uuid4()
        chunk = chunk_repo.create(
            document_id=doc_id,
            chunk_index=0,
            text="Test chunk",
            embedding=sample_embedding
        )

        assert chunk.embedding == sample_embedding

    def test_create_chunk_with_metadata(self, chunk_repo, mock_session):
        """Test creating chunk with metadata."""
        metadata = {"source": "page 1", "section": "intro"}
        chunk = chunk_repo.create(
            document_id=uuid4(),
            chunk_index=0,
            text="Test",
            metadata=metadata
        )

        assert chunk.metadata_ == metadata

    def test_create_chunk_database_error(self, chunk_repo, mock_session):
        """Test database error during creation."""
        mock_session.add.side_effect = SQLAlchemyError("Connection lost")

        with pytest.raises(DatabaseConnectionError):
            chunk_repo.create(uuid4(), 0, "Test")

        mock_session.rollback.assert_called_once()

    def test_bulk_create_chunks(self, chunk_repo, mock_session, sample_embedding):
        """Test bulk creating multiple chunks."""
        doc_id = uuid4()
        chunks_data = [
            {
                "document_id": doc_id,
                "chunk_index": i,
                "text": f"Chunk {i}",
                "embedding": sample_embedding
            }
            for i in range(3)
        ]

        created = chunk_repo.bulk_create(chunks_data)

        assert len(created) == 3
        mock_session.bulk_save_objects.assert_called_once()
        mock_session.flush.assert_called_once()

    def test_bulk_create_database_error(self, chunk_repo, mock_session):
        """Test database error during bulk create."""
        mock_session.bulk_save_objects.side_effect = SQLAlchemyError("Connection lost")
        chunks = [{"document_id": uuid4(), "chunk_index": 0, "text": "Test"}]

        with pytest.raises(DatabaseConnectionError):
            chunk_repo.bulk_create(chunks)

        mock_session.rollback.assert_called_once()


class TestChunkRepositoryRead:
    """Tests for reading chunks."""

    def test_get_by_id_found(self, chunk_repo, mock_session, sample_chunk):
        """Test retrieving chunk by ID when it exists."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = sample_chunk
        mock_session.query.return_value = mock_query

        result = chunk_repo.get_by_id(sample_chunk.chunk_id)

        assert result == sample_chunk

    def test_get_by_id_not_found(self, chunk_repo, mock_session):
        """Test retrieving non-existent chunk returns None."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_session.query.return_value = mock_query

        result = chunk_repo.get_by_id(uuid4())

        assert result is None

    def test_get_by_id_database_error(self, chunk_repo, mock_session):
        """Test database error during retrieval."""
        mock_session.query.side_effect = SQLAlchemyError("Connection lost")

        with pytest.raises(DatabaseConnectionError):
            chunk_repo.get_by_id(uuid4())

    def test_get_by_document(self, chunk_repo, mock_session):
        """Test retrieving all chunks for a document."""
        mock_chunks = [Mock() for _ in range(5)]
        mock_query = Mock()
        (mock_query.filter.return_value.order_by.return_value
         .offset.return_value.limit.return_value.all.return_value) = mock_chunks
        mock_session.query.return_value = mock_query

        doc_id = uuid4()
        results = chunk_repo.get_by_document(doc_id, skip=0, limit=10)

        assert len(results) == 5

    def test_count_by_document(self, chunk_repo, mock_session):
        """Test counting chunks for a document."""
        mock_query = Mock()
        mock_query.filter.return_value.scalar.return_value = 10
        mock_session.query.return_value = mock_query

        count = chunk_repo.count_by_document(uuid4())

        assert count == 10


class TestChunkRepositoryUpdate:
    """Tests for updating chunks."""

    def test_update_chunk_success(self, chunk_repo, mock_session, sample_chunk):
        """Test successfully updating chunk fields."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = sample_chunk
        mock_session.query.return_value = mock_query

        updated = chunk_repo.update(sample_chunk.chunk_id, text="Updated text")

        assert updated.text == "Updated text"
        mock_session.flush.assert_called_once()

    def test_update_chunk_not_found(self, chunk_repo, mock_session):
        """Test updating non-existent chunk raises error."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_session.query.return_value = mock_query

        with pytest.raises(EntityNotFoundError):
            chunk_repo.update(uuid4(), text="Test")

    def test_update_embedding(self, chunk_repo, mock_session, sample_chunk, sample_embedding):
        """Test updating chunk embedding."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = sample_chunk
        mock_session.query.return_value = mock_query

        updated = chunk_repo.update_embedding(sample_chunk.chunk_id, sample_embedding)

        assert updated.embedding == sample_embedding
        mock_session.flush.assert_called_once()

    def test_update_embedding_not_found(self, chunk_repo, mock_session, sample_embedding):
        """Test updating embedding for non-existent chunk."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_session.query.return_value = mock_query

        with pytest.raises(EntityNotFoundError):
            chunk_repo.update_embedding(uuid4(), sample_embedding)

    def test_bulk_update_embeddings(self, chunk_repo, mock_session, sample_embedding):
        """Test bulk updating embeddings."""
        chunk_ids = [uuid4() for _ in range(3)]
        updates = [(chunk_id, sample_embedding) for chunk_id in chunk_ids]

        mock_query = Mock()
        mock_query.filter.return_value.update.return_value = 1
        mock_session.query.return_value = mock_query

        count = chunk_repo.bulk_update_embeddings(updates)

        assert count == 3
        mock_session.flush.assert_called_once()

    def test_bulk_update_embeddings_database_error(self, chunk_repo, mock_session, sample_embedding):
        """Test database error during bulk update."""
        mock_session.query.side_effect = SQLAlchemyError("Connection lost")
        updates = [(uuid4(), sample_embedding)]

        with pytest.raises(DatabaseConnectionError):
            chunk_repo.bulk_update_embeddings(updates)

        mock_session.rollback.assert_called_once()


class TestChunkRepositoryDelete:
    """Tests for deleting chunks."""

    def test_delete_chunk_success(self, chunk_repo, mock_session, sample_chunk):
        """Test successfully deleting a chunk."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = sample_chunk
        mock_session.query.return_value = mock_query

        result = chunk_repo.delete(sample_chunk.chunk_id)

        assert result is True
        mock_session.delete.assert_called_once_with(sample_chunk)
        mock_session.flush.assert_called_once()

    def test_delete_chunk_not_found(self, chunk_repo, mock_session):
        """Test deleting non-existent chunk raises error."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_session.query.return_value = mock_query

        with pytest.raises(EntityNotFoundError):
            chunk_repo.delete(uuid4())

    def test_delete_by_document(self, chunk_repo, mock_session):
        """Test deleting all chunks for a document."""
        mock_query = Mock()
        mock_query.filter.return_value.delete.return_value = 5
        mock_session.query.return_value = mock_query

        count = chunk_repo.delete_by_document(uuid4())

        assert count == 5
        mock_session.flush.assert_called_once()

    def test_delete_database_error(self, chunk_repo, mock_session, sample_chunk):
        """Test database error during deletion."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = sample_chunk
        mock_session.query.return_value = mock_query
        mock_session.delete.side_effect = SQLAlchemyError("Connection lost")

        with pytest.raises(DatabaseConnectionError):
            chunk_repo.delete(sample_chunk.chunk_id)

        mock_session.rollback.assert_called_once()


class TestChunkRepositoryVectorSearch:
    """Tests for vector similarity search."""

    def test_search_similar_validation_empty_embedding(self, chunk_repo):
        """Test search with empty embedding raises error."""
        with pytest.raises(InvalidQueryError) as exc_info:
            chunk_repo.search_similar([], top_k=5)

        assert "cannot be empty" in str(exc_info.value)

    def test_search_similar_validation_invalid_top_k(self, chunk_repo, sample_embedding):
        """Test search with invalid top_k raises error."""
        with pytest.raises(InvalidQueryError) as exc_info:
            chunk_repo.search_similar(sample_embedding, top_k=0)

        assert "top_k must be at least 1" in str(exc_info.value)

    def test_search_similar_success(self, chunk_repo, mock_session, sample_embedding):
        """Test successful vector similarity search."""
        # Mock database results
        mock_rows = [
            (uuid4(), uuid4(), 0, "Chunk 1", sample_embedding, {}, None, None, 0.95),
            (uuid4(), uuid4(), 1, "Chunk 2", sample_embedding, {}, None, None, 0.85),
        ]
        mock_result = Mock()
        mock_result.fetchall.return_value = mock_rows
        mock_session.execute.return_value = mock_result

        # Mock merge to return chunks
        def merge_side_effect(chunk):
            return chunk
        mock_session.merge.side_effect = merge_side_effect

        results = chunk_repo.search_similar(sample_embedding, top_k=2)

        assert len(results) == 2
        assert results[0][1] == 0.95  # First chunk has higher similarity
        assert results[1][1] == 0.85
        mock_session.execute.assert_called_once()

    def test_search_similar_with_threshold(self, chunk_repo, mock_session, sample_embedding):
        """Test vector search with similarity threshold."""
        mock_rows = [
            (uuid4(), uuid4(), 0, "Chunk 1", sample_embedding, {}, None, None, 0.85),
        ]
        mock_result = Mock()
        mock_result.fetchall.return_value = mock_rows
        mock_session.execute.return_value = mock_result
        mock_session.merge.side_effect = lambda x: x

        results = chunk_repo.search_similar(sample_embedding, top_k=10, threshold=0.8)

        # Verify threshold was passed to query
        call_args = mock_session.execute.call_args
        assert call_args[0][1]["threshold"] == 0.8

    def test_search_similar_database_error(self, chunk_repo, mock_session, sample_embedding):
        """Test database error during vector search."""
        mock_session.execute.side_effect = SQLAlchemyError("Connection lost")

        with pytest.raises(DatabaseConnectionError):
            chunk_repo.search_similar(sample_embedding, top_k=5)

    def test_search_similar_with_filter_validation(self, chunk_repo, sample_embedding):
        """Test search with filter validates document_ids."""
        with pytest.raises(InvalidQueryError) as exc_info:
            chunk_repo.search_similar_with_filter(sample_embedding, top_k=5, document_ids=[])

        assert "document_ids cannot be empty" in str(exc_info.value)

    def test_search_similar_with_filter_success(self, chunk_repo, mock_session, sample_embedding):
        """Test vector search filtered by document IDs."""
        doc_id = uuid4()
        chunk_id = uuid4()
        mock_rows = [
            (chunk_id, doc_id, 0, "Chunk 1", sample_embedding, {}, None, None, 0.95),
        ]
        mock_result = Mock()
        mock_result.fetchall.return_value = mock_rows
        mock_session.execute.return_value = mock_result
        
        # Mock the query() call that retrieves the chunk by ID
        mock_chunk = Mock()
        mock_chunk.chunk_id = chunk_id
        mock_chunk.document_id = doc_id
        mock_chunk.text = "Chunk 1"
        mock_chunk.embedding = sample_embedding
        
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_chunk
        mock_session.query.return_value = mock_query

        results = chunk_repo.search_similar_with_filter(
            sample_embedding,
            top_k=5,
            document_ids=[doc_id]
        )

        assert len(results) == 1
        assert results[0][0].document_id == doc_id
        assert results[0][1] == 0.95  # Check similarity score

    def test_search_similar_with_metadata_success(self, chunk_repo, mock_session, sample_embedding):
        """Test vector search filtered by metadata."""
        metadata_filter = {"source": "page 1"}
        chunk_id = uuid4()
        doc_id = uuid4()
        mock_rows = [
            (chunk_id, doc_id, 0, "Chunk 1", sample_embedding,
             {"source": "page 1"}, None, None, 0.95),
        ]
        mock_result = Mock()
        mock_result.fetchall.return_value = mock_rows
        mock_session.execute.return_value = mock_result
        
        # Mock the query() call that retrieves the chunk by ID
        mock_chunk = Mock()
        mock_chunk.chunk_id = chunk_id
        mock_chunk.document_id = doc_id
        mock_chunk.text = "Chunk 1"
        mock_chunk.embedding = sample_embedding
        mock_chunk.metadata_ = {"source": "page 1"}
        
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_chunk
        mock_session.query.return_value = mock_query

        results = chunk_repo.search_similar_with_metadata(
            sample_embedding,
            top_k=5,
            metadata_filter=metadata_filter
        )

        assert len(results) == 1
        assert results[0][0].metadata_ == {"source": "page 1"}
        assert results[0][1] == 0.95  # Check similarity score
        # Verify metadata filter was included in query
        mock_session.execute.assert_called_once()


    def test_search_similar_with_metadata_validation(self, chunk_repo):
        """Test search with metadata filter validates parameters."""
        with pytest.raises(InvalidQueryError):
            chunk_repo.search_similar_with_metadata([], top_k=5, metadata_filter={})


class TestChunkRepositoryEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_create_chunk_without_embedding(self, chunk_repo, mock_session):
        """Test creating chunk without embedding is allowed."""
        chunk = chunk_repo.create(
            document_id=uuid4(),
            chunk_index=0,
            text="Test",
            embedding=None
        )

        assert chunk.embedding is None

    def test_update_with_metadata_attribute(self, chunk_repo, mock_session, sample_chunk):
        """Test updating metadata using special attribute name."""
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = sample_chunk
        mock_session.query.return_value = mock_query

        new_metadata = {"key": "value"}
        updated = chunk_repo.update(sample_chunk.chunk_id, metadata=new_metadata)

        assert updated.metadata_ == new_metadata
