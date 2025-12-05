"""Unit tests for database models."""

import uuid
import pytest
from datetime import datetime
import numpy as np

from rag_factory.database.models import Document, Chunk, Base
from sqlalchemy import inspect


class TestDocumentModel:
    """Test suite for Document model."""

    def test_document_model_has_required_columns(self):
        """Test Document model has all required columns."""
        inspector = inspect(Document)
        columns = {col.name for col in inspector.columns}

        required_columns = {
            "document_id", "filename", "source_path", "content_hash",
            "total_chunks", "metadata", "status", "created_at", "updated_at"
        }

        assert required_columns.issubset(columns), \
            f"Missing columns: {required_columns - columns}"

    def test_document_creation(self):
        """Test Document instance can be created."""
        doc = Document(
            filename="test.txt",
            source_path="/path/to/test.txt",
            content_hash="abc123",
            total_chunks=5
        )

        assert doc.filename == "test.txt"
        assert doc.source_path == "/path/to/test.txt"
        assert doc.content_hash == "abc123"
        assert doc.total_chunks == 5
        assert doc.document_id is not None
        assert isinstance(doc.document_id, uuid.UUID)

    def test_document_default_values(self):
        """Test Document has correct default values."""
        doc = Document(
            filename="test.txt",
            source_path="/path/test.txt",
            content_hash="abc"
        )

        assert doc.total_chunks == 0
        assert doc.metadata == {}
        assert doc.status == "pending"
        assert doc.document_id is not None

    def test_document_metadata_accepts_json(self):
        """Test JSONB metadata field accepts arbitrary JSON."""
        metadata = {
            "custom_field": "value",
            "nested": {"key": "val"},
            "array": [1, 2, 3],
            "number": 42
        }

        doc = Document(
            filename="test.txt",
            source_path="/path/test.txt",
            content_hash="abc",
            metadata=metadata
        )

        assert doc.metadata_["custom_field"] == "value"
        assert doc.metadata_["nested"]["key"] == "val"
        assert doc.metadata_["array"] == [1, 2, 3]
        assert doc.metadata_["number"] == 42

    def test_document_status_values(self):
        """Test Document status can be set to different values."""
        statuses = ["pending", "processing", "completed", "failed"]

        for status in statuses:
            doc = Document(
                filename="test.txt",
                source_path="/path/test.txt",
                content_hash="abc",
                status=status
            )
            assert doc.status == status

    def test_document_repr(self):
        """Test Document string representation."""
        doc = Document(
            filename="test.txt",
            source_path="/path/test.txt",
            content_hash="abc",
            total_chunks=5,
            status="completed"
        )

        repr_str = repr(doc)

        assert "Document" in repr_str
        assert "test.txt" in repr_str
        assert "completed" in repr_str
        assert "5" in repr_str

    def test_document_persistence(self, db_session):
        """Test Document can be persisted to database."""
        doc = Document(
            filename="test.txt",
            source_path="/path/to/test.txt",
            content_hash="abc123",
            total_chunks=5
        )

        db_session.add(doc)
        db_session.flush()

        # Retrieve from database
        retrieved = db_session.query(Document).filter_by(
            filename="test.txt"
        ).first()

        assert retrieved is not None
        assert retrieved.filename == "test.txt"
        assert retrieved.total_chunks == 5


class TestChunkModel:
    """Test suite for Chunk model."""

    def test_chunk_model_has_required_columns(self):
        """Test Chunk model has all required columns."""
        inspector = inspect(Chunk)
        columns = {col.name for col in inspector.columns}

        required_columns = {
            "chunk_id", "document_id", "chunk_index", "text",
            "embedding", "metadata", "created_at", "updated_at"
        }

        assert required_columns.issubset(columns), \
            f"Missing columns: {required_columns - columns}"

    def test_chunk_creation(self):
        """Test Chunk instance can be created."""
        doc_id = uuid.uuid4()

        chunk = Chunk(
            document_id=doc_id,
            chunk_index=0,
            text="Sample text"
        )

        assert chunk.document_id == doc_id
        assert chunk.chunk_index == 0
        assert chunk.text == "Sample text"
        assert chunk.chunk_id is not None
        assert isinstance(chunk.chunk_id, uuid.UUID)

    def test_chunk_creation_with_embedding(self):
        """Test Chunk can be created with vector embedding."""
        doc_id = uuid.uuid4()
        embedding = [0.1] * 1536  # 1536-dimensional vector

        chunk = Chunk(
            document_id=doc_id,
            chunk_index=0,
            text="Sample text",
            embedding=embedding
        )

        assert chunk.embedding is not None
        assert len(chunk.embedding) == 1536

    def test_chunk_default_values(self):
        """Test Chunk has correct default values."""
        chunk = Chunk(
            document_id=uuid.uuid4(),
            chunk_index=0,
            text="Sample text"
        )

        assert chunk.metadata == {}
        assert chunk.embedding is None
        assert chunk.chunk_id is not None

    def test_chunk_metadata_accepts_json(self):
        """Test JSONB metadata field accepts arbitrary JSON."""
        metadata = {
            "source": "page_1",
            "position": {"start": 0, "end": 100},
            "confidence": 0.95
        }

        chunk = Chunk(
            document_id=uuid.uuid4(),
            chunk_index=0,
            text="Sample text",
            metadata=metadata
        )

        assert chunk.metadata_["source"] == "page_1"
        assert chunk.metadata_["position"]["start"] == 0
        assert chunk.metadata_["confidence"] == 0.95

    def test_chunk_repr(self):
        """Test Chunk string representation."""
        doc_id = uuid.uuid4()
        embedding = [0.1] * 1536

        chunk = Chunk(
            document_id=doc_id,
            chunk_index=5,
            text="This is a sample text that will be truncated in repr",
            embedding=embedding
        )

        repr_str = repr(chunk)

        assert "Chunk" in repr_str
        assert "5" in repr_str  # chunk_index
        assert "embedding=yes" in repr_str

    def test_chunk_repr_without_embedding(self):
        """Test Chunk repr shows 'no' when embedding is None."""
        chunk = Chunk(
            document_id=uuid.uuid4(),
            chunk_index=0,
            text="Sample text"
        )

        repr_str = repr(chunk)
        assert "embedding=no" in repr_str

    def test_chunk_persistence(self, db_session):
        """Test Chunk can be persisted to database."""
        # First create a document
        doc = Document(
            filename="test.txt",
            source_path="/path/test.txt",
            content_hash="abc"
        )
        db_session.add(doc)
        db_session.flush()

        # Create chunk
        chunk = Chunk(
            document_id=doc.document_id,
            chunk_index=0,
            text="Sample chunk text"
        )
        db_session.add(chunk)
        db_session.flush()

        # Retrieve from database
        retrieved = db_session.query(Chunk).filter_by(
            chunk_index=0
        ).first()

        assert retrieved is not None
        assert retrieved.text == "Sample chunk text"
        assert retrieved.document_id == doc.document_id


class TestModelRelationships:
    """Test suite for relationships between models."""

    def test_foreign_key_relationship(self):
        """Test foreign key relationship between Chunk and Document."""
        # Check that Chunk has a document_id column with foreign key
        chunk_table = Chunk.__table__
        doc_id_column = chunk_table.c.document_id

        assert doc_id_column is not None

        # Check foreign keys on the column
        foreign_keys = list(doc_id_column.foreign_keys)

        assert len(foreign_keys) > 0
        fk = list(foreign_keys)[0]
        assert fk.column.table.name == "documents"

    def test_document_chunks_relationship(self, db_session):
        """Test Document.chunks relationship."""
        # Create document
        doc = Document(
            filename="test.txt",
            source_path="/path/test.txt",
            content_hash="abc"
        )
        db_session.add(doc)
        db_session.flush()

        # Create chunks
        chunks = [
            Chunk(
                document_id=doc.document_id,
                chunk_index=i,
                text=f"Chunk {i}"
            )
            for i in range(3)
        ]
        db_session.add_all(chunks)
        db_session.flush()

        # Access relationship
        retrieved_chunks = doc.chunks.all()

        assert len(retrieved_chunks) == 3
        assert all(c.document_id == doc.document_id for c in retrieved_chunks)

    def test_chunk_document_relationship(self, db_session):
        """Test Chunk.document relationship."""
        # Create document
        doc = Document(
            filename="test.txt",
            source_path="/path/test.txt",
            content_hash="abc"
        )
        db_session.add(doc)
        db_session.flush()

        # Create chunk
        chunk = Chunk(
            document_id=doc.document_id,
            chunk_index=0,
            text="Sample text"
        )
        db_session.add(chunk)
        db_session.flush()

        # Access relationship
        assert chunk.document is not None
        assert chunk.document.document_id == doc.document_id
        assert chunk.document.filename == "test.txt"

    def test_cascade_delete(self, db_session):
        """Test cascade delete removes chunks when document is deleted."""
        # Create document
        doc = Document(
            filename="test.txt",
            source_path="/path/test.txt",
            content_hash="abc"
        )
        db_session.add(doc)
        db_session.flush()

        # Create chunks
        chunks = [
            Chunk(
                document_id=doc.document_id,
                chunk_index=i,
                text=f"Chunk {i}"
            )
            for i in range(3)
        ]
        db_session.add_all(chunks)
        db_session.flush()

        doc_id = doc.document_id

        # Delete document
        db_session.delete(doc)
        db_session.flush()

        # Verify chunks are deleted
        remaining_chunks = db_session.query(Chunk).filter_by(
            document_id=doc_id
        ).count()

        assert remaining_chunks == 0


class TestModelIndexes:
    """Test suite for model indexes."""

    def test_document_indexes(self):
        """Test Document model has expected indexes."""
        inspector = inspect(Document)

        # Get all indexed columns
        indexed_columns = set()
        for col in inspector.columns:
            if col.index:
                indexed_columns.add(col.name)

        # These columns should be indexed
        assert "content_hash" in indexed_columns
        assert "filename" in indexed_columns
        assert "status" in indexed_columns

    def test_chunk_indexes(self):
        """Test Chunk model has expected indexes."""
        inspector = inspect(Chunk)

        # Get all indexed columns
        indexed_columns = set()
        for col in inspector.columns:
            if col.index:
                indexed_columns.add(col.name)

        # document_id should be indexed (foreign key)
        assert "document_id" in indexed_columns
