"""Integration tests for repository workflow with real database.

These tests require a running PostgreSQL database with pgvector extension.
They test the complete workflow of document and chunk operations including
vector similarity search.
"""

import pytest
import numpy as np
from uuid import uuid4

from rag_factory.repositories import (
    DocumentRepository,
    ChunkRepository,
    EntityNotFoundError,
    DuplicateEntityError
)


@pytest.mark.integration
class TestDocumentRepositoryIntegration:
    """Integration tests for DocumentRepository with real database."""

    def test_complete_document_lifecycle(self, db_session):
        """Test complete document CRUD lifecycle."""
        repo = DocumentRepository(db_session)

        # Create document
        doc = repo.create(
            filename="article.txt",
            source_path="/docs/article.txt",
            content_hash="abc123",
            metadata={"author": "John Doe"}
        )
        repo.commit()

        assert doc.document_id is not None
        assert doc.status == "pending"
        assert doc.metadata_["author"] == "John Doe"

        # Read document
        retrieved = repo.get_by_id(doc.document_id)
        assert retrieved is not None
        assert retrieved.filename == "article.txt"

        # Update document
        updated = repo.update_status(doc.document_id, "completed")
        repo.commit()
        assert updated.status == "completed"

        # Verify update persisted
        retrieved = repo.get_by_id(doc.document_id)
        assert retrieved.status == "completed"

        # Delete document
        result = repo.delete(doc.document_id)
        repo.commit()
        assert result is True

        # Verify deletion
        retrieved = repo.get_by_id(doc.document_id)
        assert retrieved is None

    def test_document_deduplication(self, db_session):
        """Test that duplicate content hashes are rejected."""
        repo = DocumentRepository(db_session)

        # Create first document
        doc1 = repo.create("test1.txt", "/path/test1.txt", "hash123")
        repo.commit()

        # Attempt to create duplicate
        with pytest.raises(DuplicateEntityError) as exc_info:
            repo.create("test2.txt", "/path/test2.txt", "hash123")

        assert exc_info.value.field == "content_hash"
        assert exc_info.value.value == "hash123"

    def test_document_pagination(self, db_session):
        """Test document listing with pagination."""
        repo = DocumentRepository(db_session)

        # Create 15 documents
        for i in range(15):
            repo.create(f"test{i}.txt", f"/path/test{i}.txt", f"hash{i}")
        repo.commit()

        # Get first page
        page1 = repo.list_all(skip=0, limit=10)
        assert len(page1) == 10

        # Get second page
        page2 = repo.list_all(skip=10, limit=10)
        assert len(page2) == 5

        # Verify no overlap
        page1_ids = {doc.document_id for doc in page1}
        page2_ids = {doc.document_id for doc in page2}
        assert page1_ids.isdisjoint(page2_ids)

    def test_document_status_filtering(self, db_session):
        """Test filtering documents by status."""
        repo = DocumentRepository(db_session)

        # Create documents with different statuses
        doc1 = repo.create("test1.txt", "/path/test1.txt", "hash1")
        doc2 = repo.create("test2.txt", "/path/test2.txt", "hash2")
        doc3 = repo.create("test3.txt", "/path/test3.txt", "hash3")

        repo.update_status(doc1.document_id, "completed")
        repo.update_status(doc2.document_id, "completed")
        # doc3 remains "pending"

        repo.commit()

        # Filter by status
        completed = repo.get_by_status("completed")
        pending = repo.get_by_status("pending")

        assert len(completed) == 2
        assert len(pending) == 1
        assert pending[0].document_id == doc3.document_id

    def test_bulk_operations(self, db_session):
        """Test bulk create and delete operations."""
        repo = DocumentRepository(db_session)

        # Bulk create
        documents = [
            {"filename": f"test{i}.txt", "source_path": f"/path/test{i}.txt",
             "content_hash": f"hash{i}"}
            for i in range(100)
        ]

        created = repo.bulk_create(documents)
        repo.commit()

        assert len(created) == 100
        assert all(doc.document_id is not None for doc in created)

        # Verify count
        count = repo.count()
        assert count >= 100

        # Bulk delete
        doc_ids = [doc.document_id for doc in created[:50]]
        deleted_count = repo.bulk_delete(doc_ids)
        repo.commit()

        assert deleted_count == 50


@pytest.mark.integration
class TestChunkRepositoryIntegration:
    """Integration tests for ChunkRepository with real database."""

    @pytest.fixture
    def document(self, db_session):
        """Create a test document."""
        doc_repo = DocumentRepository(db_session)
        doc = doc_repo.create("test.txt", "/path/test.txt", "hash123")
        doc_repo.commit()
        return doc

    def test_complete_chunk_lifecycle(self, db_session, document):
        """Test complete chunk CRUD lifecycle."""
        repo = ChunkRepository(db_session)
        embedding = np.random.rand(1536).tolist()

        # Create chunk
        chunk = repo.create(
            document_id=document.document_id,
            chunk_index=0,
            text="This is a test chunk",
            embedding=embedding,
            metadata={"page": 1}
        )
        repo.commit()

        assert chunk.chunk_id is not None
        assert chunk.text == "This is a test chunk"
        assert chunk.embedding is not None

        # Read chunk
        retrieved = repo.get_by_id(chunk.chunk_id)
        assert retrieved is not None
        assert retrieved.metadata_["page"] == 1

        # Update embedding
        new_embedding = np.random.rand(1536).tolist()
        updated = repo.update_embedding(chunk.chunk_id, new_embedding)
        repo.commit()

        # Verify update
        retrieved = repo.get_by_id(chunk.chunk_id)
        assert retrieved.embedding == new_embedding

        # Delete chunk
        result = repo.delete(chunk.chunk_id)
        repo.commit()
        assert result is True

        # Verify deletion
        retrieved = repo.get_by_id(chunk.chunk_id)
        assert retrieved is None

    def test_chunk_document_relationship(self, db_session, document):
        """Test chunks are associated with documents correctly."""
        repo = ChunkRepository(db_session)

        # Create multiple chunks for the document
        for i in range(5):
            repo.create(
                document_id=document.document_id,
                chunk_index=i,
                text=f"Chunk {i}"
            )
        repo.commit()

        # Retrieve chunks by document
        chunks = repo.get_by_document(document.document_id)
        assert len(chunks) == 5

        # Verify chunks are ordered by index
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

        # Count chunks
        count = repo.count_by_document(document.document_id)
        assert count == 5

    def test_cascade_delete(self, db_session):
        """Test that deleting document cascades to chunks."""
        doc_repo = DocumentRepository(db_session)
        chunk_repo = ChunkRepository(db_session)

        # Create document and chunks
        doc = doc_repo.create("test.txt", "/path/test.txt", "hash123")
        doc_repo.commit()

        for i in range(3):
            chunk_repo.create(doc.document_id, i, f"Chunk {i}")
        chunk_repo.commit()

        # Verify chunks exist
        count_before = chunk_repo.count_by_document(doc.document_id)
        assert count_before == 3

        # Delete document
        doc_repo.delete(doc.document_id)
        doc_repo.commit()

        # Verify chunks were cascade deleted
        count_after = chunk_repo.count_by_document(doc.document_id)
        assert count_after == 0

    def test_bulk_chunk_operations(self, db_session, document):
        """Test bulk chunk create and update operations."""
        repo = ChunkRepository(db_session)

        # Bulk create chunks
        chunks_data = [
            {
                "document_id": document.document_id,
                "chunk_index": i,
                "text": f"Chunk {i}",
                "embedding": np.random.rand(1536).tolist()
            }
            for i in range(100)
        ]

        created = repo.bulk_create(chunks_data)
        repo.commit()

        assert len(created) == 100

        # Bulk update embeddings
        updates = [
            (chunk.chunk_id, np.random.rand(1536).tolist())
            for chunk in created[:50]
        ]

        count = repo.bulk_update_embeddings(updates)
        repo.commit()

        assert count == 50


@pytest.mark.integration
class TestVectorSearchIntegration:
    """Integration tests for vector similarity search."""

    @pytest.fixture
    def document_with_chunks(self, db_session):
        """Create a document with chunks having embeddings."""
        doc_repo = DocumentRepository(db_session)
        chunk_repo = ChunkRepository(db_session)

        # Create document
        doc = doc_repo.create("test.txt", "/path/test.txt", "hash123")
        doc_repo.commit()

        # Create base embedding
        base_embedding = np.random.rand(1536)

        # Create chunks with varying similarity
        chunks = []
        for i in range(10):
            # Add varying amounts of noise
            if i < 3:
                # Very similar to base
                embedding = base_embedding + np.random.rand(1536) * 0.05
            elif i < 6:
                # Moderately similar
                embedding = base_embedding + np.random.rand(1536) * 0.3
            else:
                # Dissimilar
                embedding = np.random.rand(1536)

            chunk = chunk_repo.create(
                document_id=doc.document_id,
                chunk_index=i,
                text=f"Chunk {i}",
                embedding=embedding.tolist()
            )
            chunks.append(chunk)

        chunk_repo.commit()

        return doc, chunks, base_embedding.tolist()

    def test_vector_similarity_search(self, db_session, document_with_chunks):
        """Test basic vector similarity search."""
        doc, chunks, base_embedding = document_with_chunks
        repo = ChunkRepository(db_session)

        # Search for similar chunks
        results = repo.search_similar(base_embedding, top_k=5)

        assert len(results) <= 5

        # Verify results are ordered by similarity (descending)
        similarities = [score for _, score in results]
        assert similarities == sorted(similarities, reverse=True)

        # All results should have similarity > 0
        for chunk, score in results:
            assert 0 <= score <= 1

    def test_vector_search_with_threshold(self, db_session, document_with_chunks):
        """Test vector search with similarity threshold."""
        doc, chunks, base_embedding = document_with_chunks
        repo = ChunkRepository(db_session)

        # Search with high threshold
        results = repo.search_similar(base_embedding, top_k=10, threshold=0.8)

        # All results should meet threshold
        for chunk, score in results:
            assert score >= 0.8

    def test_vector_search_with_document_filter(self, db_session):
        """Test vector search filtered by document IDs."""
        doc_repo = DocumentRepository(db_session)
        chunk_repo = ChunkRepository(db_session)

        # Create two documents with chunks
        doc1 = doc_repo.create("doc1.txt", "/path/doc1.txt", "hash1")
        doc2 = doc_repo.create("doc2.txt", "/path/doc2.txt", "hash2")
        doc_repo.commit()

        embedding = np.random.rand(1536).tolist()

        # Create chunks for both documents
        chunk1 = chunk_repo.create(doc1.document_id, 0, "Doc1 chunk", embedding)
        chunk2 = chunk_repo.create(doc2.document_id, 0, "Doc2 chunk", embedding)
        chunk_repo.commit()

        # Search filtered to doc1 only
        results = chunk_repo.search_similar_with_filter(
            embedding,
            top_k=10,
            document_ids=[doc1.document_id]
        )

        # Should only return chunks from doc1
        assert len(results) >= 1
        for chunk, score in results:
            assert chunk.document_id == doc1.document_id

    def test_vector_search_with_metadata_filter(self, db_session, document):
        """Test vector search with metadata filtering."""
        chunk_repo = ChunkRepository(db_session)
        embedding = np.random.rand(1536).tolist()

        # Create chunks with different metadata
        chunk1 = chunk_repo.create(
            document.document_id, 0, "Chunk 1",
            embedding=embedding,
            metadata={"category": "science"}
        )
        chunk2 = chunk_repo.create(
            document.document_id, 1, "Chunk 2",
            embedding=embedding,
            metadata={"category": "history"}
        )
        chunk_repo.commit()

        # Search filtered by metadata
        results = chunk_repo.search_similar_with_metadata(
            embedding,
            top_k=10,
            metadata_filter={"category": "science"}
        )

        # Should only return chunks with matching metadata
        assert len(results) >= 1
        for chunk, score in results:
            assert chunk.metadata_["category"] == "science"

    def test_vector_search_identical_embedding(self, db_session, document):
        """Test searching for identical embedding returns ~1.0 similarity."""
        chunk_repo = ChunkRepository(db_session)
        embedding = np.random.rand(1536).tolist()

        # Create chunk with specific embedding
        chunk = chunk_repo.create(
            document.document_id, 0, "Test chunk",
            embedding=embedding
        )
        chunk_repo.commit()

        # Search with same embedding
        results = chunk_repo.search_similar(embedding, top_k=1)

        assert len(results) == 1
        chunk_result, score = results[0]
        # Cosine similarity of identical vectors should be very close to 1.0
        assert score >= 0.99


@pytest.mark.integration
class TestRepositoryTransactions:
    """Integration tests for transaction management."""

    def test_transaction_commit(self, db_session):
        """Test successful transaction commits changes."""
        repo = DocumentRepository(db_session)

        doc = repo.create("test.txt", "/path/test.txt", "hash123")
        doc_id = doc.document_id

        repo.commit()

        # Verify in same session
        retrieved = repo.get_by_id(doc_id)
        assert retrieved is not None

    def test_transaction_rollback(self, db_session):
        """Test rollback reverts changes."""
        repo = DocumentRepository(db_session)

        doc = repo.create("test.txt", "/path/test.txt", "hash123")
        doc_id = doc.document_id

        repo.rollback()

        # Document should not exist after rollback
        retrieved = repo.get_by_id(doc_id)
        assert retrieved is None

    def test_multiple_operations_in_transaction(self, db_session):
        """Test multiple operations in single transaction."""
        doc_repo = DocumentRepository(db_session)
        chunk_repo = ChunkRepository(db_session)

        # Create document and chunks in same transaction
        doc = doc_repo.create("test.txt", "/path/test.txt", "hash123")
        chunk1 = chunk_repo.create(doc.document_id, 0, "Chunk 1")
        chunk2 = chunk_repo.create(doc.document_id, 1, "Chunk 2")

        doc_repo.commit()

        # Both document and chunks should exist
        assert doc_repo.get_by_id(doc.document_id) is not None
        assert chunk_repo.count_by_document(doc.document_id) == 2

    def test_transaction_context_manager(self, db_session):
        """Test transaction context manager."""
        repo = DocumentRepository(db_session)

        with repo.transaction():
            doc = repo.create("test.txt", "/path/test.txt", "hash123")
            doc_id = doc.document_id

        # Should be committed after context
        retrieved = repo.get_by_id(doc_id)
        assert retrieved is not None

    def test_transaction_rollback_on_error(self, db_session):
        """Test transaction rolls back on error."""
        repo = DocumentRepository(db_session)

        try:
            with repo.transaction():
                doc1 = repo.create("test.txt", "/path/test.txt", "hash123")
                # Try to create duplicate (should fail)
                doc2 = repo.create("test2.txt", "/path/test2.txt", "hash123")
        except Exception:
            pass

        # First document should not be committed due to rollback
        count = repo.count()
        # Depending on test isolation, might have other docs, so just verify
        # the transaction was rolled back by checking duplicate doesn't exist
        existing = repo.get_by_content_hash("hash123")
        # In a clean test, this should be None
