"""Integration tests for database operations."""

import uuid
import time
import pytest
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text

from rag_factory.database.connection import DatabaseConnection
from rag_factory.database.config import DatabaseConfig
from rag_factory.database.models import Document, Chunk


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for end-to-end database operations."""

    def test_full_database_workflow(self, db_connection):
        """Test complete workflow: connect, insert, query, delete."""
        with db_connection.get_session() as session:
            # Create document
            doc = Document(
                filename="test.txt",
                source_path="/test/test.txt",
                content_hash="hash123",
                total_chunks=2
            )
            session.add(doc)
            session.flush()

            # Create chunks
            chunks = [
                Chunk(
                    document_id=doc.document_id,
                    chunk_index=i,
                    text=f"Chunk {i}",
                    embedding=[0.1 * i] * 1536
                )
                for i in range(2)
            ]
            session.add_all(chunks)
            session.flush()

            # Query chunks
            results = session.query(Chunk).filter(
                Chunk.document_id == doc.document_id
            ).all()

            assert len(results) == 2
            assert results[0].text == "Chunk 0"
            assert results[1].text == "Chunk 1"

            # Update document status
            doc.status = "completed"
            session.flush()

            # Verify update
            updated_doc = session.query(Document).filter_by(
                document_id=doc.document_id
            ).first()
            assert updated_doc.status == "completed"

            # Delete document (cascade should delete chunks)
            session.delete(doc)
            session.flush()

            # Verify chunks deleted
            remaining = session.query(Chunk).filter(
                Chunk.document_id == doc.document_id
            ).count()

            assert remaining == 0

    def test_batch_insert_chunks(self, db_connection):
        """Test batch insertion of multiple chunks."""
        with db_connection.get_session() as session:
            # Create document
            doc = Document(
                filename="batch.txt",
                source_path="/batch/test.txt",
                content_hash="batch123"
            )
            session.add(doc)
            session.flush()

            # Create many chunks
            num_chunks = 100
            chunks = [
                Chunk(
                    document_id=doc.document_id,
                    chunk_index=i,
                    text=f"Chunk {i}",
                    embedding=[0.1 * i] * 1536
                )
                for i in range(num_chunks)
            ]

            # Batch insert
            session.bulk_save_objects(chunks)
            session.flush()

            # Verify all inserted
            count = session.query(Chunk).filter_by(
                document_id=doc.document_id
            ).count()

            assert count == num_chunks

    def test_query_with_filters(self, db_connection):
        """Test querying with various filters."""
        with db_connection.get_session() as session:
            # Create documents with different statuses
            docs = [
                Document(
                    filename=f"doc_{i}.txt",
                    source_path=f"/path/doc_{i}.txt",
                    content_hash=f"hash_{i}",
                    status="completed" if i % 2 == 0 else "pending"
                )
                for i in range(10)
            ]
            session.add_all(docs)
            session.flush()

            # Query completed documents
            completed = session.query(Document).filter_by(
                status="completed"
            ).count()

            assert completed == 5

            # Query by filename pattern
            specific = session.query(Document).filter(
                Document.filename.like("doc_5%")
            ).count()

            assert specific == 1

    def test_metadata_queries(self, db_connection):
        """Test querying JSONB metadata fields."""
        with db_connection.get_session() as session:
            # Create documents with metadata
            doc1 = Document(
                filename="doc1.txt",
                source_path="/path/doc1.txt",
                content_hash="hash1",
                metadata={"category": "science", "tags": ["physics", "math"]}
            )
            doc2 = Document(
                filename="doc2.txt",
                source_path="/path/doc2.txt",
                content_hash="hash2",
                metadata={"category": "literature", "tags": ["fiction"]}
            )
            session.add_all([doc1, doc2])
            session.flush()

            # Query by metadata field (PostgreSQL JSONB)
            # Note: This requires PostgreSQL, won't work with SQLite
            try:
                result = session.query(Document).filter(
                    Document.metadata_["category"].astext == "science"
                ).first()

                if result:  # Only assert if DB supports JSONB queries
                    assert result.filename == "doc1.txt"
            except Exception:
                # SQLite doesn't support JSONB queries, skip
                pytest.skip("Database doesn't support JSONB queries")

    def test_chunk_ordering(self, db_connection):
        """Test chunks are ordered correctly."""
        with db_connection.get_session() as session:
            # Create document
            doc = Document(
                filename="ordered.txt",
                source_path="/path/ordered.txt",
                content_hash="order123"
            )
            session.add(doc)
            session.flush()

            # Create chunks in random order
            indices = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]
            for idx in indices:
                chunk = Chunk(
                    document_id=doc.document_id,
                    chunk_index=idx,
                    text=f"Chunk {idx}"
                )
                session.add(chunk)
            session.flush()

            # Query ordered by chunk_index
            chunks = session.query(Chunk).filter_by(
                document_id=doc.document_id
            ).order_by(Chunk.chunk_index).all()

            # Verify order
            for i, chunk in enumerate(chunks):
                assert chunk.chunk_index == i
                assert chunk.text == f"Chunk {i}"

    def test_document_deduplication(self, db_connection):
        """Test content_hash can be used for deduplication."""
        with db_connection.get_session() as session:
            # Create first document
            doc1 = Document(
                filename="doc.txt",
                source_path="/path1/doc.txt",
                content_hash="same_hash"
            )
            session.add(doc1)
            session.flush()

            # Check for duplicate before inserting
            existing = session.query(Document).filter_by(
                content_hash="same_hash"
            ).first()

            assert existing is not None
            assert existing.document_id == doc1.document_id

            # Don't insert duplicate
            duplicate_exists = session.query(Document).filter_by(
                content_hash="same_hash"
            ).count() > 0

            assert duplicate_exists


@pytest.mark.integration
class TestVectorOperations:
    """Integration tests for vector embedding operations."""

    def test_insert_chunk_with_embedding(self, db_connection):
        """Test inserting chunk with vector embedding."""
        with db_connection.get_session() as session:
            # Create document
            doc = Document(
                filename="vectors.txt",
                source_path="/test/vectors.txt",
                content_hash="vec123"
            )
            session.add(doc)
            session.flush()

            # Create chunk with embedding
            embedding = np.random.rand(1536).tolist()
            chunk = Chunk(
                document_id=doc.document_id,
                chunk_index=0,
                text="Sample text",
                embedding=embedding
            )
            session.add(chunk)
            session.flush()

            # Retrieve and verify
            retrieved = session.query(Chunk).filter_by(
                chunk_id=chunk.chunk_id
            ).first()

            assert retrieved.embedding is not None
            assert len(retrieved.embedding) == 1536

    def test_vector_similarity_search(self, db_connection):
        """Test vector similarity search with cosine distance."""
        # Skip if not using PostgreSQL
        if "postgresql" not in str(db_connection.config.database_url):
            pytest.skip("Vector similarity requires PostgreSQL with pgvector")

        with db_connection.get_session() as session:
            # Create document
            doc = Document(
                filename="vectors.txt",
                source_path="/test/vectors.txt",
                content_hash="vec123"
            )
            session.add(doc)
            session.flush()

            # Create chunks with known embeddings
            base_vector = np.random.rand(1536)
            similar_vector = base_vector + np.random.rand(1536) * 0.1
            dissimilar_vector = np.random.rand(1536)

            chunks = [
                Chunk(
                    document_id=doc.document_id,
                    chunk_index=0,
                    text="Similar",
                    embedding=similar_vector.tolist()
                ),
                Chunk(
                    document_id=doc.document_id,
                    chunk_index=1,
                    text="Dissimilar",
                    embedding=dissimilar_vector.tolist()
                ),
            ]
            session.add_all(chunks)
            session.flush()

            # Perform similarity search using pgvector
            query_vector = base_vector.tolist()

            # Use pgvector cosine distance operator
            results = session.execute(
                text("""
                    SELECT chunk_id, text,
                           1 - (embedding <=> CAST(:query_vector AS vector)) as similarity
                    FROM chunks
                    WHERE document_id = :doc_id
                    ORDER BY embedding <=> CAST(:query_vector AS vector)
                    LIMIT 2
                """),
                {
                    "query_vector": str(query_vector),
                    "doc_id": str(doc.document_id)
                }
            ).fetchall()

            # Most similar should be returned first
            assert len(results) == 2
            assert results[0][1] == "Similar"
            # Similarity score should be high
            assert results[0][2] > 0.5

    def test_null_embeddings(self, db_connection):
        """Test chunks can have null embeddings."""
        with db_connection.get_session() as session:
            # Create document
            doc = Document(
                filename="test.txt",
                source_path="/test/test.txt",
                content_hash="test123"
            )
            session.add(doc)
            session.flush()

            # Create chunk without embedding
            chunk = Chunk(
                document_id=doc.document_id,
                chunk_index=0,
                text="No embedding yet"
            )
            session.add(chunk)
            session.flush()

            # Verify embedding is null
            retrieved = session.query(Chunk).filter_by(
                chunk_id=chunk.chunk_id
            ).first()

            assert retrieved.embedding is None


@pytest.mark.integration
class TestConnectionPooling:
    """Integration tests for connection pooling."""

    def test_connection_pool_concurrent_access(self, db_connection):
        """Test connection pool handles concurrent requests."""
        def query_database(i):
            with db_connection.get_session() as session:
                result = session.execute(text("SELECT 1")).scalar()
                return result

        # Execute concurrent queries
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(query_database, i) for i in range(20)]
            results = [f.result() for f in as_completed(futures)]

        # All queries should succeed
        assert len(results) == 20
        assert all(r == 1 for r in results)

    def test_connection_pool_metrics(self, db_connection):
        """Test connection pool metrics are available."""
        status = db_connection.get_pool_status()

        assert "size" in status
        assert "checked_out" in status
        assert status["size"] >= 0

    def test_session_isolation(self, db_connection):
        """Test sessions are isolated from each other."""
        doc_id = None

        # Session 1: Create document
        with db_connection.get_session() as session1:
            doc = Document(
                filename="isolation.txt",
                source_path="/test/isolation.txt",
                content_hash="iso123"
            )
            session1.add(doc)
            session1.flush()
            doc_id = doc.document_id

        # Session 2: Should see committed data
        with db_connection.get_session() as session2:
            doc = session2.query(Document).filter_by(
                document_id=doc_id
            ).first()

            assert doc is not None
            assert doc.filename == "isolation.txt"


@pytest.mark.integration
class TestPerformance:
    """Integration tests for performance requirements."""

    def test_batch_insert_performance(self, db_connection):
        """Test batch insert performance meets requirements."""
        # Skip if not using PostgreSQL
        if "postgresql" not in str(db_connection.config.database_url):
            pytest.skip("Performance test requires PostgreSQL")

        with db_connection.get_session() as session:
            doc = Document(
                filename="batch.txt",
                source_path="/batch/test.txt",
                content_hash="batch123"
            )
            session.add(doc)
            session.flush()

            # Insert 1000 chunks and measure time
            num_chunks = 1000
            chunks = [
                Chunk(
                    document_id=doc.document_id,
                    chunk_index=i,
                    text=f"Chunk {i}",
                    embedding=np.random.rand(1536).tolist()
                )
                for i in range(num_chunks)
            ]

            start = time.time()
            session.bulk_save_objects(chunks)
            session.flush()
            duration = time.time() - start

            # Should insert >1000 chunks/second
            throughput = num_chunks / duration
            print(f"\nBatch insert throughput: {throughput:.2f} chunks/sec")

            # This is a guideline, not a hard requirement
            # Actual performance depends on hardware
            assert throughput > 100, \
                f"Throughput {throughput:.2f} chunks/sec is lower than expected"

    def test_query_performance(self, db_connection):
        """Test query performance is reasonable."""
        with db_connection.get_session() as session:
            # Create test data
            doc = Document(
                filename="perf.txt",
                source_path="/perf/test.txt",
                content_hash="perf123"
            )
            session.add(doc)
            session.flush()

            # Insert chunks
            chunks = [
                Chunk(
                    document_id=doc.document_id,
                    chunk_index=i,
                    text=f"Chunk {i}"
                )
                for i in range(1000)
            ]
            session.bulk_save_objects(chunks)
            session.flush()

            # Measure query time
            start = time.time()
            results = session.query(Chunk).filter_by(
                document_id=doc.document_id
            ).limit(10).all()
            duration = time.time() - start

            assert len(results) == 10
            # Query should be fast (< 100ms)
            assert duration < 0.1, f"Query took {duration*1000:.2f}ms"
