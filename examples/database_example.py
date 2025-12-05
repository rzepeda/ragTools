#!/usr/bin/env python3
"""Example demonstrating database usage with RAG Factory.

This example shows how to:
1. Connect to the database
2. Create documents and chunks
3. Perform vector similarity search
4. Query with filters

Prerequisites:
- PostgreSQL with pgvector installed
- Database created and migrations run
- DB_DATABASE_URL environment variable set
"""

import numpy as np
from rag_factory.database import DatabaseConnection, Document, Chunk
from sqlalchemy import text


def main():
    """Main example function."""

    print("=" * 60)
    print("RAG Factory Database Example")
    print("=" * 60)
    print()

    # Initialize database connection
    print("1. Connecting to database...")
    db = DatabaseConnection()

    # Check database health
    if not db.health_check():
        print("❌ Database connection failed!")
        print("   Make sure PostgreSQL is running and DB_DATABASE_URL is set")
        return

    print("✓ Database connection successful")
    print()

    # Create sample documents and chunks
    print("2. Creating sample documents and chunks...")
    with db.get_session() as session:
        # Create first document
        doc1 = Document(
            filename="machine_learning.txt",
            source_path="/docs/ml.txt",
            content_hash="hash_ml_123",
            total_chunks=3,
            status="completed"
        )
        session.add(doc1)
        session.flush()

        # Create chunks for doc1
        ml_texts = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "Neural networks are inspired by the human brain and consist of interconnected nodes called neurons.",
            "Deep learning uses multi-layer neural networks to learn hierarchical representations of data."
        ]

        for i, text in enumerate(ml_texts):
            # Generate a simple embedding (in practice, use a real embedding model)
            embedding = np.random.rand(1536).tolist()

            chunk = Chunk(
                document_id=doc1.document_id,
                chunk_index=i,
                text=text,
                embedding=embedding,
                metadata_={"topic": "machine_learning", "page": i + 1}
            )
            session.add(chunk)

        # Create second document
        doc2 = Document(
            filename="database_systems.txt",
            source_path="/docs/db.txt",
            content_hash="hash_db_456",
            total_chunks=2,
            status="completed"
        )
        session.add(doc2)
        session.flush()

        # Create chunks for doc2
        db_texts = [
            "Relational databases organize data into tables with rows and columns.",
            "Vector databases are optimized for storing and searching high-dimensional vectors."
        ]

        for i, text in enumerate(db_texts):
            embedding = np.random.rand(1536).tolist()

            chunk = Chunk(
                document_id=doc2.document_id,
                chunk_index=i,
                text=text,
                embedding=embedding,
                metadata_={"topic": "databases", "page": i + 1}
            )
            session.add(chunk)

    print(f"✓ Created 2 documents with 5 total chunks")
    print()

    # Query documents
    print("3. Querying documents...")
    with db.get_session() as session:
        docs = session.query(Document).filter_by(status="completed").all()

        print(f"   Found {len(docs)} completed documents:")
        for doc in docs:
            print(f"   - {doc.filename} ({doc.total_chunks} chunks)")
    print()

    # Query chunks for a specific document
    print("4. Querying chunks for machine_learning.txt...")
    with db.get_session() as session:
        doc = session.query(Document).filter_by(
            filename="machine_learning.txt"
        ).first()

        if doc:
            chunks = session.query(Chunk).filter_by(
                document_id=doc.document_id
            ).order_by(Chunk.chunk_index).all()

            print(f"   Found {len(chunks)} chunks:")
            for chunk in chunks:
                print(f"   [{chunk.chunk_index}] {chunk.text[:80]}...")
    print()

    # Demonstrate vector similarity search
    print("5. Performing vector similarity search...")
    print("   Query: 'neural networks and learning'")

    # Generate query embedding (in practice, use same embedding model as chunks)
    query_embedding = np.random.rand(1536).tolist()

    with db.get_session() as session:
        # Check if we're using PostgreSQL (vector search only works with pgvector)
        if "postgresql" in str(db.config.database_url):
            results = session.execute(
                text("""
                    SELECT c.chunk_id, c.text, d.filename,
                           1 - (c.embedding <=> CAST(:query_vector AS vector)) as similarity
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.document_id
                    WHERE c.embedding IS NOT NULL
                    ORDER BY c.embedding <=> CAST(:query_vector AS vector)
                    LIMIT 3
                """),
                {"query_vector": str(query_embedding)}
            ).fetchall()

            print(f"   Top {len(results)} most similar chunks:")
            for i, (chunk_id, text, filename, similarity) in enumerate(results, 1):
                print(f"   {i}. [{filename}] (similarity: {similarity:.4f})")
                print(f"      {text[:100]}...")
                print()
        else:
            print("   ⚠ Vector search requires PostgreSQL with pgvector")
            print("   Current database doesn't support vector operations")
    print()

    # Show connection pool status
    print("6. Connection pool status:")
    status = db.get_pool_status()
    print(f"   Pool size: {status['size']}")
    print(f"   Checked out: {status['checked_out']}")
    print(f"   Checked in: {status['checked_in']}")
    print(f"   Overflow: {status['overflow']}")
    print()

    # Clean up
    print("7. Cleaning up...")
    with db.get_session() as session:
        # Delete all test documents (chunks will be cascade deleted)
        session.query(Document).filter(
            Document.content_hash.in_(["hash_ml_123", "hash_db_456"])
        ).delete(synchronize_session=False)

    print("✓ Test data cleaned up")
    print()

    # Close connection
    db.close()
    print("✓ Database connection closed")
    print()
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. PostgreSQL is running")
        print("2. pgvector extension is installed")
        print("3. DB_DATABASE_URL environment variable is set")
        print("4. Database migrations have been run (alembic upgrade head)")
        import traceback
        traceback.print_exc()
