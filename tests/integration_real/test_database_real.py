"""
Real integration tests for PostgreSQL database service.

Tests actual database operations with real PostgreSQL instance configured via .env.
"""

import pytest
from rag_factory.repositories.chunk import Chunk


@pytest.mark.real_integration
@pytest.mark.requires_postgres
@pytest.mark.asyncio
async def test_postgres_connection(real_db_service):
    """Test that we can connect to PostgreSQL."""
    # Service should be initialized
    assert real_db_service is not None
    
    # Get the async pool and test basic query
    pool = await real_db_service._get_pool()
    assert pool is not None
    
    async with pool.acquire() as conn:
        result = await conn.fetchval("SELECT 1 as test")
        assert result == 1


@pytest.mark.real_integration
@pytest.mark.requires_postgres
@pytest.mark.asyncio
async def test_table_creation(real_db_service):
    """Test that the service creates the chunks table."""
    # Get pool (this will create the table)
    pool = await real_db_service._get_pool()
    
    # Check if table exists
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.tables "
            "WHERE table_name = 'test_chunks_real')"
        )
        assert exists, "Table test_chunks_real should exist"


@pytest.mark.real_integration
@pytest.mark.requires_postgres
@pytest.mark.requires_embeddings
@pytest.mark.asyncio
async def test_store_and_retrieve_chunks(real_db_service, real_embedding_service, sample_texts):
    """Test storing and retrieving chunks with real embeddings."""
    # Generate real embeddings
    chunks = []
    for i, text in enumerate(sample_texts[:3]):  # Use first 3 texts
        embedding = await real_embedding_service.embed(text)
        chunk = Chunk(
            chunk_id=f"test_chunk_{i}",
            text=text,
            embedding=embedding,
            metadata={"source": f"test_{i}.txt", "index": i}
        )
        chunks.append(chunk)
    
    # Store chunks
    await real_db_service.store_chunks(chunks)
    
    # Retrieve all chunks
    retrieved = await real_db_service.get_all_chunks()
    assert len(retrieved) >= 3  # At least our 3 chunks
    
    # Verify chunk data
    chunk_ids = [c.chunk_id for c in retrieved]
    assert "test_chunk_0" in chunk_ids
    assert "test_chunk_1" in chunk_ids
    assert "test_chunk_2" in chunk_ids


@pytest.mark.real_integration
@pytest.mark.requires_postgres
@pytest.mark.requires_embeddings
@pytest.mark.asyncio
async def test_vector_similarity_search(real_db_service, real_embedding_service):
    """Test actual vector similarity search with real embeddings."""
    # Prepare test data with semantically related texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn canine leaps above a sleepy hound",  # Similar to first
        "Python is a programming language",  # Different topic
    ]
    
    # Generate and store chunks
    chunks = []
    for i, text in enumerate(texts):
        embedding = await real_embedding_service.embed(text)
        chunk = Chunk(
            chunk_id=f"similarity_test_{i}",
            text=text,
            embedding=embedding,
            metadata={"source": "similarity_test.txt"}
        )
        chunks.append(chunk)
    
    await real_db_service.store_chunks(chunks)
    
    # Search with query similar to first two texts
    query = "fast dog jumping"
    query_embedding = await real_embedding_service.embed(query)
    
    results = await real_db_service.search_chunks(
        query_embedding=query_embedding,
        top_k=2
    )
    
    # Verify results
    assert len(results) == 2
    
    # First result should be one of the dog-related texts
    top_result = results[0]
    assert "fox" in top_result.text.lower() or "canine" in top_result.text.lower()
    
    # Results should have similarity scores
    assert hasattr(top_result, 'metadata')
    # Similarity should be reasonable (not too low)
    # Note: actual similarity values depend on the embedding model


@pytest.mark.real_integration
@pytest.mark.requires_postgres
@pytest.mark.requires_embeddings
@pytest.mark.asyncio
async def test_batch_embedding_and_storage(real_db_service, real_embedding_service, sample_texts):
    """Test batch embedding and storage operations."""
    # Generate embeddings in batch
    embeddings = await real_embedding_service.embed_batch(sample_texts)
    
    assert len(embeddings) == len(sample_texts)
    
    # Create chunks
    chunks = [
        Chunk(
            chunk_id=f"batch_test_{i}",
            text=text,
            embedding=embedding,
            metadata={"batch": True, "index": i}
        )
        for i, (text, embedding) in enumerate(zip(sample_texts, embeddings))
    ]
    
    # Store all chunks
    await real_db_service.store_chunks(chunks)
    
    # Verify all were stored
    retrieved = await real_db_service.get_all_chunks()
    batch_chunks = [c for c in retrieved if c.chunk_id.startswith("batch_test_")]
    assert len(batch_chunks) == len(sample_texts)


@pytest.mark.real_integration
@pytest.mark.requires_postgres
@pytest.mark.asyncio
async def test_chunk_metadata_persistence(real_db_service, real_embedding_service):
    """Test that chunk metadata is correctly stored and retrieved."""
    # Create chunk with rich metadata
    text = "Test document with metadata"
    embedding = await real_embedding_service.embed(text)
    
    metadata = {
        "source": "test.pdf",
        "page": 42,
        "author": "Test Author",
        "tags": ["test", "metadata"],
        "score": 0.95
    }
    
    chunk = Chunk(
        chunk_id="metadata_test",
        text=text,
        embedding=embedding,
        metadata=metadata
    )
    
    await real_db_service.store_chunks([chunk])
    
    # Retrieve and verify metadata
    retrieved = await real_db_service.get_all_chunks()
    metadata_chunk = next(c for c in retrieved if c.chunk_id == "metadata_test")
    
    assert metadata_chunk.metadata["source"] == "test.pdf"
    assert metadata_chunk.metadata["page"] == 42
    assert metadata_chunk.metadata["author"] == "Test Author"
    assert "test" in metadata_chunk.metadata["tags"]
    assert metadata_chunk.metadata["score"] == 0.95


@pytest.mark.real_integration
@pytest.mark.requires_postgres
@pytest.mark.asyncio
async def test_database_context_table_mapping(real_db_service):
    """Test DatabaseContext with custom table/field mappings."""
    # Get a database context with custom table mapping
    context = real_db_service.get_context(
        table_mapping={"chunks": "custom_chunks_test"}
    )
    
    assert context is not None
    assert context.tables["chunks"] == "custom_chunks_test"


@pytest.mark.real_integration
@pytest.mark.requires_postgres
@pytest.mark.asyncio
async def test_connection_pooling(real_db_service):
    """Test that connection pooling works correctly."""
    import asyncio
    
    pool = await real_db_service._get_pool()
    
    async def execute_query(i):
        async with pool.acquire() as conn:
            result = await conn.fetchval(f"SELECT {i} as value")
            return result
    
    # Run 10 concurrent queries
    results = await asyncio.gather(*[execute_query(i) for i in range(10)])
    
    assert results == list(range(10))
