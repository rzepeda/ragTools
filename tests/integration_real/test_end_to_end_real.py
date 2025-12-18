"""
Real end-to-end integration tests for complete RAG pipeline.

Tests complete RAG workflows with real services configured via .env.
"""

import pytest
from rag_factory.repositories.chunk import Chunk


@pytest.mark.real_integration
@pytest.mark.requires_postgres
@pytest.mark.requires_embeddings
@pytest.mark.asyncio
async def test_document_indexing_pipeline(real_db_service, real_embedding_service, sample_documents):
    """Test complete document indexing pipeline."""
    from rag_factory.strategies.indexing.vector_embedding import VectorEmbeddingIndexing
    from rag_factory.strategies.base import StrategyConfig
    from rag_factory.services.dependencies import StrategyDependencies
    from rag_factory.core.indexing_interface import IndexingContext
    from dataclasses import asdict
    
    # Create indexing strategy
    config = StrategyConfig(
        strategy_name="test_indexing",
        chunk_size=200,
        chunk_overlap=50
    )
    
    dependencies = StrategyDependencies(
        embedding_service=real_embedding_service,
        database_service=real_db_service
    )
    
    strategy = VectorEmbeddingIndexing(config=asdict(config), dependencies=dependencies)
    
    # Create indexing context
    context = IndexingContext(database_service=real_db_service, config=asdict(config))
    
    # Index documents
    await strategy.process(sample_documents, context)
    
    # Verify chunks were stored
    chunks = await real_db_service.get_all_chunks()
    assert len(chunks) > 0
    
    # Verify chunks have embeddings
    for chunk in chunks:
        assert chunk.embedding is not None
        assert len(chunk.embedding) > 0


@pytest.mark.real_integration
@pytest.mark.requires_postgres
@pytest.mark.requires_embeddings
@pytest.mark.asyncio
async def test_retrieval_pipeline(real_db_service, real_embedding_service):
    """Test complete retrieval pipeline."""
    from rag_factory.strategies.retrieval.semantic_retriever import SemanticRetriever
    from rag_factory.strategies.base import StrategyConfig
    from rag_factory.services.dependencies import StrategyDependencies
    from rag_factory.core.retrieval_interface import RetrievalContext
    
    # First, index some documents
    texts = [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "The Eiffel Tower is located in Paris, France."
    ]
    
    chunks = []
    for i, text in enumerate(texts):
        embedding = await real_embedding_service.embed(text)
        chunk = Chunk(
            chunk_id=f"retrieval_test_{i}",
            text=text,
            embedding=embedding,
            metadata={"source": f"doc{i}.txt"}
        )
        chunks.append(chunk)
    
    await real_db_service.store_chunks(chunks)
    
    # Create retrieval strategy
    config = StrategyConfig(
        strategy_name="test_retrieval",
        top_k=2
    )
    
    dependencies = StrategyDependencies(
        embedding_service=real_embedding_service,
        database_service=real_db_service
    )
    
    strategy = SemanticRetriever(config=config, dependencies=dependencies)
    
    # Create retrieval context
    retrieval_context = RetrievalContext(database_service=real_db_service, config={})
    
    # Retrieve relevant chunks
    query = "What is Python?"
    results = await strategy.retrieve(query, retrieval_context)
    
    assert len(results) > 0
    assert len(results) <= 2  # top_k=2
    
    # First result should be about Python
    assert "python" in results[0].text.lower()


@pytest.mark.real_integration
@pytest.mark.requires_postgres
@pytest.mark.requires_embeddings
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_full_rag_pipeline(real_db_service, real_embedding_service, real_llm_service):
    """Test complete RAG pipeline: indexing -> retrieval -> generation."""
    from rag_factory.strategies.indexing.vector_embedding import VectorEmbeddingIndexing
    from rag_factory.strategies.retrieval.semantic_retriever import SemanticRetriever
    from rag_factory.strategies.base import StrategyConfig
    from rag_factory.services.dependencies import StrategyDependencies
    from rag_factory.services.llm.base import Message, MessageRole
    from rag_factory.core.indexing_interface import IndexingContext
    from dataclasses import asdict
    
    # Step 1: Index documents
    documents = [
        {
            "text": "The Eiffel Tower was built in 1889 by Gustave Eiffel. It is located in Paris, France and stands 330 meters tall.",
            "id": "eiffel_doc",
            "metadata": {"source": "eiffel.txt"}
        },
        {
            "text": "Python is a high-level programming language created by Guido van Rossum in 1991. It is known for its simplicity and readability.",
            "id": "python_doc",
            "metadata": {"source": "python.txt"}
        }
    ]
    
    indexing_config = StrategyConfig(strategy_name="indexing", chunk_size=200, chunk_overlap=50)
    indexing_deps = StrategyDependencies(
        embedding_service=real_embedding_service,
        database_service=real_db_service
    )
    indexing_strategy = VectorEmbeddingIndexing(config=asdict(indexing_config), dependencies=indexing_deps)
    
    # Create indexing context
    indexing_context = IndexingContext(database_service=real_db_service, config=asdict(indexing_config))
    
    await indexing_strategy.process(documents, indexing_context)
    
    # Step 2: Retrieve relevant chunks
    retrieval_config = StrategyConfig(strategy_name="retrieval", top_k=2)
    retrieval_deps = StrategyDependencies(
        embedding_service=real_embedding_service,
        database_service=real_db_service
    )
    retrieval_strategy = SemanticRetriever(config=retrieval_config, dependencies=retrieval_deps)
    
    # Create retrieval context
    from rag_factory.core.retrieval_interface import RetrievalContext
    retrieval_context = RetrievalContext(database_service=real_db_service, config={})
    
    query = "When was the Eiffel Tower built?"
    retrieved_chunks = await retrieval_strategy.retrieve(query, retrieval_context)
    
    assert len(retrieved_chunks) > 0
    
    # Step 3: Generate answer using LLM
    context = "\n\n".join([chunk.text for chunk in retrieved_chunks])
    
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="Answer the question based only on the provided context. Be concise."
        ),
        Message(
            role=MessageRole.USER,
            content=f"Context:\n{context}\n\nQuestion: {query}"
        )
    ]
    
    response = real_llm_service.complete(messages)
    
    assert response is not None
    assert "1889" in response.content


@pytest.mark.real_integration
@pytest.mark.requires_postgres
@pytest.mark.requires_embeddings
@pytest.mark.asyncio
async def test_multiple_document_batches(real_db_service, real_embedding_service):
    """Test indexing multiple batches of documents."""
    from rag_factory.strategies.indexing.vector_embedding import VectorEmbeddingIndexing
    from rag_factory.strategies.base import StrategyConfig
    from rag_factory.services.dependencies import StrategyDependencies
    from rag_factory.core.indexing_interface import IndexingContext
    from dataclasses import asdict
    
    config = StrategyConfig(strategy_name="batch_test", chunk_size=100, chunk_overlap=20)
    dependencies = StrategyDependencies(
        embedding_service=real_embedding_service,
        database_service=real_db_service
    )
    strategy = VectorEmbeddingIndexing(config=asdict(config), dependencies=dependencies)
    
    # Create indexing context
    context = IndexingContext(database_service=real_db_service, config=asdict(config))
    
    # Index first batch
    batch1 = [
        {"text": f"Document {i} from batch 1", "id": f"batch1_doc{i}", "metadata": {"batch": 1}}
        for i in range(5)
    ]
    await strategy.process(batch1, context)
    
    # Index second batch
    batch2 = [
        {"text": f"Document {i} from batch 2", "id": f"batch2_doc{i}", "metadata": {"batch": 2}}
        for i in range(5)
    ]
    await strategy.process(batch2, context)
    
    # Verify all chunks are stored
    chunks = await real_db_service.get_all_chunks()
    batch1_chunks = [c for c in chunks if c.metadata.get("batch") == 1]
    batch2_chunks = [c for c in chunks if c.metadata.get("batch") == 2]
    
    assert len(batch1_chunks) >= 5
    assert len(batch2_chunks) >= 5


@pytest.mark.real_integration
@pytest.mark.requires_postgres
@pytest.mark.requires_embeddings
@pytest.mark.asyncio
async def test_retrieval_with_metadata_filtering(real_db_service, real_embedding_service):
    """Test retrieval with metadata filtering."""
    # Store chunks with different categories
    categories = ["science", "history", "technology"]
    
    for category in categories:
        for i in range(3):
            text = f"This is a {category} document number {i}"
            embedding = await real_embedding_service.embed(text)
            chunk = Chunk(
                chunk_id=f"{category}_{i}",
                text=text,
                embedding=embedding,
                metadata={"category": category, "index": i}
            )
            await real_db_service.store_chunks([chunk])
    
    # Retrieve all chunks
    all_chunks = await real_db_service.get_all_chunks()
    
    # Filter by category
    science_chunks = [c for c in all_chunks if c.metadata.get("category") == "science"]
    assert len(science_chunks) >= 3


@pytest.mark.real_integration
@pytest.mark.requires_postgres
@pytest.mark.requires_embeddings
@pytest.mark.slow
@pytest.mark.asyncio
async def test_large_document_indexing(real_db_service, real_embedding_service):
    """Test indexing a large document."""
    from rag_factory.strategies.indexing.vector_embedding import VectorEmbeddingIndexing
    from rag_factory.strategies.base import StrategyConfig
    from rag_factory.services.dependencies import StrategyDependencies
    from rag_factory.core.indexing_interface import IndexingContext
    from dataclasses import asdict
    
    # Create a large document
    large_text = " ".join([f"This is sentence number {i}." for i in range(1000)])
    
    document = {
        "text": large_text,
        "id": "large_doc",
        "metadata": {"source": "large_doc.txt", "size": "large"}
    }
    
    config = StrategyConfig(strategy_name="large_doc_test", chunk_size=200, chunk_overlap=50)
    dependencies = StrategyDependencies(
        embedding_service=real_embedding_service,
        database_service=real_db_service
    )
    strategy = VectorEmbeddingIndexing(config=asdict(config), dependencies=dependencies)
    
    # Create indexing context
    context = IndexingContext(database_service=real_db_service, config=asdict(config))
    
    # Index the large document
    await strategy.process([document], context)
    
    # Verify chunks were created
    chunks = await real_db_service.get_all_chunks()
    large_doc_chunks = [c for c in chunks if c.metadata.get("size") == "large"]
    
    # Should have created multiple chunks
    assert len(large_doc_chunks) > 5


@pytest.mark.real_integration
@pytest.mark.requires_postgres
@pytest.mark.requires_embeddings
@pytest.mark.asyncio
async def test_retrieval_accuracy(real_db_service, real_embedding_service):
    """Test retrieval accuracy with known relevant documents."""
    # Create documents with clear semantic relationships
    documents_data = [
        ("Dogs are domesticated animals that are often kept as pets.", "animals"),
        ("Cats are independent animals that enjoy hunting.", "animals"),
        ("Python is a popular programming language.", "programming"),
        ("JavaScript is used for web development.", "programming"),
        ("The sun is a star at the center of our solar system.", "astronomy"),
    ]
    
    chunks = []
    for i, (text, category) in enumerate(documents_data):
        embedding = await real_embedding_service.embed(text)
        chunk = Chunk(
            chunk_id=f"accuracy_test_{i}",
            text=text,
            embedding=embedding,
            metadata={"category": category}
        )
        chunks.append(chunk)
    
    await real_db_service.store_chunks(chunks)
    
    # Test retrieval with animal-related query
    query = "What pets do people keep?"
    query_embedding = await real_embedding_service.embed(query)
    results = await real_db_service.search_chunks(query_embedding, top_k=2)
    
    # Top results should be about animals
    assert len(results) == 2
    categories = [r.metadata.get("category") for r in results]
    assert "animals" in categories
