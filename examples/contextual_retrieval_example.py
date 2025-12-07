"""
Example usage of Contextual Retrieval Strategy.

This example demonstrates how to use the contextual retrieval strategy
to enrich chunks with document context for improved retrieval.
"""

import asyncio
from rag_factory.strategies.contextual import (
    ContextualRetrievalStrategy,
    ContextualRetrievalConfig
)


async def main():
    """Main example function."""
    
    # Mock services (replace with real implementations)
    from unittest.mock import Mock, AsyncMock
    
    # Setup mock services
    vector_store = Mock()
    vector_store.index_chunk = Mock()
    vector_store.search = Mock(return_value=[
        {"chunk_id": "chunk_0", "score": 0.9},
        {"chunk_id": "chunk_1", "score": 0.85},
    ])
    
    database = Mock()
    database.store_chunk = Mock()
    database.get_chunks_by_ids = Mock(return_value=[
        {
            "chunk_id": "chunk_0",
            "original_text": "Machine learning is a subset of AI.",
            "context_description": "Introduction to ML concepts",
            "contextualized_text": "Context: Introduction to ML concepts\n\nMachine learning is a subset of AI."
        },
        {
            "chunk_id": "chunk_1",
            "original_text": "Neural networks are inspired by the brain.",
            "context_description": "Neural network fundamentals",
            "contextualized_text": "Context: Neural network fundamentals\n\nNeural networks are inspired by the brain."
        }
    ])
    
    llm_service = Mock()
    response = Mock()
    response.text = "This chunk discusses machine learning fundamentals in an AI tutorial."
    llm_service.agenerate = AsyncMock(return_value=response)
    
    embedding_service = Mock()
    result = Mock()
    result.embeddings = [[0.1] * 768]
    embedding_service.embed = Mock(return_value=result)
    
    # Configure contextual retrieval strategy
    config = ContextualRetrievalConfig(
        enable_contextualization=True,
        batch_size=20,
        context_length_min=20,
        context_length_max=150,
        enable_parallel_batches=True,
        max_concurrent_batches=5,
        enable_cost_tracking=True,
        return_original_text=True
    )
    
    # Initialize strategy
    strategy = ContextualRetrievalStrategy(
        vector_store_service=vector_store,
        database_service=database,
        llm_service=llm_service,
        embedding_service=embedding_service,
        config=config
    )
    
    # Prepare document chunks
    chunks = [
        {
            "chunk_id": f"chunk_{i}",
            "document_id": "ml_guide",
            "text": f"This is chunk {i} about machine learning concepts. " * 10,
            "metadata": {
                "section_hierarchy": ["Chapter 1", f"Section {i}"],
                "page": i + 1
            }
        }
        for i in range(50)
    ]
    
    # Index document with contextualization
    print("Indexing document with contextual enrichment...")
    result = await strategy.aindex_document(
        document="Full ML guide document text",
        document_id="ml_guide",
        chunks=chunks,
        document_metadata={"title": "Machine Learning Guide"}
    )
    
    print(f"\nIndexing Results:")
    print(f"  Total chunks: {result['total_chunks']}")
    print(f"  Contextualized chunks: {result['contextualized_chunks']}")
    print(f"  Total cost: ${result['total_cost']:.4f}")
    print(f"  Total tokens: {result['total_tokens']}")
    print(f"  Average cost per chunk: ${result['avg_cost_per_chunk']:.6f}")
    
    # Retrieve with contextualized embeddings
    print("\nRetrieving chunks...")
    results = strategy.retrieve("machine learning fundamentals", top_k=5)
    
    print(f"\nRetrieved {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Chunk ID: {result.get('chunk_id')}")
        print(f"   Score: {result.get('score', 'N/A')}")
        print(f"   Text: {result.get('text', '')[:100]}...")
        if 'context' in result:
            print(f"   Context: {result['context']}")
    
    # Get cost summary
    cost_summary = strategy.get_cost_summary()
    print(f"\nFinal Cost Summary:")
    print(f"  Total chunks processed: {cost_summary['total_chunks']}")
    print(f"  Total cost: ${cost_summary['total_cost']:.4f}")
    print(f"  Cost per 1K chunks: ${cost_summary['cost_per_1k_chunks']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
