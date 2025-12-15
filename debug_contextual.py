"""Debug script to test contextual strategy."""
import asyncio
import logging
from unittest.mock import Mock, AsyncMock
from rag_factory.strategies.contextual.strategy import ContextualRetrievalStrategy
from rag_factory.strategies.contextual.config import ContextualRetrievalConfig
from rag_factory.services.dependencies import StrategyDependencies

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

async def main():
    # Setup mocks
    mock_llm_service = Mock()
    response = Mock()
    response.text = "This chunk discusses machine learning fundamentals."
    mock_llm_service.agenerate = AsyncMock(return_value=response)
    
    mock_embedding_service = Mock()
    result = Mock()
    result.embeddings = [[0.1] * 768]
    mock_embedding_service.embed = Mock(return_value=result)
    
    mock_database = Mock()
    mock_database.store_chunk = Mock()
    
    # Setup strategy
    config = ContextualRetrievalConfig(
        enable_contextualization=True,
        batch_size=10,
        enable_parallel_batches=False  # Use sequential for easier debugging
    )
    
    dependencies = StrategyDependencies(
        database_service=mock_database,
        llm_service=mock_llm_service,
        embedding_service=mock_embedding_service
    )
    
    strategy = ContextualRetrievalStrategy(
        config=config.model_dump(),
        dependencies=dependencies
    )
    
    # Prepare chunks
    chunks = [
        {
            "chunk_id": f"chunk_{i}",
            "document_id": "doc_1",
            "text": f"This is chunk {i} about machine learning concepts.",
            "metadata": {"section_hierarchy": ["Chapter 1", f"Section {i}"]}
        }
        for i in range(5)
    ]
    
    print(f"\n=== Testing with {len(chunks)} chunks ===\n")
    
    # Index document
    try:
        result = await strategy.aindex_document(
            document="Full document text",
            document_id="doc_1",
            chunks=chunks,
            document_metadata={"title": "ML Guide"}
        )
        
        print(f"\n=== RESULT ===")
        print(f"Result: {result}")
        print(f"Total chunks: {result['total_chunks']}")
        print(f"Contextualized chunks: {result['contextualized_chunks']}")
        
    except Exception as e:
        print(f"\n=== ERROR ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
