"""Quick test to see if strategies can be imported and instantiated"""
import sys
import asyncio
from unittest.mock import Mock, AsyncMock

print("Step 1: Importing strategy classes...")
try:
    from rag_factory.strategies.indexing.vector_embedding import VectorEmbeddingIndexing
    from rag_factory.strategies.retrieval.semantic_retriever import SemanticRetriever
    print("✅ Strategy imports successful")
except Exception as e:
    print(f"❌ Strategy import failed: {e}")
    sys.exit(1)

print("\nStep 2: Importing dependencies...")
try:
    from rag_factory.services.dependencies import StrategyDependencies
    from rag_factory.core.indexing_interface import IndexingContext
    from rag_factory.core.retrieval_interface import RetrievalContext
    print("✅ Dependencies imported")
except Exception as e:
    print(f"❌ Dependencies import failed: {e}")
    sys.exit(1)

print("\nStep 3: Creating mock services...")
try:
    # Mock services
    embedding_service = Mock()
    embedding_service.embed = AsyncMock(return_value=[0.1] * 384)
    embedding_service.embed_batch = AsyncMock(return_value=[[0.1] * 384])
    embedding_service.get_dimension = Mock(return_value=384)
    
    db_service = Mock()
    db_service.store_chunks = AsyncMock()
    db_service.search_chunks = AsyncMock(return_value=[
        {'id': 'chunk1', 'text': 'test content', 'score': 0.9}
    ])
    
    print("✅ Mock services created")
except Exception as e:
    print(f"❌ Mock creation failed: {e}")
    sys.exit(1)

print("\nStep 4: Creating strategy dependencies...")
try:
    deps = StrategyDependencies(
        embedding_service=embedding_service,
        database_service=db_service
    )
    print("✅ Dependencies created")
except Exception as e:
    print(f"❌ Dependencies creation failed: {e}")
    sys.exit(1)

print("\nStep 5: Instantiating indexing strategy...")
try:
    indexing = VectorEmbeddingIndexing(
        config={'chunk_size': 512, 'overlap': 50},
        dependencies=deps
    )
    print(f"✅ Indexing strategy created: {type(indexing).__name__}")
except Exception as e:
    print(f"❌ Indexing strategy creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 6: Instantiating retrieval strategy...")
try:
    retrieval = SemanticRetriever(
        config={'top_k': 5},
        dependencies=deps
    )
    print(f"✅ Retrieval strategy created: {type(retrieval).__name__}")
except Exception as e:
    print(f"❌ Retrieval strategy creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ ALL TESTS PASSED - Strategies can be imported and instantiated!")
