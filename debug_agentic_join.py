#!/usr/bin/env python
"""Test script to debug agentic strategy multi-table JOIN."""

import asyncio
import os
os.environ['DATABASE_URL'] = os.getenv('DATABASE_URL', 'postgresql://admindevmac:Passw0rd!@localhost:5432/rag_factory')

from rag_factory.registry.service_registry import ServiceRegistry
from rag_factory.config.strategy_pair_manager import StrategyPairManager
from sqlalchemy import text

async def main():
    print("=" * 60)
    print("Testing Agentic Strategy Multi-Table JOIN")
    print("=" * 60)
    
    # Initialize
    registry = ServiceRegistry()
    await registry.initialize()
    manager = StrategyPairManager(registry)
    
    # Load agentic strategy
    print("\n1. Loading agentic-rag-pair strategy...")
    indexer, retriever = manager.load_pair('agentic-rag-pair')
    
    # Get repository
    chunk_repo = retriever.deps.chunk_repository
    print(f"   Table name: {chunk_repo.table_name}")
    print(f"   Field mapping: {chunk_repo.field_mapping}")
    
    # Check data in tables
    print("\n2. Checking data in tables...")
    session = chunk_repo.session
    
    chunks_count = session.execute(text("SELECT COUNT(*) FROM agentic_chunks")).scalar()
    print(f"   agentic_chunks: {chunks_count} rows")
    
    vectors_count = session.execute(text("SELECT COUNT(*) FROM agentic_vectors")).scalar()
    print(f"   agentic_vectors: {vectors_count} rows")
    
    if chunks_count > 0:
        sample_chunk = session.execute(text("SELECT chunk_id, document_id, LEFT(text_content, 50) FROM agentic_chunks LIMIT 1")).fetchone()
        print(f"   Sample chunk: {sample_chunk}")
    
    if vectors_count > 0:
        sample_vector = session.execute(text("SELECT chunk_id, vector_embedding IS NOT NULL as has_embedding FROM agentic_vectors LIMIT 1")).fetchone()
        print(f"   Sample vector: {sample_vector}")
    
    # Test JOIN
    print("\n3. Testing JOIN query...")
    join_result = session.execute(text("""
        SELECT c.chunk_id, c.document_id, v.chunk_id as v_chunk_id, v.vector_embedding IS NOT NULL as has_embedding
        FROM agentic_chunks c
        LEFT JOIN agentic_vectors v ON c.chunk_id = v.chunk_id
        LIMIT 1
    """)).fetchone()
    print(f"   JOIN result: {join_result}")
    
    # Test search_similar
    print("\n4. Testing search_similar...")
    try:
        # Get embedding service
        embedding_service = retriever.deps.embedding_service
        test_query = "Voyager 1"
        embedding = await embedding_service.embed(test_query)
        print(f"   Generated embedding for '{test_query}': {len(embedding)} dimensions")
        
        # Search
        results = chunk_repo.search_similar(embedding=embedding, top_k=5, threshold=0.0)
        print(f"   Search results: {len(results)} chunks found")
        
        for chunk, score in results:
            print(f"     - {chunk.chunk_id}: score={score:.3f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    await registry.close()
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
