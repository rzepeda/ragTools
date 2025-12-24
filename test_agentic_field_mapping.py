#!/usr/bin/env python3
"""Test to verify field mappings are working in DatabaseContext."""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, '/mnt/MCPProyects/ragTools')

# Load environment
from dotenv import load_dotenv
load_dotenv('/mnt/MCPProyects/ragTools/.env')

from rag_factory.registry.service_registry import ServiceRegistry
from rag_factory.config.strategy_pair_manager import StrategyPairManager

async def test():
    print("=" * 60)
    print("Testing Field Mappings in Agentic Strategy")
    print("=" * 60)
    
    registry = ServiceRegistry()
    await registry.initialize()
    manager = StrategyPairManager(registry)
    
    print("\n1. Loading agentic-rag-pair strategy...")
    indexer, retriever = await manager.load_strategy_pair('agentic-rag-pair')
    print("✅ Strategy loaded")
    
    print("\n2. Checking database context...")
    db_context = retriever.deps.database_service
    print(f"   Type: {type(db_context).__name__}")
    print(f"   Tables: {db_context.tables}")
    print(f"   Fields: {db_context.fields}")
    
    print("\n3. Checking chunk repository...")
    repo = db_context.chunk_repository
    print(f"   Table name: {repo.table_name}")
    print(f"   Field mapping: {repo.field_mapping}")
    print(f"   Mapped 'text': {repo._map_field('text')}")
    print(f"   Mapped 'embedding': {repo._map_field('embedding')}")
    print(f"   Mapped 'metadata': {repo._map_field('metadata')}")
    
    print("\n4. Testing SQL query generation...")
    # Create a test embedding
    test_embedding = [0.1] * 384
    
    # Build the SQL query like the repository does
    embedding_str = "[" + ",".join(map(str, test_embedding)) + "]"
    
    chunk_id_col = repo._map_field("chunk_id")
    document_id_col = repo._map_field("document_id")
    chunk_index_col = repo._map_field("chunk_index")
    text_col = repo._map_field("text")
    embedding_col = repo._map_field("embedding")
    metadata_col = repo._map_field("metadata")
    created_at_col = repo._map_field("created_at")
    updated_at_col = repo._map_field("updated_at")
    
    print(f"   chunk_id column: {chunk_id_col}")
    print(f"   text column: {text_col}")
    print(f"   embedding column: {embedding_col}")
    print(f"   metadata column: {metadata_col}")
    
    query = f"""
        SELECT {chunk_id_col}, {document_id_col}, {chunk_index_col}, {text_col}, {embedding_col},
               {metadata_col}, {created_at_col}, {updated_at_col},
               1 - ({embedding_col} <=> '{embedding_str[:50]}...'::vector) as similarity
        FROM {repo.table_name}
        WHERE {embedding_col} IS NOT NULL
        LIMIT 5
    """
    
    print("\n5. Generated SQL (first 500 chars):")
    print(query[:500])
    
    print("\n" + "=" * 60)
    print("✅ Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test())
