#!/usr/bin/env python3
"""Simple test to check field mappings."""

from rag_factory.services.database.database_context import DatabaseContext
from sqlalchemy import create_engine

# Create engine
engine = create_engine("postgresql://test:test@localhost/test")

# Create context with mappings
table_mapping = {
    "chunks": "agentic_chunks",
    "vectors": "agentic_vectors"
}

field_mapping = {
    "text": "text_content",
    "embedding": "vector_embedding",
    "chunk_id": "chunk_id",
    "document_id": "document_id",
    "chunk_index": "chunk_index",
    "metadata": "metadata",
    "created_at": "created_at",
    "updated_at": "updated_at"
}

print("Creating DatabaseContext...")
context = DatabaseContext(engine, table_mapping, field_mapping)

print(f"Context tables: {context.tables}")
print(f"Context fields: {context.fields}")

print("\nGetting chunk_repository...")
repo = context.chunk_repository

print(f"Repository table_name: {repo.table_name}")
print(f"Repository field_mapping: {repo.field_mapping}")
print(f"Mapped 'text': {repo._map_field('text')}")
print(f"Mapped 'embedding': {repo._map_field('embedding')}")

print("\n✅ Test completed!")
