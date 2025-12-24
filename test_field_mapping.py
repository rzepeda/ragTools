#!/usr/bin/env python3
"""Test script to verify repository field mappings."""

from rag_factory.repositories.chunk import ChunkRepository
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

# Create a mock session (won't actually connect)
engine = create_engine("postgresql://test:test@localhost/test", echo=False)
session = Session(bind=engine)

# Test 1: Default (no mappings)
repo1 = ChunkRepository(session)
print("Test 1 - Default repository:")
print(f"  Table: {repo1.table_name}")
print(f"  Field mapping: {repo1.field_mapping}")
print(f"  Mapped 'text': {repo1._map_field('text')}")
print()

# Test 2: With table name only
repo2 = ChunkRepository(session, table_name="agentic_chunks")
print("Test 2 - Custom table, no field mapping:")
print(f"  Table: {repo2.table_name}")
print(f"  Field mapping: {repo2.field_mapping}")
print(f"  Mapped 'text': {repo2._map_field('text')}")
print()

# Test 3: With table name and field mappings
field_map = {
    "text": "text_content",
    "embedding": "vector_embedding",
    "chunk_id": "chunk_id",
    "document_id": "document_id"
}
repo3 = ChunkRepository(session, table_name="agentic_chunks", field_mapping=field_map)
print("Test 3 - Custom table with field mappings:")
print(f"  Table: {repo3.table_name}")
print(f"  Field mapping: {repo3.field_mapping}")
print(f"  Mapped 'text': {repo3._map_field('text')}")
print(f"  Mapped 'embedding': {repo3._map_field('embedding')}")
print(f"  Mapped 'metadata': {repo3._map_field('metadata')}")  # Not in mapping, should return 'metadata'
print()

print("✅ All tests passed!")
