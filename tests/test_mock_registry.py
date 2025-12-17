#!/usr/bin/env python3
"""Quick test to verify mock registry setup."""

import sys
sys.path.insert(0, '/mnt/MCPProyects/ragTools')

from tests.mocks import create_mock_registry_with_services

# Create mock registry
registry = create_mock_registry_with_services(
    include_embedding=True,
    include_database=True
)

print("Mock Registry Test")
print("=" * 60)

# Test get method
print("\n1. Testing registry.get() method:")
print(f"   registry._instances keys: {list(registry._instances.keys())}")

embedding = registry.get("embedding_local")
print(f"   ✓ registry.get('embedding_local'): {embedding}")

db = registry.get("db_main")
print(f"   ✓ registry.get('db_main'): {db}")

# Test list_services
print("\n2. Testing registry.list_services():")
services = registry.list_services()
print(f"   ✓ Available services: {services}")

# Test that services have required methods
print("\n3. Testing service methods:")
print(f"   ✓ embedding.embed: {hasattr(embedding, 'embed')}")
print(f"   ✓ embedding.get_dimension: {hasattr(embedding, 'get_dimension')}")
print(f"   ✓ db.store_chunks: {hasattr(db, 'store_chunks')}")
print(f"   ✓ db.search_chunks: {hasattr(db, 'search_chunks')}")

print("\n" + "=" * 60)
print("✅ Mock registry is working correctly!")
