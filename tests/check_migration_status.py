#!/usr/bin/env python3
"""Script to help migrate integration tests to use centralized mocks.

This script identifies integration test files that still have local mock_registry
fixtures and can be migrated to use centralized fixtures.
"""

import os
import re
from pathlib import Path
from collections import defaultdict

# Test directory
test_dir = Path("/mnt/MCPProyects/ragTools/tests/integration")

# Track migration status
migration_status = {
    'migrated': [],
    'needs_migration': [],
    'no_mocks': []
}

# Patterns to identify
patterns = {
    'local_fixture': r'@pytest\.fixture\s+def\s+mock_registry',
    'uses_mock': r'from unittest\.mock import.*Mock',
    'uses_centralized': r'mock_registry_with_services|mock_registry_with_llm_services|mock_registry_with_graph_services'
}

print("="*80)
print("Integration Test Migration Status")
print("="*80)
print()

for test_file in sorted(test_dir.glob("test_*.py")):
    try:
        content = test_file.read_text()
        
        has_local_fixture = bool(re.search(patterns['local_fixture'], content))
        uses_mock = bool(re.search(patterns['uses_mock'], content))
        uses_centralized = bool(re.search(patterns['uses_centralized'], content))
        
        if uses_centralized:
            migration_status['migrated'].append(test_file.name)
            status = "✅ MIGRATED"
        elif has_local_fixture:
            migration_status['needs_migration'].append(test_file.name)
            status = "🔄 NEEDS MIGRATION"
            
            # Count lines in fixture
            fixture_match = re.search(
                r'@pytest\.fixture\s+def\s+mock_registry.*?(?=\n@|\nclass\s|\ndef\s|async def\s|\Z)',
                content,
                re.DOTALL
            )
            if fixture_match:
                fixture_lines = len(fixture_match.group().split('\n'))
                status += f" ({fixture_lines} lines to remove)"
        elif uses_mock:
            migration_status['needs_migration'].append(test_file.name)
            status = "⚠️  HAS MOCKS (check manually)"
        else:
            migration_status['no_mocks'].append(test_file.name)
            status = "ℹ️  NO MOCKS"
        
        print(f"{status:40} {test_file.name}")
        
    except Exception as e:
        print(f"❌ ERROR: {test_file.name} - {e}")

print()
print("="*80)
print("Summary")
print("="*80)
print(f"✅ Migrated:        {len(migration_status['migrated'])}")
print(f"🔄 Needs Migration: {len(migration_status['needs_migration'])}")
print(f"ℹ️  No Mocks:        {len(migration_status['no_mocks'])}")
print()

if migration_status['needs_migration']:
    print("Files that need migration:")
    for filename in migration_status['needs_migration']:
        print(f"  - {filename}")
    print()

print(f"Migration Progress: {len(migration_status['migrated'])}/{len(migration_status['migrated']) + len(migration_status['needs_migration'])} files")
percentage = (len(migration_status['migrated']) / max(1, len(migration_status['migrated']) + len(migration_status['needs_migration']))) * 100
print(f"Completion: {percentage:.1f}%")
print("="*80)
