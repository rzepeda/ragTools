#!/usr/bin/env python3
"""Batch migration script for integration tests.

This script automatically migrates integration test files to use
centralized mock fixtures from conftest.py.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

# Test files and their required fixtures
MIGRATION_MAP = {
    # LLM-dependent tests (need mock_registry_with_llm_services)
    'test_agentic_rag_pair.py': 'mock_registry_with_llm_services',
    'test_query_expansion_pair.py': 'mock_registry_with_llm_services',
    'test_self_reflective_pair.py': 'mock_registry_with_llm_services',
    'test_multi_query_pair.py': 'mock_registry_with_llm_services',
    'test_contextual_retrieval_pair.py': 'mock_registry_with_llm_services',
    
    # Standard tests (need mock_registry_with_services)
    'test_reranking_pair.py': 'mock_registry_with_services',
    'test_hierarchical_rag_pair.py': 'mock_registry_with_services',
    'test_context_aware_chunking_pair.py': 'mock_registry_with_services',
    'test_late_chunking_pair.py': 'mock_registry_with_services',
    'test_hybrid_search_pair.py': 'mock_registry_with_services',
    'test_semantic_api_pair.py': 'mock_registry_with_services',
    'test_fine_tuned_embeddings_pair.py': 'mock_registry_with_services',
    'test_keyword_pair.py': 'mock_registry_with_services',
}

def migrate_file(filepath: Path, fixture_name: str) -> Tuple[bool, str, int]:
    """Migrate a single test file to use centralized fixtures.
    
    Args:
        filepath: Path to test file
        fixture_name: Name of centralized fixture to use
        
    Returns:
        Tuple of (success, message, lines_removed)
    """
    try:
        content = filepath.read_text()
        original_lines = len(content.split('\n'))
        
        # Step 1: Remove unnecessary imports
        content = re.sub(
            r'from unittest\.mock import Mock, AsyncMock, patch',
            'from unittest.mock import patch',
            content
        )
        content = re.sub(
            r'from rag_factory\.registry\.service_registry import ServiceRegistry\n',
            '',
            content
        )
        content = re.sub(
            r'from rag_factory\.services\.dependencies import ServiceDependency\n',
            '',
            content
        )
        
        # Step 2: Remove local mock_registry fixture
        # Find and remove the entire fixture definition
        fixture_pattern = r'@pytest\.fixture\s+def\s+mock_registry\(\):.*?(?=\n@pytest|\nclass\s|\nasync def test_|\ndef test_|\Z)'
        fixture_match = re.search(fixture_pattern, content, re.DOTALL)
        
        lines_removed = 0
        if fixture_match:
            fixture_text = fixture_match.group()
            lines_removed = len(fixture_text.split('\n'))
            
            # Replace fixture with comment
            replacement = f'# Note: Using centralized {fixture_name} fixture from conftest.py\n'
            content = content.replace(fixture_text, replacement)
        
        # Step 3: Update test function signatures
        content = re.sub(
            r'async def (test_\w+)\(mock_registry\)',
            rf'async def \1({fixture_name})',
            content
        )
        
        # Step 4: Update fixture usage in test body
        content = re.sub(
            r'service_registry=mock_registry,',
            rf'service_registry={fixture_name},',
            content
        )
        
        # Step 5: Clean up extra blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Write back
        filepath.write_text(content)
        
        new_lines = len(content.split('\n'))
        actual_removed = original_lines - new_lines
        
        return True, f"✅ Migrated successfully ({actual_removed} lines removed)", actual_removed
        
    except Exception as e:
        return False, f"❌ Error: {str(e)}", 0


def main():
    """Run batch migration."""
    test_dir = Path('/mnt/MCPProyects/ragTools/tests/integration')
    
    print("="*80)
    print("Batch Migration of Integration Tests")
    print("="*80)
    print()
    
    total_removed = 0
    successful = 0
    failed = 0
    
    for filename, fixture_name in MIGRATION_MAP.items():
        filepath = test_dir / filename
        
        if not filepath.exists():
            print(f"⚠️  SKIP: {filename} (file not found)")
            continue
        
        print(f"Migrating {filename}...")
        success, message, lines_removed = migrate_file(filepath, fixture_name)
        
        print(f"  {message}")
        print(f"  Fixture: {fixture_name}")
        print()
        
        if success:
            successful += 1
            total_removed += lines_removed
        else:
            failed += 1
    
    print("="*80)
    print("Migration Summary")
    print("="*80)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed:     {failed}")
    print(f"📊 Total lines removed: {total_removed}")
    print("="*80)
    
    return successful, failed, total_removed


if __name__ == "__main__":
    main()
