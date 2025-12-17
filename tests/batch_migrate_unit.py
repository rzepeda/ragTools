#!/usr/bin/env python3
"""Batch migration script for remaining high-impact unit tests.

This script migrates the remaining 6-8 high-impact unit test files
to use centralized mock fixtures.
"""

import re
from pathlib import Path
from typing import Tuple

# Files to migrate with their specific patterns
MIGRATIONS = [
    {
        'file': 'tests/unit/core/test_indexing_interface.py',
        'description': 'Core indexing interface tests',
        'replacements': [
            # Import centralized mocks
            ('from unittest.mock import Mock, AsyncMock, MagicMock',
             'from unittest.mock import Mock, AsyncMock\nfrom tests.mocks import create_mock_indexing_strategy, create_mock_database_service, create_mock_embedding_service'),
            # Simplify mock creation in tests
            ('strategy = Mock\\(\\)', 'strategy = create_mock_indexing_strategy()'),
        ]
    },
    {
        'file': 'tests/unit/core/test_retrieval_interface.py',
        'description': 'Core retrieval interface tests',
        'replacements': [
            ('from unittest.mock import Mock, AsyncMock, MagicMock',
             'from unittest.mock import Mock, AsyncMock\nfrom tests.mocks import create_mock_retrieval_strategy, create_mock_database_service, create_mock_embedding_service'),
        ]
    },
    {
        'file': 'tests/unit/repositories/test_chunk_repository.py',
        'description': 'Chunk repository tests',
        'replacements': [
            ('from unittest.mock import Mock, MagicMock, patch',
             'from unittest.mock import Mock, patch\nfrom tests.mocks import create_mock_session, create_mock_database_service'),
        ]
    },
    {
        'file': 'tests/unit/repositories/test_document_repository.py',
        'description': 'Document repository tests',
        'replacements': [
            ('from unittest.mock import Mock, MagicMock, patch',
             'from unittest.mock import Mock, patch\nfrom tests.mocks import create_mock_session, create_mock_database_service'),
        ]
    },
    {
        'file': 'tests/unit/cli/test_check_consistency_command.py',
        'description': 'CLI consistency check command tests',
        'replacements': [
            ('from unittest.mock import Mock, patch, MagicMock',
             'from unittest.mock import Mock, patch\nfrom tests.mocks import create_mock_indexing_strategy, create_mock_retrieval_strategy'),
        ]
    },
    {
        'file': 'tests/unit/services/database/test_migration_validator.py',
        'description': 'Migration validator tests',
        'replacements': [
            ('from unittest.mock import Mock, MagicMock, patch',
             'from unittest.mock import Mock, patch\nfrom tests.mocks import create_mock_engine, create_mock_connection, create_mock_migration_validator'),
        ]
    },
]


def migrate_file(file_path: str, replacements: list) -> Tuple[bool, str, int]:
    """Migrate a single file.
    
    Args:
        file_path: Path to file
        replacements: List of (pattern, replacement) tuples
        
    Returns:
        (success, message, lines_changed)
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {file_path}", 0
        
        content = path.read_text()
        original_lines = len(content.split('\n'))
        
        # Apply replacements
        modified = False
        for pattern, replacement in replacements:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True
        
        if modified:
            # Clean up extra imports
            content = re.sub(r'from unittest\.mock import.*MagicMock', 
                           lambda m: m.group(0).replace(', MagicMock', '').replace('MagicMock, ', ''),
                           content)
            
            # Write back
            path.write_text(content)
            new_lines = len(content.split('\n'))
            lines_changed = original_lines - new_lines
            
            return True, f"✅ Migrated ({lines_changed:+d} lines)", lines_changed
        else:
            return True, "ℹ️  No changes needed", 0
            
    except Exception as e:
        return False, f"❌ Error: {str(e)}", 0


def main():
    """Run batch migration."""
    print("="*80)
    print("Batch Migration of High-Impact Unit Tests (Phase 3A)")
    print("="*80)
    print()
    
    total_changed = 0
    successful = 0
    failed = 0
    
    for migration in MIGRATIONS:
        file_path = migration['file']
        description = migration['description']
        replacements = migration['replacements']
        
        print(f"Migrating: {description}")
        print(f"  File: {Path(file_path).name}")
        
        success, message, lines_changed = migrate_file(file_path, replacements)
        print(f"  {message}")
        print()
        
        if success:
            successful += 1
            total_changed += lines_changed
        else:
            failed += 1
    
    print("="*80)
    print("Migration Summary")
    print("="*80)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed:     {failed}")
    print(f"📊 Total lines changed: {total_changed:+d}")
    print("="*80)
    
    return successful, failed, total_changed


if __name__ == "__main__":
    main()
