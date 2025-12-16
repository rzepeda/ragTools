#!/usr/bin/env python3
"""
Script to add @register_strategy decorators to all unregistered strategies.
"""

import os
from pathlib import Path

# Define strategies that need registration
strategies_to_register = [
    {
        "file": "rag_factory/strategies/agentic/strategy.py",
        "class_name": "AgenticRAGStrategy",
        "register_name": "AgenticRAGStrategy",
        "import_line": 12,  # Line number to add import after
    },
    {
        "file": "rag_factory/strategies/self_reflective/strategy.py",
        "class_name": "SelfReflectiveRAGStrategy",
        "register_name": "SelfReflectiveRAGStrategy",
        "import_line": 12,
    },
    {
        "file": "rag_factory/strategies/multi_query/strategy.py",
        "class_name": "MultiQueryRAGStrategy",
        "register_name": "MultiQueryRAGStrategy",
        "import_line": 12,
    },
    {
        "file": "rag_factory/strategies/contextual/strategy.py",
        "class_name": "ContextualRetrievalStrategy",
        "register_name": "ContextualRetrievalStrategy",
        "import_line": 12,
    },
    {
        "file": "rag_factory/strategies/knowledge_graph/strategy.py",
        "class_name": "KnowledgeGraphRAGStrategy",
        "register_name": "KnowledgeGraphRAGStrategy",
        "import_line": 12,
    },
    {
        "file": "rag_factory/strategies/late_chunking/strategy.py",
        "class_name": "LateChunkingStrategy",
        "register_name": "LateChunkingStrategy",
        "import_line": 8,
    },
]

def add_registration(filepath: str, class_name: str, register_name: str):
    """Add @register_strategy decorator to a strategy class."""
    path = Path(filepath)
    
    if not path.exists():
        print(f"❌ File not found: {filepath}")
        return False
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Check if already registered
    content = ''.join(lines)
    if f'@register_strategy("{register_name}")' in content:
        print(f"✅ Already registered: {register_name}")
        return True
    
    # Find the class definition line
    class_line_idx = None
    for i, line in enumerate(lines):
        if f'class {class_name}' in line:
            class_line_idx = i
            break
    
    if class_line_idx is None:
        print(f"❌ Class {class_name} not found in {filepath}")
        return False
    
    # Check if import exists
    has_import = 'from rag_factory.factory import register_strategy' in content or \
                 'from ...factory import register_strategy' in content
    
    # Add import if needed
    if not has_import:
        # Find a good place to add import (after other imports)
        import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('from') or line.startswith('import'):
                import_idx = i + 1
        
        # Determine correct import path based on file location
        if 'strategies/indexing' in filepath:
            import_line = 'from rag_factory.factory import register_strategy\n'
        else:
            import_line = 'from ...factory import register_strategy\n'
        
        lines.insert(import_idx, import_line)
        class_line_idx += 1  # Adjust for inserted line
    
    # Add decorator before class
    indent = len(lines[class_line_idx]) - len(lines[class_line_idx].lstrip())
    decorator = ' ' * indent + f'@register_strategy("{register_name}")\n'
    lines.insert(class_line_idx, decorator)
    
    # Write back
    with open(path, 'w') as f:
        f.writelines(lines)
    
    print(f"✅ Registered: {register_name} in {filepath}")
    return True

def main():
    """Main function."""
    print("🔧 Adding @register_strategy decorators...\n")
    
    success_count = 0
    for strategy in strategies_to_register:
        if add_registration(
            strategy["file"],
            strategy["class_name"],
            strategy["register_name"]
        ):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Successfully registered {success_count}/{len(strategies_to_register)} strategies")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
