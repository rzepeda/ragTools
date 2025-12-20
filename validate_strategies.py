#!/usr/bin/env python3
"""
Validate all strategy YAML configurations.

This script validates that all strategy YAML files are correctly configured
and can be loaded with the current services.yaml configuration.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Import strategies to force registration BEFORE importing factory
print("Importing strategies to force registration...")
try:
    from rag_factory.strategies.retrieval.semantic_retriever import SemanticRetriever
    from rag_factory.strategies.retrieval.keyword_retriever import KeywordRetriever
    from rag_factory.strategies.retrieval.knowledge_graph_retriever import KnowledgeGraphRetriever
    from rag_factory.strategies.retrieval.multi_query_retriever import MultiQueryRetriever
    from rag_factory.strategies.retrieval.query_expansion_retriever import QueryExpansionRetriever
    print("✅ Retrieval strategies imported")
except Exception as e:
    print(f"⚠️  Could not import some retrieval strategies: {e}")

from rag_factory.registry.service_registry import ServiceRegistry
from rag_factory.config.strategy_pair_manager import StrategyPairManager

def main():
    print("=" * 70)
    print("Strategy Configuration Validation")
    print("=" * 70)
    
    # Initialize services
    print("\n1. Loading services...")
    try:
        registry = ServiceRegistry("config/services.yaml")
        services = registry.list_services()
        print(f"   ✅ Loaded {len(services)} services: {', '.join(services)}")
    except Exception as e:
        print(f"   ❌ Failed to load services: {e}")
        return 1
    
    # Initialize strategy manager
    print("\n2. Initializing strategy manager...")
    try:
        manager = StrategyPairManager(registry)
        print("   ✅ Strategy manager initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize strategy manager: {e}")
        return 1
    
    # Find all strategy YAML files
    print("\n3. Finding strategy files...")
    strategies_dir = Path("strategies")
    strategy_files = list(strategies_dir.glob("*.yaml"))
    print(f"   ✅ Found {len(strategy_files)} strategy files")
    
    # Validate each strategy
    print("\n4. Validating strategies...")
    print("-" * 70)
    
    results = {
        'passed': [],
        'failed': [],
        'warnings': []
    }
    
    for strategy_file in sorted(strategy_files):
        strategy_name = strategy_file.stem
        print(f"\n   Testing: {strategy_name}")
        
        try:
            # Load YAML
            with open(strategy_file) as f:
                config = yaml.safe_load(f)
            
            # Check required fields
            if 'indexer' not in config:
                results['failed'].append((strategy_name, "Missing 'indexer' section"))
                print(f"      ❌ Missing 'indexer' section")
                continue
            
            if 'retriever' not in config:
                results['failed'].append((strategy_name, "Missing 'retriever' section"))
                print(f"      ❌ Missing 'retriever' section")
                continue
            
            # Try to load the strategy pair
            try:
                indexing, retrieval = manager.load_pair(strategy_name)
                print(f"      ✅ Loaded successfully")
                print(f"         - Indexing: {type(indexing).__name__}")
                print(f"         - Retrieval: {type(retrieval).__name__}")
                results['passed'].append(strategy_name)
            except Exception as e:
                error_msg = str(e)
                results['failed'].append((strategy_name, error_msg))
                print(f"      ❌ Load failed: {error_msg[:100]}")
                
        except Exception as e:
            results['failed'].append((strategy_name, f"YAML error: {e}"))
            print(f"      ❌ YAML error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)
    
    print(f"\n✅ Passed: {len(results['passed'])}")
    for name in results['passed']:
        print(f"   - {name}")
    
    if results['failed']:
        print(f"\n❌ Failed: {len(results['failed'])}")
        for name, error in results['failed']:
            print(f"   - {name}")
            print(f"     Error: {error[:150]}")
    
    if results['warnings']:
        print(f"\n⚠️  Warnings: {len(results['warnings'])}")
        for name, warning in results['warnings']:
            print(f"   - {name}: {warning}")
    
    print("\n" + "=" * 70)
    
    if results['failed']:
        print("❌ VALIDATION FAILED")
        return 1
    else:
        print("✅ ALL STRATEGIES VALIDATED SUCCESSFULLY")
        return 0

if __name__ == "__main__":
    sys.exit(main())
