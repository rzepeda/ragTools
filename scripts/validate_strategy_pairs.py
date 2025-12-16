#!/usr/bin/env python3
"""
Validate all strategy pair YAML configurations.
This script checks that all YAML files in the strategies directory are valid.
"""

import yaml
import sys
from pathlib import Path
from typing import List, Tuple

def validate_yaml_file(filepath: Path) -> Tuple[bool, str]:
    """
    Validate a single YAML file.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required top-level keys
        required_keys = ['strategy_name', 'version', 'description', 'indexer', 'retriever']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            return False, f"Missing required keys: {missing_keys}"
        
        # Check indexer structure
        if 'strategy' not in config['indexer']:
            return False, "Indexer missing 'strategy' key"
        
        if 'services' not in config['indexer']:
            return False, "Indexer missing 'services' key"
        
        # Check retriever structure
        if 'strategy' not in config['retriever']:
            return False, "Retriever missing 'strategy' key"
        
        if 'services' not in config['retriever']:
            return False, "Retriever missing 'services' key"
        
        return True, "Valid"
        
    except yaml.YAMLError as e:
        return False, f"YAML parsing error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main validation function."""
    strategies_dir = Path(__file__).parent.parent / "strategies"
    
    if not strategies_dir.exists():
        print(f"❌ Strategies directory not found: {strategies_dir}")
        sys.exit(1)
    
    yaml_files = list(strategies_dir.glob("*.yaml"))
    
    if not yaml_files:
        print(f"❌ No YAML files found in {strategies_dir}")
        sys.exit(1)
    
    print(f"🔍 Validating {len(yaml_files)} strategy pair configurations...\n")
    
    results = []
    for yaml_file in sorted(yaml_files):
        is_valid, message = validate_yaml_file(yaml_file)
        results.append((yaml_file.name, is_valid, message))
        
        status = "✅" if is_valid else "❌"
        print(f"{status} {yaml_file.name}: {message}")
    
    # Summary
    valid_count = sum(1 for _, is_valid, _ in results if is_valid)
    total_count = len(results)
    
    print(f"\n{'='*60}")
    print(f"Summary: {valid_count}/{total_count} configurations are valid")
    print(f"{'='*60}")
    
    if valid_count == total_count:
        print("✅ All strategy pair configurations are valid!")
        sys.exit(0)
    else:
        print(f"❌ {total_count - valid_count} configuration(s) have errors")
        sys.exit(1)

if __name__ == "__main__":
    main()
