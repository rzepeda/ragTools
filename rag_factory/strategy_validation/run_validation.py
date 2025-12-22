#!/usr/bin/env python3
"""CLI entry point for strategy validation tool.

This script runs the strategy validation tool from the command line.
"""

import sys
import asyncio
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded environment variables from {env_file}")
    else:
        print(f"⚠️  Warning: .env file not found at {env_file}")
except ImportError:
    print("⚠️  Warning: python-dotenv not installed. Environment variables from .env will not be loaded.")
    print("   Install with: pip install python-dotenv")

from rag_factory.strategy_validation.validate_strategies import run_validation


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Validate RAG strategies by running indexing and retrieval operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all strategies
  python -m rag_factory.strategy_validation.run_validation
  
  # Validate specific strategies
  python -m rag_factory.strategy_validation.run_validation -s semantic-api-pair hybrid-search-pair
  
  # Use custom output file
  python -m rag_factory.strategy_validation.run_validation -o my_results.json
  
  # Verbose logging
  python -m rag_factory.strategy_validation.run_validation -v
        """
    )
    
    parser.add_argument(
        "-c", "--config",
        default="config/services.yaml",
        help="Path to services configuration file (default: config/services.yaml)"
    )
    
    parser.add_argument(
        "-d", "--strategies-dir",
        default="strategies",
        help="Directory containing strategy YAML files (default: strategies)"
    )
    
    parser.add_argument(
        "-t", "--test-data",
        default="rag_factory/strategy_validation/basetext.json",
        help="Path to test data JSON file (default: rag_factory/strategy_validation/basetext.json)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="validation_results.json",
        help="Path to output JSON file (default: validation_results.json)"
    )
    
    parser.add_argument(
        "-s", "--strategies",
        nargs="+",
        help="Specific strategies to validate (default: all)"
    )
    
    parser.add_argument(
        "-i", "--test-case-index",
        type=int,
        default=0,
        help="Index of test case to use from basetext.json (default: 0)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 60)
    print("RAG Factory - Strategy Validation Tool")
    print("=" * 60)
    print()
    
    # Run validation
    try:
        results = asyncio.run(run_validation(
            config_path=args.config,
            strategies_dir=args.strategies_dir,
            test_data_path=args.test_data,
            output_path=args.output,
            strategies=args.strategies,
            test_case_index=args.test_case_index,
            verbose=args.verbose
        ))
        
        # Print summary
        print()
        print("=" * 60)
        successful = sum(1 for r in results if r["error"] is None)
        failed = len(results) - successful
        
        print(f"Validation Complete!")
        print(f"  Total: {len(results)}")
        print(f"  ✓ Successful: {successful}")
        if failed > 0:
            print(f"  ✗ Failed: {failed}")
            print()
            print("Failed strategies:")
            for r in results:
                if r["error"]:
                    print(f"  - {r['strategy_name']}: {r['error']}")
        
        print()
        print(f"Results saved to: {args.output}")
        print("=" * 60)
        
        # Exit with error code if any failed
        sys.exit(0 if failed == 0 else 1)
        
    except KeyboardInterrupt:
        print("\n\nValidation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nError running validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
