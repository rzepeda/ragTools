#!/usr/bin/env python3
"""
Launch script for RAG Factory GUI.

This script launches the RAG Factory GUI application.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
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

from rag_factory.gui.main_window import RAGFactoryGUI


def main():
    """Launch the GUI application."""
    print("=" * 60)
    print("RAG Factory - Strategy Pair Tester")
    print("=" * 60)
    print()
    print("Launching GUI...")
    print()
    print("Prerequisites:")
    print("  1. PostgreSQL must be running")
    print("  2. Database migrations must be applied (alembic upgrade head)")
    print("  3. config/services.yaml must be configured")
    print()
    
    try:
        # Create and run GUI
        app = RAGFactoryGUI(
            config_path="config/services.yaml",
            strategies_dir="strategies",
            alembic_config="alembic.ini"
        )
        app.run()
    except KeyboardInterrupt:
        print("\nGUI closed by user")
    except Exception as e:
        print(f"\nError launching GUI: {e}")
        print("\nTroubleshooting:")
        print("  - Check that PostgreSQL is running")
        print("  - Verify config/services.yaml exists and is valid")
        print("  - Run: alembic upgrade head")
        print("  - Check logs for more details")
        sys.exit(1)


if __name__ == "__main__":
    main()
