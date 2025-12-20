#!/usr/bin/env python3
"""
Test GUI launch with full error capture.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

# Load .env
from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("Testing GUI Launch")
print("=" * 60)

try:
    print("\n1. Testing imports...")
    from rag_factory.gui.main_window import RAGFactoryGUI
    print("   ✅ Imports successful")
    
    print("\n2. Creating GUI instance...")
    app = RAGFactoryGUI(
        config_path="config/services.yaml",
        strategies_dir="strategies",
        alembic_config="alembic.ini"
    )
    print("   ✅ GUI created successfully")
    print(f"   - Window title: {app.root.title()}")
    
    print("\n3. Checking backend status...")
    if app.service_registry:
        print("   ✅ ServiceRegistry initialized")
    else:
        print("   ❌ ServiceRegistry not initialized")
    
    if app.strategy_manager:
        print("   ✅ StrategyPairManager initialized")
    else:
        print("   ❌ StrategyPairManager not initialized")
    
    print("\n4. Checking status bar...")
    status = app.status_bar.status_var.get()
    print(f"   Status: {status}")
    
    print("\n✅ GUI ready to run!")
    print("\nTo launch GUI window, run:")
    print("  python launch_gui.py")
    
    # Cleanup
    app.root.destroy()
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
