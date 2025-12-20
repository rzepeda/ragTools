#!/usr/bin/env python3
"""
Simple test to verify GUI can launch without import errors.

This test checks:
1. All imports work
2. GUI class can be instantiated
3. Window can be created
4. No immediate crashes
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_gui_imports():
    """Test that all GUI imports work."""
    print("Testing GUI imports...")
    try:
        from rag_factory.gui.main_window import RAGFactoryGUI
        print("✅ GUI imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gui_instantiation():
    """Test that GUI can be instantiated."""
    print("\nTesting GUI instantiation...")
    try:
        from rag_factory.gui.main_window import RAGFactoryGUI
        from unittest.mock import Mock, patch
        
        # Mock the backend components
        with patch('rag_factory.gui.main_window.ServiceRegistry') as mock_registry_class, \
             patch('rag_factory.gui.main_window.StrategyPairManager') as mock_manager_class:
            
            mock_registry = Mock()
            mock_registry.list_services.return_value = ["db_main"]
            mock_registry.get.return_value = Mock()
            mock_registry_class.return_value = mock_registry
            
            mock_manager = Mock()
            mock_manager.list_available_pairs.return_value = []
            mock_manager_class.return_value = mock_manager
            
            # Create GUI instance
            app = RAGFactoryGUI(
                config_path="config/services.yaml",
                strategies_dir="strategies",
                alembic_config="alembic.ini"
            )
            
            print("✅ GUI instantiation successful")
            print(f"   - Window title: {app.root.title()}")
            print(f"   - Window size: {app.root.geometry()}")
            
            # Cleanup
            app.root.destroy()
            
            return True
            
    except Exception as e:
        print(f"❌ Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gui_components():
    """Test that GUI components are created."""
    print("\nTesting GUI components...")
    try:
        from rag_factory.gui.main_window import RAGFactoryGUI
        from unittest.mock import Mock, patch
        
        with patch('rag_factory.gui.main_window.ServiceRegistry') as mock_registry_class, \
             patch('rag_factory.gui.main_window.StrategyPairManager') as mock_manager_class:
            
            mock_registry = Mock()
            mock_registry.list_services.return_value = ["db_main"]
            mock_registry.get.return_value = Mock()
            mock_registry_class.return_value = mock_registry
            
            mock_manager = Mock()
            mock_manager.list_available_pairs.return_value = []
            mock_manager_class.return_value = mock_manager
            
            app = RAGFactoryGUI(
                config_path="config/services.yaml",
                strategies_dir="strategies",
                alembic_config="alembic.ini"
            )
            
            # Check key components exist
            components = {
                'strategy_dropdown': 'Strategy dropdown',
                'text_input': 'Text input',
                'results_display': 'Results display',
                'status_bar': 'Status bar',
                'index_text_btn': 'Index text button',
                'index_file_btn': 'Index file button',
                'retrieve_btn': 'Retrieve button',
            }
            
            all_ok = True
            for attr, name in components.items():
                if hasattr(app, attr):
                    print(f"   ✅ {name}")
                else:
                    print(f"   ❌ {name} missing")
                    all_ok = False
            
            app.root.destroy()
            
            if all_ok:
                print("✅ All components present")
            else:
                print("❌ Some components missing")
            
            return all_ok
            
    except Exception as e:
        print(f"❌ Component check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all GUI tests."""
    print("=" * 60)
    print("RAG Factory GUI - Launch Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_gui_imports()))
    
    # Test 2: Instantiation
    if results[0][1]:  # Only if imports work
        results.append(("Instantiation", test_gui_instantiation()))
    
    # Test 3: Components
    if len(results) > 1 and results[1][1]:  # Only if instantiation works
        results.append(("Components", test_gui_components()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - GUI is ready to launch!")
        print("\nTo run the GUI:")
        print("  python -m rag_factory.gui.main_window")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Fix errors before launching")
        return 1


if __name__ == "__main__":
    sys.exit(main())
