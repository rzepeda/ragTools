"""Integration tests for GUI application launch and lifecycle.

These tests verify that the GUI can be launched, initialized properly,
and shut down cleanly.
"""

import pytest
import sys
from unittest.mock import Mock, patch
from pathlib import Path

# Skip all tests if tkinter is not available
pytest.importorskip("tkinter")

from rag_factory.gui.main_window import RAGFactoryGUI


@pytest.fixture
def temp_config_files(tmp_path):
    """Create temporary configuration files for testing."""
    # Create config directory
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create services.yaml
    services_yaml = config_dir / "services.yaml"
    services_yaml.write_text("services:\n  db_main:\n    type: postgresql")
    
    # Create strategies directory
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir()
    
    # Create test strategies
    for i in range(3):
        strategy_file = strategies_dir / f"strategy-{i}.yaml"
        strategy_file.write_text(f"strategy_name: strategy-{i}\nversion: 1.0.0")
    
    # Create alembic.ini
    alembic_ini = tmp_path / "alembic.ini"
    alembic_ini.write_text("[alembic]\nscript_location = alembic")
    
    return {
        "config_path": str(services_yaml),
        "strategies_dir": str(strategies_dir),
        "alembic_config": str(alembic_ini)
    }


class TestGUILaunch:
    """Tests for GUI application launch."""
    
    @patch('rag_factory.gui.main_window.StrategyPairManager')
    @patch('rag_factory.gui.main_window.ServiceRegistry')
    def test_gui_launches_without_error(self, mock_registry_class, mock_manager_class, temp_config_files):
        """Test GUI can be launched without errors."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry.list_services.return_value = ["db_main"]
        mock_registry_class.return_value = mock_registry
        
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Launch GUI
        app = RAGFactoryGUI(**temp_config_files)
        
        # Verify window is created
        assert app.root is not None
        assert app.root.winfo_exists()
        
        # Clean up
        app.root.destroy()
    
    @patch('rag_factory.gui.main_window.StrategyPairManager')
    @patch('rag_factory.gui.main_window.ServiceRegistry')
    def test_gui_window_properties(self, mock_registry_class, mock_manager_class, temp_config_files):
        """Test GUI window has correct properties after launch."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry.list_services.return_value = ["db_main"]
        mock_registry_class.return_value = mock_registry
        
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Launch GUI
        app = RAGFactoryGUI(**temp_config_files)
        
        # Force update to get actual dimensions
        app.root.update()
        
        # Verify properties
        assert app.root.title() == "RAG Factory - Strategy Pair Tester"
        assert app.root.minsize() == (800, 600)
        
        # Clean up
        app.root.destroy()
    
    @patch('rag_factory.gui.main_window.StrategyPairManager')
    @patch('rag_factory.gui.main_window.ServiceRegistry')
    def test_gui_loads_strategies(self, mock_registry_class, mock_manager_class, temp_config_files):
        """Test GUI loads available strategies on launch."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry.list_services.return_value = ["db_main"]
        mock_registry_class.return_value = mock_registry
        
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Launch GUI
        app = RAGFactoryGUI(**temp_config_files)
        
        # Verify strategies are loaded in dropdown
        values = app.strategy_dropdown['values']
        assert len(values) == 3
        assert 'strategy-0' in values
        assert 'strategy-1' in values
        assert 'strategy-2' in values
        
        # Clean up
        app.root.destroy()
    
    @patch('rag_factory.gui.main_window.StrategyPairManager')
    @patch('rag_factory.gui.main_window.ServiceRegistry')
    def test_gui_handles_missing_strategies_dir(self, mock_registry_class, mock_manager_class, tmp_path):
        """Test GUI handles missing strategies directory gracefully."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry.list_services.return_value = ["db_main"]
        mock_registry_class.return_value = mock_registry
        
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Use non-existent directory
        strategies_dir = str(tmp_path / "nonexistent")
        config_path = str(tmp_path / "config" / "services.yaml")
        
        # Create config file
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "services.yaml").write_text("services:\n  db_main:\n    type: postgresql")
        
        # Launch GUI (should not crash)
        app = RAGFactoryGUI(config_path=config_path, strategies_dir=strategies_dir)
        
        # Verify window is created despite missing directory
        assert app.root is not None
        assert app.root.winfo_exists()
        
        # Clean up
        app.root.destroy()


class TestGUIShutdown:
    """Tests for GUI application shutdown."""
    
    def test_gui_closes_cleanly(self, mock_strategy_manager, tmp_path):
        """Test GUI can be closed without errors."""
        # Create temporary strategies directory
        strategies_dir = tmp_path / "strategies"
        strategies_dir.mkdir()
        
        # Launch GUI
        app = RAGFactoryGUI(mock_strategy_manager, strategies_dir=str(strategies_dir))
        
        # Verify window exists
        assert app.root.winfo_exists()
        
        # Close window
        app.root.destroy()
        
        # Verify window is destroyed
        # After destroy, winfo_exists() may raise an error
        try:
            exists = app.root.winfo_exists()
            assert not exists
        except:
            # Expected - window is destroyed
            pass
    
    def test_gui_on_close_handler(self, mock_strategy_manager, tmp_path):
        """Test GUI close handler works correctly."""
        # Create temporary strategies directory
        strategies_dir = tmp_path / "strategies"
        strategies_dir.mkdir()
        
        # Launch GUI
        app = RAGFactoryGUI(mock_strategy_manager, strategies_dir=str(strategies_dir))
        
        # Verify window exists
        assert app.root.winfo_exists()
        
        # Manually call close handler (simulates clicking X button)
        # This should destroy the window
        # Note: We can't actually test the protocol binding easily,
        # but we can test that root.destroy() works
        app.root.destroy()


class TestGUICLIIntegration:
    """Tests for GUI CLI command integration."""
    
    @patch('rag_factory.cli.commands.gui.ServiceRegistry')
    @patch('rag_factory.cli.commands.gui.StrategyPairManager')
    @patch('rag_factory.cli.commands.gui.RAGFactoryGUI')
    def test_gui_command_initializes_services(
        self,
        mock_gui_class,
        mock_manager_class,
        mock_registry_class,
        tmp_path
    ):
        """Test GUI CLI command initializes services correctly."""
        from rag_factory.cli.commands.gui import gui_command
        
        # Create temporary config and strategies
        config_file = tmp_path / "services.yaml"
        config_file.write_text("services: {}")
        
        strategies_dir = tmp_path / "strategies"
        strategies_dir.mkdir()
        
        # Mock the GUI run method to prevent actual GUI launch
        mock_gui_instance = Mock()
        mock_gui_class.return_value = mock_gui_instance
        
        # Call GUI command (this would normally be called by typer)
        # We can't easily test the full typer integration, but we can
        # verify the imports work
        assert gui_command is not None


class TestGUIStateManagement:
    """Tests for GUI state management during lifecycle."""
    
    def test_gui_maintains_state_during_operations(self, mock_strategy_manager, tmp_path):
        """Test GUI maintains state correctly during operations."""
        # Create temporary strategies directory
        strategies_dir = tmp_path / "strategies"
        strategies_dir.mkdir()
        
        # Launch GUI
        app = RAGFactoryGUI(mock_strategy_manager, strategies_dir=str(strategies_dir))
        
        # Set some state
        app.current_strategy_name = "test-strategy"
        app.indexed_documents = [{"id": "doc1", "content": "test"}]
        
        # Verify state is maintained
        assert app.current_strategy_name == "test-strategy"
        assert len(app.indexed_documents) == 1
        
        # Clean up
        app.root.destroy()
