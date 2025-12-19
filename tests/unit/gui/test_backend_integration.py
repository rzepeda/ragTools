"""Unit tests for backend integration.

These tests verify ServiceRegistry and StrategyPairManager integration
with the GUI, including error handling for various failure scenarios.
"""

import pytest
import tkinter as tk
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path

# Skip all tests if tkinter is not available
pytest.importorskip("tkinter")

from rag_factory.gui.main_window import RAGFactoryGUI
from rag_factory.core.exceptions import MigrationError


@pytest.fixture
def temp_config_files(tmp_path):
    """Create temporary configuration files for testing."""
    # Create config directory
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create services.yaml
    services_yaml = config_dir / "services.yaml"
    services_yaml.write_text("""
services:
  db_main:
    type: postgresql
    connection_string: postgresql://localhost/test
""")
    
    # Create strategies directory
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir()
    
    # Create a test strategy
    strategy_file = strategies_dir / "test-strategy.yaml"
    strategy_file.write_text("""
strategy_name: test-strategy
version: 1.0.0
indexer:
  type: semantic
retriever:
  type: semantic
""")
    
    # Create alembic.ini
    alembic_ini = tmp_path / "alembic.ini"
    alembic_ini.write_text("[alembic]\nscript_location = alembic")
    
    return {
        "config_path": str(services_yaml),
        "strategies_dir": str(strategies_dir),
        "alembic_config": str(alembic_ini)
    }


class TestBackendInitialization:
    """Tests for backend initialization."""
    
    @patch('rag_factory.gui.main_window.StrategyPairManager')
    @patch('rag_factory.gui.main_window.ServiceRegistry')
    def test_successful_backend_initialization(
        self,
        mock_registry_class,
        mock_manager_class,
        temp_config_files
    ):
        """Test successful backend initialization."""
        # Mock ServiceRegistry
        mock_registry = Mock()
        mock_registry.list_services.return_value = ["db_main", "embedding_local"]
        mock_registry_class.return_value = mock_registry
        
        # Mock StrategyPairManager
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Create GUI
        app = RAGFactoryGUI(**temp_config_files)
        
        # Verify backend initialized
        assert app.service_registry is not None
        assert app.strategy_manager is not None
        
        # Verify ServiceRegistry was created with correct path
        mock_registry_class.assert_called_once_with(temp_config_files["config_path"])
        
        # Verify StrategyPairManager was created
        mock_manager_class.assert_called_once()
        
        # Clean up
        app.root.destroy()
    
    def test_missing_config_file(self, tmp_path):
        """Test error handling when config file is missing."""
        # Use non-existent config path
        config_path = str(tmp_path / "nonexistent" / "services.yaml")
        strategies_dir = str(tmp_path / "strategies")
        
        # Create GUI (should not crash)
        app = RAGFactoryGUI(
            config_path=config_path,
            strategies_dir=strategies_dir
        )
        
        # Verify backend not initialized
        assert app.service_registry is None
        assert app.strategy_manager is None
        
        # Clean up
        app.root.destroy()
    
    @patch('rag_factory.gui.main_window.ServiceRegistry')
    def test_service_registry_initialization_error(
        self,
        mock_registry_class,
        temp_config_files
    ):
        """Test error handling when ServiceRegistry initialization fails."""
        # Mock ServiceRegistry to raise error
        mock_registry_class.side_effect = Exception("Invalid YAML")
        
        # Create GUI (should not crash)
        app = RAGFactoryGUI(**temp_config_files)
        
        # Verify backend not initialized
        assert app.service_registry is None
        assert app.strategy_manager is None
        
        # Clean up
        app.root.destroy()
    
    @patch('rag_factory.gui.main_window.StrategyPairManager')
    @patch('rag_factory.gui.main_window.ServiceRegistry')
    def test_missing_required_services(
        self,
        mock_registry_class,
        mock_manager_class,
        temp_config_files
    ):
        """Test error handling when required services are missing."""
        # Mock ServiceRegistry without required services
        mock_registry = Mock()
        mock_registry.list_services.return_value = []  # No services
        mock_registry_class.return_value = mock_registry
        
        # Create GUI
        app = RAGFactoryGUI(**temp_config_files)
        
        # Verify ServiceRegistry created but StrategyPairManager not created
        assert app.service_registry is not None
        assert app.strategy_manager is None
        
        # Clean up
        app.root.destroy()
    
    @patch('rag_factory.gui.main_window.StrategyPairManager')
    @patch('rag_factory.gui.main_window.ServiceRegistry')
    def test_strategy_manager_initialization_error(
        self,
        mock_registry_class,
        mock_manager_class,
        temp_config_files
    ):
        """Test error handling when StrategyPairManager initialization fails."""
        # Mock ServiceRegistry
        mock_registry = Mock()
        mock_registry.list_services.return_value = ["db_main"]
        mock_registry_class.return_value = mock_registry
        
        # Mock StrategyPairManager to raise error
        mock_manager_class.side_effect = Exception("Alembic error")
        
        # Create GUI
        app = RAGFactoryGUI(**temp_config_files)
        
        # Verify ServiceRegistry created but StrategyPairManager not created
        assert app.service_registry is not None
        assert app.strategy_manager is None
        
        # Clean up
        app.root.destroy()


class TestServiceVerification:
    """Tests for service verification."""
    
    @patch('rag_factory.gui.main_window.StrategyPairManager')
    @patch('rag_factory.gui.main_window.ServiceRegistry')
    def test_verify_required_services_success(
        self,
        mock_registry_class,
        mock_manager_class,
        temp_config_files
    ):
        """Test successful service verification."""
        # Mock ServiceRegistry with all required services
        mock_registry = Mock()
        mock_registry.list_services.return_value = ["db_main", "embedding_local"]
        mock_registry_class.return_value = mock_registry
        
        # Create GUI
        app = RAGFactoryGUI(**temp_config_files)
        
        # Verify services were checked
        mock_registry.list_services.assert_called()
        
        # Clean up
        app.root.destroy()
    
    @patch('rag_factory.gui.main_window.ServiceRegistry')
    def test_verify_required_services_failure(
        self,
        mock_registry_class,
        temp_config_files
    ):
        """Test service verification failure."""
        # Mock ServiceRegistry without db_main
        mock_registry = Mock()
        mock_registry.list_services.return_value = ["embedding_local"]  # Missing db_main
        mock_registry_class.return_value = mock_registry
        
        # Create GUI
        app = RAGFactoryGUI(**temp_config_files)
        
        # Verify StrategyPairManager not created
        assert app.strategy_manager is None
        
        # Clean up
        app.root.destroy()


class TestErrorHandling:
    """Tests for enhanced error handling."""
    
    @patch('rag_factory.gui.main_window.StrategyPairManager')
    @patch('rag_factory.gui.main_window.ServiceRegistry')
    def test_migration_error_handling(
        self,
        mock_registry_class,
        mock_manager_class,
        temp_config_files
    ):
        """Test enhanced error message for migration errors."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry.list_services.return_value = ["db_main"]
        mock_registry_class.return_value = mock_registry
        
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Create GUI
        app = RAGFactoryGUI(**temp_config_files)
        
        # Simulate migration error
        error_msg = "Missing migration: abc123"
        app._on_strategy_load_error(error_msg)
        
        # Verify error was categorized as migration error
        # (This is tested by checking the status bar text)
        status_text = app.status_bar.status_label.cget('text')
        assert "migration" in status_text.lower()
        
        # Clean up
        app.root.destroy()
    
    @patch('rag_factory.gui.main_window.StrategyPairManager')
    @patch('rag_factory.gui.main_window.ServiceRegistry')
    def test_service_error_handling(
        self,
        mock_registry_class,
        mock_manager_class,
        temp_config_files
    ):
        """Test enhanced error message for service errors."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry.list_services.return_value = ["db_main"]
        mock_registry_class.return_value = mock_registry
        
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Create GUI
        app = RAGFactoryGUI(**temp_config_files)
        
        # Simulate service error
        error_msg = "Missing service: embedding_local"
        app._on_strategy_load_error(error_msg)
        
        # Verify error was categorized as service error
        status_text = app.status_bar.status_label.cget('text')
        assert "service" in status_text.lower()
        
        # Clean up
        app.root.destroy()


class TestStrategyLoading:
    """Tests for strategy loading with backend integration."""
    
    @patch('rag_factory.gui.main_window.StrategyPairManager')
    @patch('rag_factory.gui.main_window.ServiceRegistry')
    def test_load_strategies_without_backend(
        self,
        mock_registry_class,
        mock_manager_class,
        temp_config_files
    ):
        """Test that strategy loading is skipped when backend not initialized."""
        # Mock ServiceRegistry to fail
        mock_registry_class.side_effect = Exception("Config error")
        
        # Create GUI
        app = RAGFactoryGUI(**temp_config_files)
        
        # Verify strategy manager is None
        assert app.strategy_manager is None
        
        # Verify strategy dropdown is empty
        assert len(app.strategy_dropdown['values']) == 0
        
        # Clean up
        app.root.destroy()
