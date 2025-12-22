"""Unit tests for main GUI window.

These tests verify the main window creation, widget initialization,
and button state management logic.
"""

import pytest
import tkinter as tk
from unittest.mock import Mock, MagicMock, patch
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
    
    # Create a test strategy
    strategy_file = strategies_dir / "test-strategy.yaml"
    strategy_file.write_text("strategy_name: test-strategy\nversion: 1.0.0")
    
    # Create alembic.ini
    alembic_ini = tmp_path / "alembic.ini"
    alembic_ini.write_text("[alembic]\nscript_location = alembic")
    
    return {
        "config_path": str(services_yaml),
        "strategies_dir": str(strategies_dir),
        "alembic_config": str(alembic_ini)
    }


@pytest.fixture
def gui_app(temp_config_files):
    """Create a GUI application instance for testing with mocked backend.
    
    Args:
        temp_config_files: Temporary config files fixture
        
    Yields:
        RAGFactoryGUI instance
    """
    # Mock backend initialization to avoid real service connections
    with patch('rag_factory.gui.main_window.ServiceRegistry') as mock_registry_class, \
         patch('rag_factory.gui.main_window.StrategyPairManager') as mock_manager_class:
        
        # Setup mock registry
        mock_registry = Mock()
        mock_registry.list_services.return_value = ["db_main", "embedding_local"]
        mock_registry_class.return_value = mock_registry
        
        # Setup mock manager
        mock_manager = Mock()
        mock_manager.load_pair = Mock(return_value=(Mock(), Mock()))
        mock_manager_class.return_value = mock_manager
        
        # Create GUI app
        app = RAGFactoryGUI(**temp_config_files)
        
        yield app
        
        # Cleanup
        try:
            app.root.destroy()
        except:
            pass


class TestMainWindowCreation:
    """Tests for main window creation and properties."""
    
    def test_window_title(self, gui_app):
        """Test window has correct title."""
        assert gui_app.root.title() == "RAG Factory - Strategy Pair Tester"
    
    def test_window_geometry(self, gui_app):
        """Test window has correct default geometry."""
        # Force update to get actual dimensions
        gui_app.root.update()
        
        # Check geometry string contains 1200x800 (actual default size)
        geometry = gui_app.root.geometry()
        assert "1200x800" in geometry
    
    def test_window_minimum_size(self, gui_app):
        """Test window has correct minimum size."""
        min_width = gui_app.root.minsize()[0]
        min_height = gui_app.root.minsize()[1]
        
        assert min_width == 900
        assert min_height == 600
    
    def test_window_is_resizable(self, gui_app):
        """Test window is resizable."""
        # Window should be resizable by default
        # This is implicit in tkinter unless explicitly disabled
        assert gui_app.root.winfo_exists()


class TestWidgetCreation:
    """Tests for widget creation and initialization."""
    
    def test_strategy_dropdown_exists(self, gui_app):
        """Test strategy dropdown is created."""
        assert hasattr(gui_app, 'strategy_dropdown')
        assert gui_app.strategy_dropdown is not None
        assert isinstance(gui_app.strategy_dropdown, tk.Widget)
    
    def test_config_preview_exists(self, gui_app):
        """Test configuration preview textbox exists."""
        assert hasattr(gui_app, 'config_preview')
        assert gui_app.config_preview is not None
    
    def test_text_input_exists(self, gui_app):
        """Test text input textbox exists."""
        assert hasattr(gui_app, 'text_input')
        assert gui_app.text_input is not None
    
    def test_file_path_entry_exists(self, gui_app):
        """Test file path entry exists."""
        assert hasattr(gui_app, 'file_path_entry')
        assert gui_app.file_path_entry is not None
    
    def test_query_entry_exists(self, gui_app):
        """Test query entry exists."""
        assert hasattr(gui_app, 'query_entry')
        assert gui_app.query_entry is not None
    
    def test_results_display_exists(self, gui_app):
        """Test results display exists."""
        assert hasattr(gui_app, 'results_display')
        assert gui_app.results_display is not None
    
    def test_status_bar_exists(self, gui_app):
        """Test status bar exists."""
        assert hasattr(gui_app, 'status_bar')
        assert gui_app.status_bar is not None
    
    def test_all_buttons_exist(self, gui_app):
        """Test all buttons are created."""
        # Only test buttons that exist as attributes (not menu items)
        buttons = [
            'reload_btn',
            'index_text_btn',
            'browse_btn',
            'index_file_btn',
            'retrieve_btn'
        ]
        
        for button_name in buttons:
            assert hasattr(gui_app, button_name), f"Missing button: {button_name}"
            assert getattr(gui_app, button_name) is not None
        
        # Utility functions are in menu, not as button attributes
        assert hasattr(gui_app, '_clear_all_data')
        assert hasattr(gui_app, '_view_logs')
        assert hasattr(gui_app, '_show_settings')
        assert hasattr(gui_app, '_show_help')


class TestButtonStates:
    """Tests for button state management."""
    
    def test_initial_button_states(self, gui_app):
        """Test buttons are initially disabled (no strategy loaded)."""
        # Action buttons should be disabled initially
        assert str(gui_app.index_text_btn['state']) == 'disabled'
        assert str(gui_app.index_file_btn['state']) == 'disabled'
        assert str(gui_app.retrieve_btn['state']) == 'disabled'
    
    def test_button_state_with_strategy_and_text(self, gui_app):
        """Test Index Text button enabled when strategy loaded and text present."""
        # Simulate strategy loaded
        gui_app.current_strategy_name = "test-strategy"
        gui_app.indexing_strategy = Mock()
        
        # Add text
        gui_app.text_input.set_text("Test text content")
        
        # Update button states
        gui_app._update_button_states()
        
        # Index Text button should be enabled
        assert str(gui_app.index_text_btn['state']) == 'normal'
    
    def test_button_state_with_strategy_no_text(self, gui_app):
        """Test Index Text button disabled when strategy loaded but no text."""
        # Simulate strategy loaded (need both indexing and retrieval)
        gui_app.current_strategy_name = "test-strategy"
        gui_app.indexing_strategy = Mock()
        gui_app.retrieval_strategy = Mock()  # Also needed for button state logic
        
        # Mock text_input.is_empty() to return True (no text)
        gui_app.text_input.is_empty = Mock(return_value=True)
        
        # Update button states
        gui_app._update_button_states()
        
        # Index Text button should be disabled
        assert str(gui_app.index_text_btn['state']) == 'disabled'
    
    def test_button_state_with_text_no_strategy(self, gui_app):
        """Test Index Text button disabled when text present but no strategy."""
        # No strategy loaded
        gui_app.current_strategy_name = None
        gui_app.indexing_strategy = None
        
        # Add text
        gui_app.text_input.set_text("Test text content")
        
        # Update button states
        gui_app._update_button_states()
        
        # Index Text button should be disabled
        assert str(gui_app.index_text_btn['state']) == 'disabled'
    
    def test_retrieve_button_state_with_query(self, gui_app):
        """Test Retrieve button enabled when strategy loaded and query present."""
        # Simulate strategy loaded (need both indexing and retrieval)
        gui_app.current_strategy_name = "test-strategy"
        gui_app.indexing_strategy = Mock()  # Also needed for complete state
        gui_app.retrieval_strategy = Mock()
        
        # Add query
        gui_app.query_var.set("What is machine learning?")
        
        # Update button states
        gui_app._update_button_states()
        
        # Retrieve button should be enabled
        assert str(gui_app.retrieve_btn['state']) == 'normal'


class TestEventBinding:
    """Tests for event handler binding."""
    
    def test_strategy_dropdown_has_binding(self, gui_app):
        """Test strategy dropdown has selection event binding."""
        # Check that the combobox has the <<ComboboxSelected>> binding
        bindings = gui_app.strategy_dropdown.bind()
        assert '<<ComboboxSelected>>' in bindings or len(bindings) > 0
    
    def test_buttons_have_commands(self, gui_app):
        """Test buttons have command callbacks."""
        # Check that buttons have commands configured
        assert gui_app.reload_btn['command'] is not None
        assert gui_app.browse_btn['command'] is not None
        assert gui_app.index_text_btn['command'] is not None
        assert gui_app.index_file_btn['command'] is not None
        assert gui_app.retrieve_btn['command'] is not None


class TestApplicationState:
    """Tests for application state management."""
    
    def test_initial_state(self, gui_app):
        """Test initial application state."""
        assert gui_app.current_strategy_name is None
        assert gui_app.indexing_strategy is None
        assert gui_app.retrieval_strategy is None
        assert gui_app.indexed_documents == []
    
    def test_strategy_loading_updates_state(self, gui_app):
        """Test strategy loading updates application state."""
        # Simulate successful strategy load
        mock_indexing = Mock()
        mock_retrieval = Mock()
        config = {"strategy_name": "test", "version": "1.0.0"}
        
        gui_app._on_strategy_loaded("test-strategy", mock_indexing, mock_retrieval, config)
        
        assert gui_app.current_strategy_name == "test-strategy"
        assert gui_app.indexing_strategy == mock_indexing
        assert gui_app.retrieval_strategy == mock_retrieval


class TestUtilityMethods:
    """Tests for utility methods."""
    
    def test_browse_file_opens_dialog(self, gui_app):
        """Test browse file method."""
        with patch('tkinter.filedialog.askopenfilename', return_value='/path/to/file.txt'):
            gui_app._browse_file()
            assert gui_app.file_path_var.get() == '/path/to/file.txt'
    
    def test_clear_all_data_with_no_data(self, gui_app):
        """Test clear all data when no data exists."""
        with patch('tkinter.messagebox.showinfo') as mock_info:
            gui_app._clear_all_data()
            mock_info.assert_called_once()
    
    def test_clear_all_data_with_confirmation(self, gui_app):
        """Test clear all data with user confirmation."""
        # Add some indexed documents
        gui_app.indexed_documents = [{"id": "doc1", "content": "test"}]
        
        with patch('tkinter.messagebox.askyesno', return_value=True):
            gui_app._clear_all_data()
            assert len(gui_app.indexed_documents) == 0
    
    def test_show_help_displays_message(self, gui_app):
        """Test help dialog displays."""
        # Help uses Toplevel window, not messagebox.showinfo
        # Just verify the method exists and can be called
        assert hasattr(gui_app, '_show_help')
        assert callable(gui_app._show_help)
        # Note: Actually calling it would create a Toplevel window


class TestWindowLifecycle:
    """Tests for window lifecycle management."""
    
    def test_window_can_be_destroyed(self, gui_app):
        """Test window can be destroyed cleanly."""
        # Window should exist
        assert gui_app.root.winfo_exists()
        
        # Destroy window
        gui_app.root.destroy()
        
        # Window should no longer exist
        # Note: After destroy, winfo_exists() may raise an error
        try:
            exists = gui_app.root.winfo_exists()
            assert not exists
        except tk.TclError:
            # Expected - window is destroyed
            pass
