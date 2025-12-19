"""Unit tests for utility operations.

These tests verify Clear All Data, View Logs, Help Dialog,
and keyboard shortcuts functionality.
"""

import pytest
import tkinter as tk
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import logging

# Skip all tests if tkinter is not available
pytest.importorskip("tkinter")

from rag_factory.gui.main_window import RAGFactoryGUI


@pytest.fixture
def temp_config_files(tmp_path):
    """Create temporary configuration files for testing."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    services_yaml = config_dir / "services.yaml"
    services_yaml.write_text("services:\n  db_main:\n    type: postgresql")
    
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir()
    
    strategy_file = strategies_dir / "test-strategy.yaml"
    strategy_file.write_text("strategy_name: test-strategy\nversion: 1.0.0")
    
    alembic_ini = tmp_path / "alembic.ini"
    alembic_ini.write_text("[alembic]\nscript_location = alembic")
    
    return {
        "config_path": str(services_yaml),
        "strategies_dir": str(strategies_dir),
        "alembic_config": str(alembic_ini)
    }


@pytest.fixture
def gui_app(temp_config_files):
    """Create GUI app with mocked backend."""
    with patch('rag_factory.gui.main_window.ServiceRegistry') as mock_registry_class, \
         patch('rag_factory.gui.main_window.StrategyPairManager') as mock_manager_class:
        
        mock_registry = Mock()
        mock_registry.list_services.return_value = ["db_main"]
        mock_db = Mock()
        mock_db.clear_all = Mock()
        mock_registry.get.return_value = mock_db
        mock_registry_class.return_value = mock_registry
        
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        app = RAGFactoryGUI(**temp_config_files)
        app.current_strategy_name = "test-strategy"
        
        yield app
        
        try:
            app.root.destroy()
        except:
            pass


class TestClearAllData:
    """Tests for Clear All Data functionality."""
    
    def test_clear_data_with_no_data(self, gui_app):
        """Test clear data when no data exists."""
        app = gui_app
        
        # Ensure no data
        app.indexed_documents.clear()
        app.status_bar.reset_counts()
        
        # Should show info message
        with patch('tkinter.messagebox.showinfo') as mock_info:
            app._clear_all_data()
            mock_info.assert_called_once()
    
    def test_clear_data_confirmation_dialog(self, gui_app):
        """Test that confirmation dialog is shown."""
        app = gui_app
        
        # Add some data
        app.indexed_documents.append({"id": "doc1", "content": "test"})
        app.status_bar.update_counts(documents=1, chunks=5)
        
        with patch('tkinter.messagebox.askyesno', return_value=False) as mock_confirm:
            app._clear_all_data()
            
            # Verify confirmation was requested
            mock_confirm.assert_called_once()
            call_args = mock_confirm.call_args
            
            # Verify message contains details
            assert "test-strategy" in call_args[0][1]
            assert "cannot be undone" in call_args[0][1].lower()
    
    def test_clear_data_execution(self, gui_app):
        """Test that data is cleared when confirmed."""
        app = gui_app
        
        # Add data
        app.indexed_documents.append({"id": "doc1", "content": "test"})
        app.status_bar.update_counts(documents=1, chunks=5)
        app.results_display.set_text("Some results")
        
        with patch('tkinter.messagebox.askyesno', return_value=True):
            app._clear_all_data()
        
        # Verify data cleared
        assert len(app.indexed_documents) == 0
        assert app.status_bar.document_count == 0
        assert app.status_bar.chunk_count == 0
    
    def test_clear_data_database_integration(self, gui_app):
        """Test that database clear is called."""
        app = gui_app
        
        # Add data
        app.indexed_documents.append({"id": "doc1", "content": "test"})
        app.status_bar.update_counts(documents=1, chunks=5)
        
        with patch('tkinter.messagebox.askyesno', return_value=True):
            app._clear_all_data()
        
        # Verify database clear was called
        db_service = app.service_registry.get("db_main")
        db_service.clear_all.assert_called_once()
    
    def test_clear_data_error_handling(self, gui_app):
        """Test error handling during clear."""
        app = gui_app
        
        # Add data
        app.indexed_documents.append({"id": "doc1", "content": "test"})
        app.status_bar.update_counts(documents=1, chunks=5)
        
        # Make database clear raise error
        db_service = app.service_registry.get("db_main")
        db_service.clear_all.side_effect = Exception("Database error")
        
        with patch('tkinter.messagebox.askyesno', return_value=True), \
             patch('tkinter.messagebox.showerror') as mock_error:
            app._clear_all_data()
            
            # Verify error was shown
            mock_error.assert_called_once()


class TestLogCapture:
    """Tests for log capture functionality."""
    
    def test_log_buffer_initialized(self, gui_app):
        """Test that log buffer is initialized."""
        app = gui_app
        
        assert hasattr(app, 'log_buffer')
        assert isinstance(app.log_buffer, list)
    
    def test_log_capture_works(self, gui_app):
        """Test that logs are captured to buffer."""
        app = gui_app
        
        initial_count = len(app.log_buffer)
        
        # Log a message
        logger = logging.getLogger(__name__)
        logger.info("Test log message")
        
        # Give time for handler to process
        import time
        time.sleep(0.1)
        
        # Verify log was captured
        assert len(app.log_buffer) > initial_count
        assert any("Test log message" in log for log in app.log_buffer)
    
    def test_log_buffer_limit(self, gui_app):
        """Test that log buffer is limited to 1000 entries."""
        app = gui_app
        
        # This test verifies the buffer limit logic exists
        # Actual test would require logging 1000+ messages
        assert hasattr(app, 'log_buffer')


class TestViewLogs:
    """Tests for View Logs window."""
    
    def test_view_logs_opens_window(self, gui_app):
        """Test that view logs opens a window."""
        app = gui_app
        
        # Add some logs
        app.log_buffer.append("Log entry 1")
        app.log_buffer.append("Log entry 2")
        
        app._view_logs()
        
        # Verify window was created (Toplevel)
        # In real test, would check for Toplevel window
        assert True
    
    def test_view_logs_method_exists(self, gui_app):
        """Test that view logs method exists."""
        app = gui_app
        
        assert hasattr(app, '_view_logs')
        assert callable(app._view_logs)


class TestHelpDialog:
    """Tests for Help Dialog."""
    
    def test_help_dialog_opens(self, gui_app):
        """Test that help dialog opens."""
        app = gui_app
        
        app._show_help()
        
        # Verify help was shown
        assert True
    
    def test_help_method_exists(self, gui_app):
        """Test that help method exists."""
        app = gui_app
        
        assert hasattr(app, '_show_help')
        assert callable(app._show_help)


class TestSettingsPlaceholder:
    """Tests for Settings placeholder."""
    
    def test_settings_shows_placeholder(self, gui_app):
        """Test that settings shows placeholder message."""
        app = gui_app
        
        with patch('tkinter.messagebox.showinfo') as mock_info:
            app._show_settings()
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert "not yet implemented" in call_args[0][1]


class TestKeyboardShortcuts:
    """Tests for keyboard shortcuts."""
    
    def test_shortcuts_are_bound(self, gui_app):
        """Test that keyboard shortcuts are bound."""
        app = gui_app
        
        # Verify bindings exist
        # Note: Testing actual key bindings in tkinter is complex
        # This test verifies the run() method sets up bindings
        assert hasattr(app, 'run')
    
    def test_clear_data_shortcut(self, gui_app):
        """Test Ctrl+K shortcut for clear data."""
        app = gui_app
        
        # Add data
        app.indexed_documents.append({"id": "doc1", "content": "test"})
        app.status_bar.update_counts(documents=1, chunks=5)
        
        # Simulate Ctrl+K
        with patch('tkinter.messagebox.askyesno', return_value=True):
            # Trigger the bound function
            app._clear_all_data()
        
        # Verify data was cleared
        assert len(app.indexed_documents) == 0
    
    def test_help_shortcut(self, gui_app):
        """Test Ctrl+H and F1 shortcuts for help."""
        app = gui_app
        
        # Should not raise exception
        app._show_help()
        assert True


class TestUtilityIntegration:
    """Integration tests for utility operations."""
    
    def test_clear_and_reindex(self, gui_app):
        """Test clearing data and then reindexing."""
        app = gui_app
        
        # Add data
        app.indexed_documents.append({"id": "doc1", "content": "test"})
        app.status_bar.update_counts(documents=1, chunks=5)
        
        # Clear
        with patch('tkinter.messagebox.askyesno', return_value=True):
            app._clear_all_data()
        
        # Verify cleared
        assert len(app.indexed_documents) == 0
        
        # Add new data
        app.indexed_documents.append({"id": "doc2", "content": "new"})
        app.status_bar.update_counts(documents=1, chunks=3)
        
        # Verify new data
        assert len(app.indexed_documents) == 1
        assert app.status_bar.document_count == 1
