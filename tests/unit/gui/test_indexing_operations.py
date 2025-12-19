"""Unit tests for indexing operations.

These tests verify text and file indexing functionality,
including error handling and counter updates.
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
def gui_app_with_strategy(temp_config_files):
    """Create GUI app with mocked backend and loaded strategy."""
    with patch('rag_factory.gui.main_window.ServiceRegistry') as mock_registry_class, \
         patch('rag_factory.gui.main_window.StrategyPairManager') as mock_manager_class:
        
        mock_registry = Mock()
        mock_registry.list_services.return_value = ["db_main"]
        mock_registry.get.return_value = Mock()
        mock_registry_class.return_value = mock_registry
        
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        app = RAGFactoryGUI(**temp_config_files)
        
        # Mock loaded strategy with async process
        mock_indexing = Mock()
        
        async def mock_process(docs, context):
            result = Mock()
            result.total_chunks = 5
            return result
        
        mock_indexing.process = mock_process
        app.indexing_strategy = mock_indexing
        app.current_strategy_name = "test-strategy"
        
        yield app
        
        try:
            app.root.destroy()
        except:
            pass


class TestTextIndexing:
    """Tests for text indexing operations."""
    
    def test_index_text_with_empty_input(self, gui_app_with_strategy):
        """Test that empty text is not indexed."""
        app = gui_app_with_strategy
        app.text_input.clear()
        
        initial_count = len(app.indexed_documents)
        app._index_text()
        
        assert len(app.indexed_documents) == initial_count
    
    def test_index_text_button_disabled_during_operation(self, gui_app_with_strategy):
        """Test that Index Text button is disabled during operation."""
        app = gui_app_with_strategy
        app.text_input.set_text("Test content")
        
        app._update_button_states()
        assert str(app.index_text_btn['state']) == 'normal'
        
        app._index_text()
        assert str(app.index_text_btn['state']) == 'disabled'


class TestFileIndexing:
    """Tests for file indexing operations."""
    
    def test_index_file_with_missing_file(self, gui_app_with_strategy, tmp_path):
        """Test error handling for missing file."""
        app = gui_app_with_strategy
        app.file_path_var.set(str(tmp_path / "nonexistent.txt"))
        
        initial_count = len(app.indexed_documents)
        app._index_file()
        
        import time
        time.sleep(0.3)
        app.root.update()
        
        assert len(app.indexed_documents) == initial_count
    
    def test_index_file_button_disabled_during_operation(self, gui_app_with_strategy, tmp_path):
        """Test that Index File button is disabled during operation."""
        app = gui_app_with_strategy
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content", encoding='utf-8')
        app.file_path_var.set(str(test_file))
        
        app._update_button_states()
        assert str(app.index_file_btn['state']) == 'normal'
        
        app._index_file()
        assert str(app.index_file_btn['state']) == 'disabled'


class TestErrorHandling:
    """Tests for error handling in indexing operations."""
    
    def test_error_handling_method_exists(self, gui_app_with_strategy):
        """Test that error handling methods exist."""
        app = gui_app_with_strategy
        
        assert hasattr(app, '_on_index_error')
        assert hasattr(app, '_on_index_complete')
    
    def test_error_callback_works(self, gui_app_with_strategy):
        """Test that error callback can be called."""
        app = gui_app_with_strategy
        
        # Should not raise exception
        app._on_index_error("Test error message")
        app._update_button_states()


class TestCounterUpdates:
    """Tests for document and chunk counter updates."""
    
    def test_counters_have_increment_method(self, gui_app_with_strategy):
        """Test that status bar has counter methods."""
        app = gui_app_with_strategy
        
        assert hasattr(app.status_bar, 'increment_counts')
        assert hasattr(app.status_bar, 'update_counts')
    
    def test_counters_persist(self, gui_app_with_strategy):
        """Test that counters persist across operations."""
        app = gui_app_with_strategy
        
        app.status_bar.update_counts(documents=5, chunks=25)
        
        assert app.status_bar.document_count == 5
        assert app.status_bar.chunk_count == 25
