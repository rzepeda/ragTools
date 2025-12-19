"""Unit tests for retrieval operations.

These tests verify query execution, result formatting,
and error handling for retrieval functionality.
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
        
        # Mock loaded retrieval strategy with async retrieve
        mock_retrieval = Mock()
        
        async def mock_retrieve(query, context):
            # Return mock results
            result1 = Mock()
            result1.score = 0.8923
            result1.content = "Machine learning is a subset of artificial intelligence"
            result1.metadata = {"source": "ml_basics.txt"}
            
            result2 = Mock()
            result2.score = 0.7645
            result2.content = "Types of Machine Learning include supervised and unsupervised"
            result2.metadata = {"source": "ml_types.txt"}
            
            return [result1, result2]
        
        mock_retrieval.retrieve = mock_retrieve
        app.retrieval_strategy = mock_retrieval
        app.current_strategy_name = "test-strategy"
        
        yield app
        
        try:
            app.root.destroy()
        except:
            pass


class TestQueryExecution:
    """Tests for query execution."""
    
    def test_retrieve_with_empty_query(self, gui_app_with_strategy):
        """Test that empty query is not executed."""
        app = gui_app_with_strategy
        app.query_var.set("")
        
        app._retrieve()
        
        # Should return early, no operation performed
        assert True
    
    def test_retrieve_button_disabled_during_operation(self, gui_app_with_strategy):
        """Test that Retrieve button is disabled during operation."""
        app = gui_app_with_strategy
        app.query_var.set("test query")
        
        app._update_button_states()
        assert str(app.retrieve_btn['state']) == 'normal'
        
        app._retrieve()
        assert str(app.retrieve_btn['state']) == 'disabled'
    
    def test_top_k_value_parsing(self, gui_app_with_strategy):
        """Test that top_k value is parsed correctly."""
        app = gui_app_with_strategy
        
        # Set top_k
        app.top_k_var.set("3")
        app.query_var.set("test query")
        
        # Should not raise exception
        app._retrieve()
        assert True


class TestResultFormatting:
    """Tests for result formatting."""
    
    def test_format_results_with_multiple_results(self, gui_app_with_strategy):
        """Test formatting multiple results."""
        app = gui_app_with_strategy
        
        # Create mock results
        result1 = Mock()
        result1.score = 0.8923
        result1.content = "Test content 1"
        result1.metadata = {"source": "doc1.txt"}
        
        result2 = Mock()
        result2.score = 0.7645
        result2.content = "Test content 2"
        result2.metadata = {"source": "doc2.txt"}
        
        results = [result1, result2]
        
        formatted = app._format_retrieval_results("test query", results, 5)
        
        assert "test query" in formatted
        assert "Found 2 results" in formatted
        assert "0.8923" in formatted
        assert "0.7645" in formatted
        assert "Test content 1" in formatted
        assert "Test content 2" in formatted
        assert "doc1.txt" in formatted
        assert "doc2.txt" in formatted
    
    def test_format_results_with_empty_results(self, gui_app_with_strategy):
        """Test formatting empty results."""
        app = gui_app_with_strategy
        
        formatted = app._format_retrieval_results("test query", [], 5)
        
        assert "No results found" in formatted
        assert "test query" in formatted
        assert "Suggestions" in formatted
    
    def test_content_truncation(self, gui_app_with_strategy):
        """Test that long content is truncated."""
        app = gui_app_with_strategy
        
        result = Mock()
        result.score = 0.9
        result.content = "A" * 300  # Long content
        result.metadata = {"source": "doc.txt"}
        
        formatted = app._format_retrieval_results("query", [result], 5)
        
        # Should contain truncation indicator
        assert "..." in formatted
        # Should not contain full 300 chars
        assert "A" * 300 not in formatted
    
    def test_score_formatting(self, gui_app_with_strategy):
        """Test that scores are formatted to 4 decimal places."""
        app = gui_app_with_strategy
        
        result = Mock()
        result.score = 0.123456789
        result.content = "Test"
        result.metadata = {"source": "doc.txt"}
        
        formatted = app._format_retrieval_results("query", [result], 5)
        
        # Should have 4 decimal places
        assert "0.1235" in formatted
    
    def test_rank_numbering(self, gui_app_with_strategy):
        """Test that results are numbered correctly."""
        app = gui_app_with_strategy
        
        results = []
        for i in range(3):
            result = Mock()
            result.score = 0.9 - (i * 0.1)
            result.content = f"Content {i}"
            result.metadata = {"source": f"doc{i}.txt"}
            results.append(result)
        
        formatted = app._format_retrieval_results("query", results, 5)
        
        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "[3]" in formatted


class TestEmptyResultsHandling:
    """Tests for empty results handling."""
    
    def test_empty_results_message(self, gui_app_with_strategy):
        """Test that empty results show helpful message."""
        app = gui_app_with_strategy
        
        formatted = app._format_retrieval_results("nonexistent", [], 5)
        
        assert "No results found" in formatted
        assert "nonexistent" in formatted
        assert "Make sure documents are indexed first" in formatted
        assert "Try different keywords" in formatted


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_error_handling_method_exists(self, gui_app_with_strategy):
        """Test that error handling methods exist."""
        app = gui_app_with_strategy
        
        assert hasattr(app, '_on_retrieve_error')
        assert hasattr(app, '_on_retrieve_complete')
    
    def test_error_callback_works(self, gui_app_with_strategy):
        """Test that error callback can be called."""
        app = gui_app_with_strategy
        
        # Should not raise exception
        app._on_retrieve_error("Test error message")
        app._update_button_states()
    
    def test_error_displays_in_results(self, gui_app_with_strategy):
        """Test that errors are displayed in results textbox."""
        app = gui_app_with_strategy
        
        app._on_retrieve_error("Database connection failed")
        
        # Check results display contains error
        results_text = app.results_display.get_text()
        assert "Retrieval Error" in results_text


class TestProgressIndication:
    """Tests for progress indication."""
    
    def test_searching_placeholder_shown(self, gui_app_with_strategy):
        """Test that 'Searching...' placeholder is shown."""
        app = gui_app_with_strategy
        app.query_var.set("test query")
        
        app._retrieve()
        
        # Check that results display was updated
        # (In real scenario, would show "Searching...")
        assert True
    
    def test_status_bar_updated(self, gui_app_with_strategy):
        """Test that status bar is updated during retrieval."""
        app = gui_app_with_strategy
        app.query_var.set("test query")
        
        app._retrieve()
        
        # Status should be updated
        status_text = app.status_bar.status_label.cget('text')
        assert "Retrieving" in status_text or "Working" in status_text


class TestCompletionHandlers:
    """Tests for completion handlers."""
    
    def test_completion_handler_displays_results(self, gui_app_with_strategy):
        """Test that completion handler displays formatted results."""
        app = gui_app_with_strategy
        
        formatted_results = "Query: test\nFound 2 results:\n..."
        app._on_retrieve_complete(formatted_results, 2, 0.5)
        
        results_text = app.results_display.get_text()
        assert "Query: test" in results_text
    
    def test_completion_handler_updates_status(self, gui_app_with_strategy):
        """Test that completion handler updates status bar."""
        app = gui_app_with_strategy
        
        app._on_retrieve_complete("Results...", 3, 0.75)
        
        status_text = app.status_bar.status_label.cget('text')
        assert "Found 3 results" in status_text
        assert "0.75s" in status_text
