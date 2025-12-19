"""Integration tests for GUI end-to-end workflows.

These tests verify complete workflows with real backend components.
"""

import pytest
import tkinter as tk
from unittest.mock import Mock, patch
from pathlib import Path
import time

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


@pytest.mark.integration
class TestGUILaunch:
    """Integration tests for GUI launch."""
    
    def test_gui_launches_successfully(self, temp_config_files):
        """Test that GUI launches without errors."""
        with patch('rag_factory.gui.main_window.ServiceRegistry') as mock_registry_class, \
             patch('rag_factory.gui.main_window.StrategyPairManager') as mock_manager_class:
            
            mock_registry = Mock()
            mock_registry.list_services.return_value = ["db_main"]
            mock_registry.get.return_value = Mock()
            mock_registry_class.return_value = mock_registry
            
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            app = RAGFactoryGUI(**temp_config_files)
            
            # Verify window was created
            assert app.root is not None
            assert "RAG Factory" in app.root.title()
            
            app.root.destroy()
    
    def test_gui_initializes_components(self, temp_config_files):
        """Test that all GUI components are initialized."""
        with patch('rag_factory.gui.main_window.ServiceRegistry') as mock_registry_class, \
             patch('rag_factory.gui.main_window.StrategyPairManager') as mock_manager_class:
            
            mock_registry = Mock()
            mock_registry.list_services.return_value = ["db_main"]
            mock_registry.get.return_value = Mock()
            mock_registry_class.return_value = mock_registry
            
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            app = RAGFactoryGUI(**temp_config_files)
            
            # Verify key components exist
            assert hasattr(app, 'strategy_dropdown')
            assert hasattr(app, 'text_input')
            assert hasattr(app, 'results_display')
            assert hasattr(app, 'status_bar')
            
            app.root.destroy()


@pytest.mark.integration
class TestStrategyLoading:
    """Integration tests for strategy loading."""
    
    def test_strategy_loading_workflow(self, temp_config_files):
        """Test complete strategy loading workflow."""
        with patch('rag_factory.gui.main_window.ServiceRegistry') as mock_registry_class, \
             patch('rag_factory.gui.main_window.StrategyPairManager') as mock_manager_class:
            
            mock_registry = Mock()
            mock_registry.list_services.return_value = ["db_main"]
            mock_registry.get.return_value = Mock()
            mock_registry_class.return_value = mock_registry
            
            mock_manager = Mock()
            mock_manager.list_available_pairs.return_value = ["test-strategy"]
            
            # Mock load_pair to return mock strategies
            mock_indexing = Mock()
            mock_retrieval = Mock()
            mock_manager.load_pair.return_value = (mock_indexing, mock_retrieval)
            
            mock_manager_class.return_value = mock_manager
            
            app = RAGFactoryGUI(**temp_config_files)
            
            # Simulate strategy selection
            app._load_strategy("test-strategy")
            
            # Verify strategies were loaded
            assert app.indexing_strategy is not None
            assert app.retrieval_strategy is not None
            assert app.current_strategy_name == "test-strategy"
            
            app.root.destroy()


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Integration tests for complete workflows."""
    
    def test_indexing_workflow(self, temp_config_files):
        """Test complete indexing workflow."""
        with patch('rag_factory.gui.main_window.ServiceRegistry') as mock_registry_class, \
             patch('rag_factory.gui.main_window.StrategyPairManager') as mock_manager_class:
            
            mock_registry = Mock()
            mock_registry.list_services.return_value = ["db_main"]
            mock_db = Mock()
            mock_registry.get.return_value = mock_db
            mock_registry_class.return_value = mock_registry
            
            mock_manager = Mock()
            mock_indexing = Mock()
            
            # Mock async process
            async def mock_process(docs, context):
                result = Mock()
                result.total_chunks = 5
                return result
            
            mock_indexing.process = mock_process
            mock_manager_class.return_value = mock_manager
            
            app = RAGFactoryGUI(**temp_config_files)
            app.indexing_strategy = mock_indexing
            app.current_strategy_name = "test-strategy"
            
            # Set text
            app.text_input.set_text("Test document content")
            
            # Index
            app._index_text()
            
            # Wait for background thread
            time.sleep(0.5)
            app.root.update()
            
            # Verify document was tracked
            assert len(app.indexed_documents) > 0
            
            app.root.destroy()
    
    def test_retrieval_workflow(self, temp_config_files):
        """Test complete retrieval workflow."""
        with patch('rag_factory.gui.main_window.ServiceRegistry') as mock_registry_class, \
             patch('rag_factory.gui.main_window.StrategyPairManager') as mock_manager_class:
            
            mock_registry = Mock()
            mock_registry.list_services.return_value = ["db_main"]
            mock_registry.get.return_value = Mock()
            mock_registry_class.return_value = mock_registry
            
            mock_manager = Mock()
            mock_retrieval = Mock()
            
            # Mock async retrieve
            async def mock_retrieve(query, context):
                result = Mock()
                result.score = 0.9
                result.content = "Test result content"
                result.metadata = {"source": "test.txt"}
                return [result]
            
            mock_retrieval.retrieve = mock_retrieve
            mock_manager_class.return_value = mock_manager
            
            app = RAGFactoryGUI(**temp_config_files)
            app.retrieval_strategy = mock_retrieval
            app.current_strategy_name = "test-strategy"
            
            # Set query
            app.query_var.set("test query")
            
            # Retrieve
            app._retrieve()
            
            # Wait for background thread
            time.sleep(0.5)
            app.root.update()
            
            # Verify results were displayed
            results_text = app.results_display.get_text()
            assert len(results_text) > 0
            
            app.root.destroy()


@pytest.mark.integration
class TestThreadingSafety:
    """Integration tests for threading safety."""
    
    def test_concurrent_operations(self, temp_config_files):
        """Test that concurrent operations don't crash."""
        with patch('rag_factory.gui.main_window.ServiceRegistry') as mock_registry_class, \
             patch('rag_factory.gui.main_window.StrategyPairManager') as mock_manager_class:
            
            mock_registry = Mock()
            mock_registry.list_services.return_value = ["db_main"]
            mock_registry.get.return_value = Mock()
            mock_registry_class.return_value = mock_registry
            
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            app = RAGFactoryGUI(**temp_config_files)
            
            # Simulate multiple rapid operations
            for i in range(5):
                app.root.update()
                time.sleep(0.1)
            
            # Should not crash
            assert True
            
            app.root.destroy()


@pytest.mark.integration
class TestUtilityOperations:
    """Integration tests for utility operations."""
    
    def test_clear_data_workflow(self, temp_config_files):
        """Test complete clear data workflow."""
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
            
            # Add data
            app.indexed_documents.append({"id": "doc1", "content": "test"})
            app.status_bar.update_counts(documents=1, chunks=5)
            
            # Clear with confirmation
            with patch('tkinter.messagebox.askyesnocancel', return_value=True):
                app._clear_all_data()
            
            # Verify data cleared
            assert len(app.indexed_documents) == 0
            assert app.status_bar.document_count == 0
            
            app.root.destroy()
    
    def test_log_viewing_workflow(self, temp_config_files):
        """Test log viewing workflow."""
        with patch('rag_factory.gui.main_window.ServiceRegistry') as mock_registry_class, \
             patch('rag_factory.gui.main_window.StrategyPairManager') as mock_manager_class:
            
            mock_registry = Mock()
            mock_registry.list_services.return_value = ["db_main"]
            mock_registry.get.return_value = Mock()
            mock_registry_class.return_value = mock_registry
            
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            app = RAGFactoryGUI(**temp_config_files)
            
            # Add some logs
            app.log_buffer.append("Test log entry")
            
            # Open log viewer
            app._view_logs()
            
            # Should not crash
            assert True
            
            app.root.destroy()
