import pytest
import typer
from typer.testing import CliRunner
from unittest.mock import Mock, patch
from rag_factory.cli.commands.strategies import list_strategies

class TestListStrategiesCommand:
    """Test suite for list-strategies command."""
    
    @pytest.fixture
    def app(self) -> typer.Typer:
        """Create Typer app with command."""
        import typer
        app = typer.Typer()
        app.command(name="strategies")(list_strategies)
        return app

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner."""
        return CliRunner()
    
    @patch('rag_factory.factory.RAGFactory.list_strategies')
    def test_list_all_strategies(self, mock_list: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test listing all strategies."""
        mock_list.return_value = ['basic_chunker', 'semantic_chunker', 'reranking']
        
        result = runner.invoke(app, [])
        
        assert result.exit_code == 0
        assert 'Available RAG Strategies' in result.output
        assert 'basic_chunker' in result.output
        assert 'semantic_chunker' in result.output
        assert 'reranking' in result.output
    
    @patch('rag_factory.factory.RAGFactory.list_strategies')
    def test_list_strategies_empty(self, mock_list: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test listing strategies when none are registered."""
        mock_list.return_value = []
        
        result = runner.invoke(app, [])
        
        assert result.exit_code == 1
        assert 'No strategies registered' in result.output
    
    @patch('rag_factory.factory.RAGFactory.list_strategies')
    def test_filter_strategies_by_type(self, mock_list: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test filtering strategies by type."""
        mock_list.return_value = ['basic_chunker', 'semantic_chunker', 'reranking']
        
        result = runner.invoke(app, ['--type', 'chunking'])
        
        assert result.exit_code == 0
        assert 'basic_chunker' in result.output
        assert 'semantic_chunker' in result.output
        # reranking should be filtered out based on the implementation logic
        # which infers type from name
        assert 'reranking' not in result.output
    
    @patch('rag_factory.factory.RAGFactory.list_strategies')
    def test_filter_strategies_no_match(self, mock_list: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test filtering strategies with no matches."""
        mock_list.return_value = ['basic_chunker']
        
        result = runner.invoke(app, ['--type', 'reranking'])
        
        assert result.exit_code == 1
        assert 'No strategies found for type' in result.output
    
    @patch('rag_factory.factory.RAGFactory.list_strategies')
    def test_list_strategies_verbose(self, mock_list: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test listing strategies with verbose output."""
        mock_list.return_value = ['basic_chunker']
        
        result = runner.invoke(app, ['--verbose'])
        
        assert result.exit_code == 0
        assert 'Strategy Details' in result.output
        assert 'Description:' in result.output
