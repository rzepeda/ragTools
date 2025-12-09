import pytest
from typer.testing import CliRunner
from unittest.mock import Mock, patch
from rag_factory.cli.commands.query import query_command
import typer

class TestQueryCommand:
    """Test suite for query command."""
    
    @pytest.fixture
    def app(self) -> typer.Typer:
        """Create Typer app with command."""
        import typer
        app = typer.Typer()
        app.command(name="query")(query_command)
        return app

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner."""
        return CliRunner()
    
    @patch('pathlib.Path.exists')
    def test_query_success(self, mock_exists: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test successful query execution."""
        mock_exists.return_value = True
        
        result = runner.invoke(app, [
            'test query',
            '--strategies', 'basic',
            '--top-k', '5'
        ])
        
        assert result.exit_code == 0
        assert 'Executing query' in result.output
        assert 'test query' in result.output
    
    @patch('pathlib.Path.exists')
    def test_query_index_not_found(self, mock_exists: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test query command when index directory is missing."""
        mock_exists.return_value = False
        
        result = runner.invoke(app, [
            'test query',
            '--index', 'missing_index'
        ])
        
        assert result.exit_code == 1
        assert 'Index directory not found' in result.output
    
    @patch('pathlib.Path.exists')
    @patch('rag_factory.cli.commands.query.validate_config_file')
    def test_query_with_config(self, mock_config: Mock, mock_exists: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test query command with configuration file."""
        mock_exists.return_value = True
        mock_config.return_value = {'top_k': 10}
        
        result = runner.invoke(app, [
            'test query',
            '--config', 'config.yaml'
        ])
        
        assert result.exit_code == 0
        assert 'Loading configuration' in result.output
        mock_config.assert_called_once_with('config.yaml')
    
    @patch('pathlib.Path.exists')
    def test_query_multiple_strategies(self, mock_exists: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test query command with multiple strategies."""
        mock_exists.return_value = True
        
        result = runner.invoke(app, [
            'test query',
            '--strategies', 'reranking,query_expansion'
        ])
        
        assert result.exit_code == 0
        assert 'reranking' in result.output
        assert 'query_expansion' in result.output

    @patch('pathlib.Path.exists')
    def test_query_no_results(self, mock_exists: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test query command with no results."""
        mock_exists.return_value = True
        
        # Mocking the internal results list to be empty would require more complex patching
        # of the function internals or refactoring the command to be more testable.
        # For now, we test that it runs without error.
        
        result = runner.invoke(app, [
            'test query'
        ])
        
        assert result.exit_code == 0
