import pytest
from typer.testing import CliRunner
from unittest.mock import Mock, patch
from rag_factory.cli.commands.index import index_command
import typer

class TestIndexCommand:
    """Test suite for index command."""
    
    @pytest.fixture
    def app(self) -> typer.Typer:
        """Create Typer app with command."""
        import typer
        app = typer.Typer()
        app.command(name="index")(index_command)
        return app

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner."""
        return CliRunner()
    
    @patch('rag_factory.cli.commands.index.validate_path_exists')
    @patch('rag_factory.cli.commands.index._collect_documents')
    @patch('rag_factory.cli.commands.index.validate_config_file')
    def test_index_success(self, mock_config: Mock, mock_collect: Mock, mock_validate: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test successful indexing command execution."""
        mock_validate.return_value = Mock()
        mock_collect.return_value = [Mock()]
        mock_config.return_value = {}
        
        result = runner.invoke(app, [
            'test_docs/',
            '--strategy', 'fixed_size_chunker',
            '--chunk-size', '512'
        ])
        
        assert result.exit_code == 0
        assert 'Successfully indexed' in result.output
    
    @patch('rag_factory.cli.commands.index.validate_path_exists')
    def test_index_with_invalid_path(self, mock_validate: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test index command with invalid path."""
        mock_validate.side_effect = ValueError("Path not found")
        
        result = runner.invoke(app, [
            'nonexistent_path/'
        ])
        
        assert result.exit_code == 1
        assert 'Path not found' in result.output
    
    @patch('rag_factory.cli.commands.index.validate_path_exists')
    @patch('rag_factory.cli.commands.index._collect_documents')
    def test_index_no_documents_found(self, mock_collect: Mock, mock_validate: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test index command when no documents are found."""
        mock_validate.return_value = Mock()
        mock_collect.return_value = []
        
        result = runner.invoke(app, [
            'empty_dir/'
        ])
        
        assert result.exit_code == 1
        assert 'No documents found' in result.output
    
    @patch('rag_factory.cli.commands.index.validate_path_exists')
    @patch('rag_factory.cli.commands.index._collect_documents')
    @patch('rag_factory.cli.commands.index.validate_config_file')
    def test_index_with_config(self, mock_config: Mock, mock_collect: Mock, mock_validate: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test index command with configuration file."""
        mock_validate.return_value = Mock()
        mock_collect.return_value = [Mock()]
        mock_config.return_value = {'chunk_size': 1024}
        
        result = runner.invoke(app, [
            'test_docs/',
            '--config', 'config.yaml'
        ])
        
        assert result.exit_code == 0
        assert 'Loading configuration' in result.output
        mock_config.assert_called_once_with('config.yaml')

    @patch('rag_factory.cli.commands.index.validate_path_exists')
    @patch('rag_factory.cli.commands.index._collect_documents')
    def test_index_output_directory(self, mock_collect: Mock, mock_validate: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test index command with custom output directory."""
        mock_validate.return_value = Mock()
        mock_collect.return_value = [Mock()]
        
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            result = runner.invoke(app, [
                'test_docs/',
                '--output', 'custom_index'
            ])
            
            assert result.exit_code == 0
            assert 'Output Directory' in result.output
            assert 'custom_index' in result.output
