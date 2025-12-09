import pytest
import typer
from typer.testing import CliRunner
from unittest.mock import Mock, patch
from rag_factory.cli.commands.benchmark import run_benchmark

class TestBenchmarkCommand:
    """Test suite for benchmark command."""
    
    @pytest.fixture
    def app(self) -> typer.Typer:
        """Create Typer app with command."""
        import typer
        app = typer.Typer()
        app.command(name="benchmark")(run_benchmark)
        return app

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner."""
        return CliRunner()
    
    @patch('rag_factory.cli.commands.benchmark.validate_path_exists')
    @patch('rag_factory.cli.commands.benchmark._load_benchmark_dataset')
    @patch('pathlib.Path.exists')
    @patch('rag_factory.factory.RAGFactory.list_strategies')
    def test_benchmark_default_strategies(self, mock_list: Mock, mock_exists: Mock, mock_load: Mock, mock_validate: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test benchmark with default strategies."""
        mock_validate.return_value = Mock()
        mock_load.return_value = [{'query': 'test', 'expected_docs': []}]
        mock_exists.return_value = True
        mock_list.return_value = ['strategy1', 'strategy2']
        
        result = runner.invoke(app, ['dataset.json'])
        
        assert result.exit_code == 0
        assert 'Running benchmarks' in result.output
        assert 'strategy1' in result.output
        assert 'strategy2' in result.output
        assert 'Benchmark completed' in result.output
    
    @patch('rag_factory.cli.commands.benchmark.validate_path_exists')
    @patch('rag_factory.cli.commands.benchmark._load_benchmark_dataset')
    @patch('pathlib.Path.exists')
    def test_benchmark_specific_strategies(self, mock_exists: Mock, mock_load: Mock, mock_validate: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test benchmark with specific strategies."""
        mock_validate.return_value = Mock()
        mock_load.return_value = [{'query': 'test', 'expected_docs': []}]
        mock_exists.return_value = True
        
        result = runner.invoke(app, [
            'dataset.json',
            '--strategies', 'strategy1'
        ])
        
        assert result.exit_code == 0
        assert 'strategy1' in result.output
    
    @patch('rag_factory.cli.commands.benchmark.validate_path_exists')
    @patch('rag_factory.cli.commands.benchmark._load_benchmark_dataset')
    @patch('pathlib.Path.exists')
    @patch('rag_factory.factory.RAGFactory.list_strategies')
    @patch('rag_factory.cli.commands.benchmark._export_results')
    def test_benchmark_export_results(self, mock_export: Mock, mock_list: Mock, mock_exists: Mock, mock_load: Mock, mock_validate: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test benchmark with result export."""
        mock_validate.return_value = Mock()
        mock_load.return_value = [{'query': 'test', 'expected_docs': []}]
        mock_exists.return_value = True
        mock_list.return_value = ['strategy1']
        
        result = runner.invoke(app, [
            'dataset.json',
            '--output', 'results.json'
        ])
        
        assert result.exit_code == 0
        assert 'Exporting results' in result.output
        mock_export.assert_called_once()
    
    @patch('rag_factory.cli.commands.benchmark.validate_path_exists')
    def test_benchmark_missing_dataset(self, mock_validate: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test benchmark with missing dataset."""
        mock_validate.side_effect = ValueError("File not found")
        
        result = runner.invoke(app, ['missing.json'])
        
        assert result.exit_code == 1
        assert 'File not found' in result.output
    
    @patch('rag_factory.cli.commands.benchmark.validate_path_exists')
    @patch('rag_factory.cli.commands.benchmark._load_benchmark_dataset')
    @patch('pathlib.Path.exists')
    def test_benchmark_missing_index(self, mock_exists: Mock, mock_load: Mock, mock_validate: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test benchmark when index is missing."""
        mock_validate.return_value = Mock()
        mock_load.return_value = []
        mock_exists.return_value = False
        
        result = runner.invoke(app, ['dataset.json'])
        
        assert result.exit_code == 1
        assert 'Index directory not found' in result.output
