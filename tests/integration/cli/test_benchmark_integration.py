import json
import pytest
import typer
from pathlib import Path
from typer.testing import CliRunner
from rag_factory.cli.commands.benchmark import run_benchmark

class TestBenchmarkIntegration:
    """Integration tests for benchmark command."""
    
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
    
    @pytest.fixture
    def dataset_file(self, tmp_path: Path) -> Path:
        """Create a temporary benchmark dataset."""
        dataset = [
            {
                "query": "What is RAG?",
                "expected_docs": ["doc1.txt"],
                "metadata": {"category": "intro"}
            },
            {
                "query": "How does indexing work?",
                "expected_docs": ["doc2.txt"],
                "metadata": {"category": "indexing"}
            }
        ]
        
        dataset_path = tmp_path / "dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(dataset, f)
        
        return dataset_path
    
    @pytest.fixture
    def index_dir(self, tmp_path: Path) -> Path:
        """Create a temporary index directory."""
        index_path = tmp_path / "rag_index"
        index_path.mkdir()
        return index_path

    def test_benchmark_execution_flow(self, runner: CliRunner, dataset_file: Path, index_dir: Path, tmp_path: Path, app: typer.Typer) -> None:
        """Test full benchmark execution flow."""
        output_file = tmp_path / "results.json"
        
        # We need to mock RAGFactory.list_strategies since we don't have a real registry in this env
        # But for integration tests we usually want to use real components. 
        # However, since the command implementation mocks the actual execution (time.sleep),
        # we can rely on that behavior.
        # We just need to ensure strategies are available or mocked if the registry is empty.
        
        from unittest.mock import patch
        with patch('rag_factory.factory.RAGFactory.list_strategies', return_value=['test_strategy']):
            result = runner.invoke(app, [
                str(dataset_file),
                '--index', str(index_dir),
                '--output', str(output_file),
                '--iterations', '1'
            ])
            
            assert result.exit_code == 0
            assert 'Running benchmarks' in result.output
            assert 'test_strategy' in result.output
            assert output_file.exists()
            
            with open(output_file) as f:
                results = json.load(f)
                assert 'test_strategy' in results
                assert results['test_strategy']['total_queries'] == 2
