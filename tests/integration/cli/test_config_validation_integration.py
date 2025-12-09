import pytest
import yaml
import typer
from pathlib import Path
from typer.testing import CliRunner
from rag_factory.cli.commands.config import validate_config

class TestConfigValidationIntegration:
    """Integration tests for config validation command."""
    
    @pytest.fixture
    def app(self) -> typer.Typer:
        """Create Typer app with command."""
        import typer
        app = typer.Typer()
        app.command(name="config")(validate_config)
        return app

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner."""
        return CliRunner()
    
    @pytest.fixture
    def valid_config_file(self, tmp_path: Path) -> Path:
        """Create a valid configuration file."""
        config = {
            'strategy_name': 'semantic_chunking',
            'chunk_size': 512,
            'chunk_overlap': 50,
            'embedding_model': 'text-embedding-3-small'
        }
        
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        return config_path
    
    @pytest.fixture
    def invalid_config_file(self, tmp_path: Path) -> Path:
        """Create an invalid configuration file."""
        config = {
            'chunk_size': 512
            # Missing strategy_name
        }
        
        config_path = tmp_path / "invalid_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        return config_path

    def test_validate_valid_config_flow(self, runner: CliRunner, valid_config_file: Path, app: typer.Typer) -> None:
        """Test validation flow for a valid config file."""
        result = runner.invoke(app, [str(valid_config_file), '--show'])
        
        assert result.exit_code == 0
        assert 'File format is valid' in result.output
        assert 'No issues found' in result.output
        assert 'semantic_chunking' in result.output
    
    def test_validate_invalid_config_flow(self, runner: CliRunner, invalid_config_file: Path, app: typer.Typer) -> None:
        """Test validation flow for an invalid config file."""
        result = runner.invoke(app, [str(invalid_config_file)])
        
        assert result.exit_code == 1
        assert 'Validation failed with errors' in result.output
        assert 'Missing required field' in result.output
