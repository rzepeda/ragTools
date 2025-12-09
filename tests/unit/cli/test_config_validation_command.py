import pytest
import typer
from typer.testing import CliRunner
from unittest.mock import Mock, patch
from rag_factory.cli.commands.config import validate_config

class TestConfigValidationCommand:
    """Test suite for config validation command."""
    
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
    
    @patch('rag_factory.cli.commands.config.validate_config_file')
    def test_validate_valid_config(self, mock_validate: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test validation of a valid configuration file."""
        mock_validate.return_value = {
            'strategy_name': 'test_strategy',
            'chunk_size': 512,
            'chunk_overlap': 50
        }
        
        result = runner.invoke(app, ['config.yaml'])
        
        assert result.exit_code == 0
        assert 'File format is valid' in result.output
        assert 'No issues found' in result.output
    
    @patch('rag_factory.cli.commands.config.validate_config_file')
    def test_validate_invalid_config_missing_field(self, mock_validate: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test validation with missing required field."""
        mock_validate.return_value = {
            'chunk_size': 512
        }
        
        result = runner.invoke(app, ['config.yaml'])
        
        assert result.exit_code == 1
        assert 'Missing required field' in result.output
        assert 'strategy_name' in result.output
    
    @patch('rag_factory.cli.commands.config.validate_config_file')
    def test_validate_invalid_config_values(self, mock_validate: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test validation with invalid values."""
        mock_validate.return_value = {
            'strategy_name': 'test',
            'chunk_size': -1
        }
        
        result = runner.invoke(app, ['config.yaml'])
        
        assert result.exit_code == 1
        assert 'chunk_size' in result.output
        assert 'must be positive' in result.output
    
    @patch('rag_factory.cli.commands.config.validate_config_file')
    def test_validate_config_warnings(self, mock_validate: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test validation with warnings."""
        mock_validate.return_value = {
            'strategy_name': 'test',
            'chunk_size': 4096  # Should trigger warning
        }
        
        result = runner.invoke(app, ['config.yaml'])
        
        assert result.exit_code == 0
        assert 'Validation warnings' in result.output
        assert 'Large chunk_size' in result.output
    
    @patch('rag_factory.cli.commands.config.validate_config_file')
    def test_validate_config_strict_mode(self, mock_validate: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test strict validation mode."""
        mock_validate.return_value = {
            'strategy_name': 'test',
            'chunk_size': 4096  # Should trigger warning
        }
        
        result = runner.invoke(app, ['config.yaml', '--strict'])
        
        assert result.exit_code == 1
        assert 'Strict mode enabled' in result.output
    
    @patch('rag_factory.cli.commands.config.validate_config_file')
    def test_validate_show_config(self, mock_validate: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test showing configuration contents."""
        mock_validate.return_value = {'strategy_name': 'test'}
        
        result = runner.invoke(app, ['config.yaml', '--show'])
        
        assert result.exit_code == 0
        assert 'Configuration contents' in result.output
        assert 'strategy_name' in result.output
