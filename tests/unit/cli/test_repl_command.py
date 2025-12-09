import pytest
import typer
from typer.testing import CliRunner
from unittest.mock import Mock, patch
from rag_factory.cli.commands.repl import start_repl, REPLSession

class TestREPLCommand:
    """Test suite for REPL command."""
    
    @pytest.fixture
    def app(self) -> typer.Typer:
        """Create Typer app with command."""
        import typer
        app = typer.Typer()
        app.command(name="repl")(start_repl)
        return app

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI runner."""
        return CliRunner()
    
    @patch('rag_factory.cli.commands.repl.REPLSession')
    def test_repl_start(self, mock_session_cls: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test REPL startup."""
        mock_session = Mock()
        mock_session_cls.return_value = mock_session
        
        result = runner.invoke(app, [])
        
        print(f"Output: {result.output}")
        assert result.exit_code == 0
        mock_session.run.assert_called_once()
    
    @patch('rag_factory.cli.commands.repl.REPLSession')
    def test_repl_with_config(self, mock_session_cls: Mock, runner: CliRunner, app: typer.Typer) -> None:
        """Test REPL startup with config."""
        mock_session = Mock()
        mock_session_cls.return_value = mock_session
        
        result = runner.invoke(app, ['--config', 'config.yaml'])
        
        assert result.exit_code == 0
        mock_session_cls.assert_called_once_with('config.yaml')
        mock_session.run.assert_called_once()

class TestREPLSession:
    """Test suite for REPLSession class."""
    
    @patch('rag_factory.cli.commands.repl.validate_config_file')
    def test_repl_session_init(self, mock_validate: Mock) -> None:
        """Test REPL session initialization."""
        mock_validate.return_value = {'strategy_name': 'test'}
        
        session = REPLSession(config_path='config.yaml')
        
        assert session.config == {'strategy_name': 'test'}
        mock_validate.assert_called_once_with('config.yaml')
    
    def test_repl_set_command(self) -> None:
        """Test 'set' command in REPL."""
        session = REPLSession()
        
        # Test setting valid parameter
        # Note: _handle_set only supports 'strategy' and 'index_dir' in the current implementation
        session._handle_set(['strategy', 'new_strategy'])
        assert session.current_strategy == 'new_strategy'
        
        # Test setting invalid parameter format
        with patch('rag_factory.cli.commands.repl.print_error') as mock_print_error:
            session._handle_set(['invalid_format'])
            mock_print_error.assert_called_with("Usage: set <key> <value>")
