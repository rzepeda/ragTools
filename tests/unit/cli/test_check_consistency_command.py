"""Unit tests for check-consistency command."""

import pytest
from typer.testing import CliRunner
from unittest.mock import Mock, patch

from rag_factory.cli.main import app


@pytest.fixture
def cli_runner():
    """Create CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_factory():
    """Create mock factory."""
    factory = Mock()
    factory.dependencies = Mock()
    return factory


class TestCheckConsistencyCommand:
    """Tests for check-consistency command."""

    def test_command_help(self, cli_runner):
        """Test command help displays correctly."""
        result = cli_runner.invoke(app, ["check-consistency", "--help"])
        
        assert result.exit_code == 0
        assert "check-consistency" in result.stdout.lower() or "check consistency" in result.stdout.lower()
        assert "--strategies" in result.stdout or "-s" in result.stdout
        assert "--type" in result.stdout or "-t" in result.stdout
        assert "--verbose" in result.stdout or "-v" in result.stdout

    @patch("rag_factory.cli.commands.check_consistency.RAGFactory")
    def test_all_strategies_consistent(self, mock_factory_cls, cli_runner, mock_factory):
        """Test output when all strategies are consistent."""
        # Setup mock
        mock_factory_cls.return_value = mock_factory
        mock_factory.check_all_strategies = Mock(return_value={
            'strategy1': {
                'type': 'indexing',
                'warnings': [],
                'error': None
            },
            'strategy2': {
                'type': 'retrieval',
                'warnings': [],
                'error': None
            }
        })
        
        # Run command
        result = cli_runner.invoke(app, ["check-consistency"])
        
        # Should exit with 0 even when all consistent
        assert result.exit_code == 0
        assert "strategy1" in result.stdout
        assert "strategy2" in result.stdout

    @patch("rag_factory.cli.commands.check_consistency.RAGFactory")
    def test_strategies_with_warnings(self, mock_factory_cls, cli_runner, mock_factory):
        """Test output when strategies have warnings."""
        # Setup mock
        mock_factory_cls.return_value = mock_factory
        mock_factory.check_all_strategies = Mock(return_value={
            'bad_strategy': {
                'type': 'indexing',
                'warnings': [
                    '⚠️ BadStrategy: Produces VECTORS but doesn\'t require EMBEDDING service.'
                ],
                'error': None
            },
            'good_strategy': {
                'type': 'indexing',
                'warnings': [],
                'error': None
            }
        })
        
        # Run command
        result = cli_runner.invoke(app, ["check-consistency"])
        
        # Should exit with 0 even with warnings (warnings don't block)
        assert result.exit_code == 0
        assert "bad_strategy" in result.stdout
        assert "good_strategy" in result.stdout

    @patch("rag_factory.cli.commands.check_consistency.RAGFactory")
    def test_type_filter_indexing(self, mock_factory_cls, cli_runner, mock_factory):
        """Test filtering by indexing type."""
        # Setup mock
        mock_factory_cls.return_value = mock_factory
        mock_factory.check_all_strategies = Mock(return_value={
            'indexing_strategy': {
                'type': 'indexing',
                'warnings': [],
                'error': None
            }
        })
        
        # Run command with type filter
        result = cli_runner.invoke(app, ["check-consistency", "--type", "indexing"])
        
        assert result.exit_code == 0
        # Verify the factory method was called with correct filter
        mock_factory.check_all_strategies.assert_called_once()
        call_kwargs = mock_factory.check_all_strategies.call_args[1]
        assert call_kwargs['type_filter'] == 'indexing'

    @patch("rag_factory.cli.commands.check_consistency.RAGFactory")
    def test_type_filter_retrieval(self, mock_factory_cls, cli_runner, mock_factory):
        """Test filtering by retrieval type."""
        # Setup mock
        mock_factory_cls.return_value = mock_factory
        mock_factory.check_all_strategies = Mock(return_value={
            'retrieval_strategy': {
                'type': 'retrieval',
                'warnings': [],
                'error': None
            }
        })
        
        # Run command with type filter
        result = cli_runner.invoke(app, ["check-consistency", "--type", "retrieval"])
        
        assert result.exit_code == 0
        # Verify the factory method was called with correct filter
        call_kwargs = mock_factory.check_all_strategies.call_args[1]
        assert call_kwargs['type_filter'] == 'retrieval'

    @patch("rag_factory.cli.commands.check_consistency.RAGFactory")
    def test_invalid_type_filter(self, mock_factory_cls, cli_runner, mock_factory):
        """Test error handling for invalid type filter."""
        # Setup mock
        mock_factory_cls.return_value = mock_factory
        
        # Run command with invalid type
        result = cli_runner.invoke(app, ["check-consistency", "--type", "invalid"])
        
        assert result.exit_code == 1

    @patch("rag_factory.cli.commands.check_consistency.RAGFactory")
    @patch("rag_factory.cli.commands.check_consistency.parse_strategy_list")
    def test_strategy_filter(self, mock_parse, mock_factory_cls, cli_runner, mock_factory):
        """Test filtering specific strategies."""
        # Setup mocks
        mock_factory_cls.return_value = mock_factory
        mock_parse.return_value = ['strategy1', 'strategy2']
        mock_factory.check_all_strategies = Mock(return_value={
            'strategy1': {
                'type': 'indexing',
                'warnings': [],
                'error': None
            },
            'strategy2': {
                'type': 'retrieval',
                'warnings': [],
                'error': None
            }
        })
        
        # Run command with strategy filter
        result = cli_runner.invoke(
            app,
            ["check-consistency", "--strategies", "strategy1,strategy2"]
        )
        
        assert result.exit_code == 0
        # Verify parse was called
        mock_parse.assert_called_once_with("strategy1,strategy2")
        # Verify factory method was called with filtered list
        call_kwargs = mock_factory.check_all_strategies.call_args[1]
        assert call_kwargs['strategy_filter'] == ['strategy1', 'strategy2']

    @patch("rag_factory.cli.commands.check_consistency.RAGFactory")
    def test_verbose_mode(self, mock_factory_cls, cli_runner, mock_factory):
        """Test verbose mode shows additional information."""
        # Setup mock
        mock_factory_cls.return_value = mock_factory
        mock_factory.check_all_strategies = Mock(return_value={
            'strategy1': {
                'type': 'indexing',
                'warnings': [],
                'error': None
            }
        })
        
        # Run command with verbose flag
        result = cli_runner.invoke(app, ["check-consistency", "--verbose"])
        
        assert result.exit_code == 0
        # Verbose mode should show additional details
        assert "Checking all registered strategies" in result.stdout or "Type filter" in result.stdout

    @patch("rag_factory.cli.commands.check_consistency.validate_config_file")
    @patch("rag_factory.cli.commands.check_consistency.RAGFactory")
    def test_config_file_loading(
        self,
        mock_factory_cls,
        mock_validate_config,
        cli_runner,
        mock_factory
    ):
        """Test config file is loaded when provided."""
        # Setup mocks
        mock_validate_config.return_value = {"some": "config"}
        mock_factory_cls.return_value = mock_factory
        mock_factory.check_all_strategies = Mock(return_value={})
        
        # Run command with config
        result = cli_runner.invoke(
            app,
            ["check-consistency", "--config", "config.yaml"]
        )
        
        # Verify config was loaded
        mock_validate_config.assert_called_once_with("config.yaml")

    @patch("rag_factory.cli.commands.check_consistency.RAGFactory")
    def test_system_error_returns_exit_code_1(self, mock_factory_cls, cli_runner):
        """Test system errors return exit code 1."""
        # Setup mock to raise error
        mock_factory_cls.side_effect = Exception("System error")
        
        # Run command
        result = cli_runner.invoke(app, ["check-consistency"])
        
        assert result.exit_code == 1

    @patch("rag_factory.cli.commands.check_consistency.RAGFactory")
    def test_strategies_with_errors(self, mock_factory_cls, cli_runner, mock_factory):
        """Test output when strategies have instantiation errors."""
        # Setup mock
        mock_factory_cls.return_value = mock_factory
        mock_factory.check_all_strategies = Mock(return_value={
            'broken_strategy': {
                'type': 'unknown',
                'warnings': [],
                'error': 'Could not instantiate strategy: Missing required service'
            },
            'good_strategy': {
                'type': 'indexing',
                'warnings': [],
                'error': None
            }
        })
        
        # Run command
        result = cli_runner.invoke(app, ["check-consistency"])
        
        # Should still exit with 0 (errors in strategies, not system errors)
        assert result.exit_code == 0
        assert "broken_strategy" in result.stdout
        assert "good_strategy" in result.stdout
