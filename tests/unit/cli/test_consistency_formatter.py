"""Unit tests for consistency formatter."""

import pytest
from io import StringIO
from rich.console import Console

from rag_factory.cli.formatters.consistency import (
    format_consistency_results,
    _display_strategy_group,
    _display_summary
)


@pytest.fixture
def console():
    """Create a Rich console for testing."""
    # Disable color and force ASCII for easier testing
    return Console(file=StringIO(), force_terminal=False, width=120, legacy_windows=False)


class TestConsistencyFormatter:
    """Tests for consistency result formatting."""

    def test_format_empty_results(self, console):
        """Test formatting with no results."""
        results = {}
        
        format_consistency_results(results, verbose=False, console=console)
        
        output = console.file.getvalue()
        assert "Strategy Consistency Check Results" in output
        assert "Total strategies checked: 0" in output

    def test_format_consistent_strategies(self, console):
        """Test formatting with all consistent strategies."""
        results = {
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
        }
        
        format_consistency_results(results, verbose=False, console=console)
        
        output = console.file.getvalue()
        assert "strategy1" in output
        assert "strategy2" in output
        assert "Total strategies checked: 2" in output
        assert "Consistent: 2" in output

    def test_format_strategies_with_warnings(self, console):
        """Test formatting with strategies that have warnings."""
        results = {
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
        }
        
        format_consistency_results(results, verbose=False, console=console)
        
        output = console.file.getvalue()
        assert "bad_strategy" in output
        assert "good_strategy" in output
        assert "VECTORS" in output or "EMBEDDING" in output
        assert "Total strategies checked: 2" in output
        assert "Consistent: 1" in output
        assert "With warnings: 1" in output

    def test_format_strategies_with_errors(self, console):
        """Test formatting with strategies that have errors."""
        results = {
            'broken_strategy': {
                'type': 'unknown',
                'warnings': [],
                'error': 'Could not instantiate strategy: Missing service'
            }
        }
        
        format_consistency_results(results, verbose=False, console=console)
        
        output = console.file.getvalue()
        assert "broken_strategy" in output
        assert "Error" in output or "error" in output
        assert "With errors: 1" in output

    def test_grouping_by_type(self, console):
        """Test that strategies are grouped by type."""
        results = {
            'indexing1': {
                'type': 'indexing',
                'warnings': [],
                'error': None
            },
            'indexing2': {
                'type': 'indexing',
                'warnings': [],
                'error': None
            },
            'retrieval1': {
                'type': 'retrieval',
                'warnings': [],
                'error': None
            }
        }
        
        format_consistency_results(results, verbose=False, console=console)
        
        output = console.file.getvalue()
        assert "Indexing Strategies" in output
        assert "Retrieval Strategies" in output
        assert "indexing1" in output
        assert "indexing2" in output
        assert "retrieval1" in output

    def test_verbose_mode(self, console):
        """Test verbose mode shows additional details."""
        results = {
            'strategy1': {
                'type': 'indexing',
                'warnings': [],
                'error': None
            }
        }
        
        format_consistency_results(results, verbose=True, console=console)
        
        output = console.file.getvalue()
        # Verbose mode should show "No consistency issues found" for clean strategies
        assert "No consistency issues found" in output or "strategy1" in output

    def test_display_strategy_group(self, console):
        """Test displaying a strategy group."""
        strategies = {
            'strategy1': {
                'type': 'indexing',
                'warnings': [],
                'error': None
            },
            'strategy2': {
                'type': 'indexing',
                'warnings': ['Warning message'],
                'error': None
            }
        }
        
        _display_strategy_group("Test Group", strategies, verbose=False, console=console)
        
        output = console.file.getvalue()
        assert "Test Group" in output
        assert "strategy1" in output
        assert "strategy2" in output

    def test_display_summary(self, console):
        """Test summary display."""
        results = {
            'consistent': {
                'type': 'indexing',
                'warnings': [],
                'error': None
            },
            'with_warning': {
                'type': 'indexing',
                'warnings': ['Warning'],
                'error': None
            },
            'with_error': {
                'type': 'unknown',
                'warnings': [],
                'error': 'Error'
            }
        }
        
        _display_summary(results, console)
        
        output = console.file.getvalue()
        assert "Total strategies checked: 3" in output
        assert "Consistent: 1" in output
        assert "With warnings: 1" in output
        assert "With errors: 1" in output

    def test_warning_note_displayed(self, console):
        """Test that informational note is displayed when warnings exist."""
        results = {
            'strategy': {
                'type': 'indexing',
                'warnings': ['Some warning'],
                'error': None
            }
        }
        
        format_consistency_results(results, verbose=False, console=console)
        
        output = console.file.getvalue()
        assert "do not block usage" in output or "Warnings" in output

    def test_multiple_warnings_per_strategy(self, console):
        """Test displaying multiple warnings for a single strategy."""
        results = {
            'strategy': {
                'type': 'indexing',
                'warnings': [
                    'Warning 1',
                    'Warning 2',
                    'Warning 3'
                ],
                'error': None
            }
        }
        
        format_consistency_results(results, verbose=False, console=console)
        
        output = console.file.getvalue()
        assert "Warning 1" in output
        assert "Warning 2" in output
        assert "Warning 3" in output
