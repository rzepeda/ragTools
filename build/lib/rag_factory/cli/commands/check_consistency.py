"""Consistency checking command."""

import sys
from typing import Optional

import typer
from rich.console import Console

from rag_factory.cli.formatters.consistency import format_consistency_results
from rag_factory.cli.formatters import print_error
from rag_factory.cli.utils.validation import parse_strategy_list, validate_config_file
from rag_factory.factory import RAGFactory

console = Console()


def check_consistency(
    strategies: Optional[str] = typer.Option(
        None,
        "--strategies",
        "-s",
        help="Comma-separated strategy names to check (default: all)"
    ),
    type_filter: str = typer.Option(
        "all",
        "--type",
        "-t",
        help="Filter by strategy type: indexing, retrieval, or all"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed checking information"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file path (YAML or JSON)"
    ),
) -> None:
    """
    Check strategies for consistency.

    This command checks all registered strategies for consistency between
    their declared capabilities and service dependencies. It reports warnings
    for potential misconfigurations without blocking usage.

    Examples:

        Check all strategies:
        $ rag-factory check-consistency

        Check only indexing strategies:
        $ rag-factory check-consistency --type indexing

        Check specific strategies:
        $ rag-factory check-consistency --strategies "context_aware,vector_embedding"

        Verbose output:
        $ rag-factory check-consistency --verbose
    """
    try:
        console.print("\n[bold]Checking strategy consistency...[/bold]\n")

        # Validate type filter
        if type_filter not in ['indexing', 'retrieval', 'all']:
            print_error(f"Invalid type filter: {type_filter}. Must be 'indexing', 'retrieval', or 'all'")
            raise typer.Exit(1)

        # Parse strategy names if provided
        strategy_list = None
        if strategies:
            strategy_list = parse_strategy_list(strategies)
            if verbose:
                console.print(f"[dim]Checking strategies: {', '.join(strategy_list)}[/dim]")
        else:
            if verbose:
                console.print("[dim]Checking all registered strategies[/dim]")

        if verbose:
            console.print(f"[dim]Type filter: {type_filter}[/dim]")
            console.print()

        # Validate and load config if provided
        config_data = None
        if config:
            if verbose:
                console.print(f"[dim]Loading config from: {config}[/dim]")
            config_data = validate_config_file(config)

        # Create factory instance
        # Note: For consistency checking, we create a factory without services
        # The check will identify which services are required
        factory = RAGFactory(config=config_data) if config_data else RAGFactory()

        # Run consistency checks
        if verbose:
            console.print("[dim]Running consistency checks...[/dim]\n")

        results = factory.check_all_strategies(
            strategy_filter=strategy_list,
            type_filter=type_filter if type_filter != 'all' else None
        )

        # Display results
        format_consistency_results(results, verbose, console)

        # Exit with code 0 even if warnings are present
        # Warnings don't block usage
        sys.exit(0)

    except typer.Exit:
        raise
    except Exception as e:
        if "--debug" in sys.argv:
            raise
        print_error(f"Consistency check failed: {e}")
        raise typer.Exit(1)
