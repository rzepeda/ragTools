"""Strategies listing command."""

import sys
from typing import Optional

import typer
from rich.console import Console

from rag_factory.cli.formatters import print_error
from rag_factory.cli.formatters.results import format_strategy_list
from rag_factory.factory import RAGFactory

console = Console()


def list_strategies(
    strategy_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter strategies by type (e.g., 'chunking', 'reranking', 'query_expansion')",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information about each strategy",
    ),
) -> None:
    """
    List all available RAG strategies.

    This command displays all registered strategies in the RAG Factory,
    grouped by type. You can filter by strategy type to see only specific
    categories.

    Examples:

        List all strategies:
        $ rag-factory strategies

        List only chunking strategies:
        $ rag-factory strategies --type chunking

        List with verbose details:
        $ rag-factory strategies --verbose
    """
    try:
        console.print("[bold]Available RAG Strategies[/bold]\n")

        # Get registered strategies from factory
        strategy_names = RAGFactory.list_strategies()

        if not strategy_names:
            print_error(
                "No strategies registered.\n\n"
                "This might indicate an issue with the library setup.\n"
                "Please ensure strategies are properly registered."
            )
            raise typer.Exit(1)

        # Build strategy information list
        # In a real implementation, we would query each strategy for its metadata
        strategies_info = []
        for name in strategy_names:
            # Infer type from name (in real implementation, would query strategy class)
            if "chunk" in name.lower():
                strategy_type_inferred = "chunking"
            elif "rerank" in name.lower():
                strategy_type_inferred = "reranking"
            elif "expand" in name.lower() or "hyde" in name.lower():
                strategy_type_inferred = "query_expansion"
            else:
                strategy_type_inferred = "other"

            strategies_info.append({
                "name": name,
                "type": strategy_type_inferred,
                "description": f"Strategy for {strategy_type_inferred}",
            })

        # Filter by type if specified
        if strategy_type:
            strategies_info = [
                s for s in strategies_info
                if s["type"].lower() == strategy_type.lower()
            ]

            if not strategies_info:
                print_error(
                    f"No strategies found for type: {strategy_type}\n\n"
                    f"Available types: chunking, reranking, query_expansion, other"
                )
                raise typer.Exit(1)

        # Display strategies
        tree = format_strategy_list(strategies_info, strategy_type)
        console.print(tree)

        # Show summary
        console.print(
            f"\n[dim]Total: {len(strategies_info)} strateg{'ies' if len(strategies_info) != 1 else 'y'}[/dim]"
        )

        if verbose:
            console.print("\n[bold]Strategy Details:[/bold]")
            for strategy in strategies_info:
                console.print(f"\n[cyan]{strategy['name']}[/cyan]")
                console.print(f"  Type: {strategy['type']}")
                console.print(f"  Description: {strategy['description']}")

        console.print(
            f"\n[dim]Use 'rag-factory index --strategy <name>' to use a strategy[/dim]"
        )

    except Exception as e:
        if "--debug" in sys.argv:
            raise
        print_error(f"Failed to list strategies: {e}")
        raise typer.Exit(1)
