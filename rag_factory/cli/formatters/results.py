"""Result formatting utilities for queries, benchmarks, and strategy listings."""

from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich.tree import Tree

console = Console()


def format_query_results(
    results: List[Dict[str, Any]],
    query: str,
    strategy_name: Optional[str] = None,
) -> Table:
    """
    Format query results as a Rich table.

    Args:
        results: List of result dictionaries with 'text', 'score', and 'metadata'
        query: The original query string
        strategy_name: Optional strategy name used for the query

    Returns:
        Table: Rich table with formatted results
    """
    title = f"Query Results: '{query}'"
    if strategy_name:
        title += f" (Strategy: {strategy_name})"

    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Score", justify="right", style="green", width=10)
    table.add_column("Content", style="white", no_wrap=False)
    table.add_column("Source", style="dim", width=20)

    for idx, result in enumerate(results, 1):
        score = result.get("score", 0.0)
        text = result.get("text", "")
        metadata = result.get("metadata", {})
        source = metadata.get("source", "Unknown")

        # Truncate text if too long
        if len(text) > 200:
            text = text[:197] + "..."

        table.add_row(
            str(idx),
            f"{score:.4f}",
            text,
            source,
        )

    return table


def format_strategy_list(
    strategies: List[Dict[str, str]],
    filter_type: Optional[str] = None,
) -> Tree:
    """
    Format strategy list as a Rich tree grouped by type.

    Args:
        strategies: List of strategy dictionaries with 'name', 'type', 'description'
        filter_type: Optional type filter to show only specific types

    Returns:
        Tree: Rich tree with formatted strategy list
    """
    title = "Available Strategies"
    if filter_type:
        title += f" (Type: {filter_type})"

    tree = Tree(f"[bold cyan]{title}[/bold cyan]")

    # Group strategies by type
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for strategy in strategies:
        strategy_type = strategy.get("type", "unknown")
        if filter_type and strategy_type != filter_type:
            continue

        if strategy_type not in grouped:
            grouped[strategy_type] = []
        grouped[strategy_type].append(strategy)

    # Add grouped strategies to tree
    for strategy_type, type_strategies in sorted(grouped.items()):
        type_branch = tree.add(f"[bold yellow]{strategy_type.title()}[/bold yellow]")

        for strategy in sorted(type_strategies, key=lambda x: x.get("name", "")):
            name = strategy.get("name", "unknown")
            description = strategy.get("description", "No description available")

            type_branch.add(f"[green]{name}[/green]: {description}")

    return tree


def format_benchmark_results(
    results: Dict[str, Any],
    strategies: List[str],
) -> Table:
    """
    Format benchmark results as a comparison table.

    Args:
        results: Benchmark results dictionary with metrics per strategy
        strategies: List of strategy names that were benchmarked

    Returns:
        Table: Rich table with benchmark comparison
    """
    table = Table(
        title="Benchmark Results",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Strategy", style="bold white")
    table.add_column("Avg Latency (ms)", justify="right", style="yellow")
    table.add_column("Queries", justify="right", style="dim")
    table.add_column("Success Rate", justify="right", style="green")
    table.add_column("Avg Score", justify="right", style="cyan")

    for strategy in strategies:
        strategy_results = results.get(strategy, {})

        avg_latency = strategy_results.get("avg_latency_ms", 0.0)
        total_queries = strategy_results.get("total_queries", 0)
        success_rate = strategy_results.get("success_rate", 0.0)
        avg_score = strategy_results.get("avg_score", 0.0)

        table.add_row(
            strategy,
            f"{avg_latency:.2f}",
            str(total_queries),
            f"{success_rate * 100:.1f}%",
            f"{avg_score:.4f}",
        )

    return table


def format_statistics(stats: Dict[str, Any]) -> Table:
    """
    Format statistics as a simple key-value table.

    Args:
        stats: Dictionary of statistics

    Returns:
        Table: Rich table with formatted statistics
    """
    table = Table(
        title="Statistics",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
    )

    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="green")

    for key, value in stats.items():
        # Format key to be more readable
        readable_key = key.replace("_", " ").title()

        # Format value based on type
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        elif isinstance(value, int):
            formatted_value = f"{value:,}"
        else:
            formatted_value = str(value)

        table.add_row(readable_key, formatted_value)

    return table
