"""Consistency check result formatters for CLI."""

from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel


def format_consistency_results(
    results: Dict[str, Dict[str, Any]],
    verbose: bool,
    console: Console
) -> None:
    """
    Format and display consistency check results with rich formatting.

    Args:
        results: Dict mapping strategy names to check results from factory.check_all_strategies()
        verbose: Whether to show detailed information
        console: Rich console for output
    """
    console.print()
    console.print("[bold]Strategy Consistency Check Results[/bold]")
    console.print()

    # Group strategies by type
    indexing_strategies = {}
    retrieval_strategies = {}
    unknown_strategies = {}

    for name, result in results.items():
        if result['type'] == 'indexing':
            indexing_strategies[name] = result
        elif result['type'] == 'retrieval':
            retrieval_strategies[name] = result
        else:
            unknown_strategies[name] = result

    # Display indexing strategies
    if indexing_strategies:
        _display_strategy_group(
            "Indexing Strategies",
            indexing_strategies,
            verbose,
            console
        )

    # Display retrieval strategies
    if retrieval_strategies:
        _display_strategy_group(
            "Retrieval Strategies",
            retrieval_strategies,
            verbose,
            console
        )

    # Display unknown strategies if they have errors or in verbose mode
    if unknown_strategies:
        # Always show if there are errors, otherwise only in verbose mode
        has_errors = any(s.get('error') for s in unknown_strategies.values())
        if has_errors or verbose:
            _display_strategy_group(
                "Other Strategies",
                unknown_strategies,
                verbose,
                console
            )

    # Display summary
    _display_summary(results, console)


def _display_strategy_group(
    group_name: str,
    strategies: Dict[str, Dict[str, Any]],
    verbose: bool,
    console: Console
) -> None:
    """Display a group of strategies with their consistency status."""
    console.print(f"[bold cyan]{group_name}:[/bold cyan]")
    console.print()

    for name, result in sorted(strategies.items()):
        # Check for errors
        if result.get('error'):
            console.print(f"  [red]✗[/red] {name}")
            console.print(f"    [dim red]Error: {result['error']}[/dim red]")
            console.print()
            continue

        # Check for warnings
        warnings = result.get('warnings', [])
        if warnings:
            console.print(f"  [yellow]⚠[/yellow] {name}")
            for warning in warnings:
                # Remove emoji from warning if present (we add our own)
                clean_warning = warning.replace('⚠️', '').strip()
                console.print(f"    [yellow]•[/yellow] [dim]{clean_warning}[/dim]")
            console.print()
        else:
            console.print(f"  [green]✓[/green] {name}")
            if verbose:
                console.print("    [dim]No consistency issues found[/dim]")
            console.print()


def _display_summary(results: Dict[str, Dict[str, Any]], console: Console) -> None:
    """Display summary statistics."""
    total = len(results)
    consistent = sum(1 for r in results.values() if not r.get('warnings') and not r.get('error'))
    warnings = sum(1 for r in results.values() if r.get('warnings'))
    errors = sum(1 for r in results.values() if r.get('error'))

    summary_lines = [
        f"[bold]Total strategies checked:[/bold] {total}",
        f"[green]✓ Consistent:[/green] {consistent}",
    ]

    if warnings > 0:
        summary_lines.append(f"[yellow]⚠ With warnings:[/yellow] {warnings}")

    if errors > 0:
        summary_lines.append(f"[red]✗ With errors:[/red] {errors}")

    panel = Panel(
        "\n".join(summary_lines),
        border_style="blue",
        title="Summary"
    )

    console.print(panel)
    console.print()

    # Add informational note
    if warnings > 0:
        console.print(
            "[dim]Note: Warnings indicate potential misconfigurations but do not block usage.[/dim]"
        )
        console.print()
