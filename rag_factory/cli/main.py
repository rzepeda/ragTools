"""Main entry point for the RAG Factory CLI.

This module sets up the Typer application and registers all commands.
"""

import sys
from typing import Optional

import typer
from rich.console import Console

console = Console()

app = typer.Typer(
    name="rag-factory",
    help="RAG Factory CLI - Development tool for testing strategies",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    """Show version information."""
    if value:
        from rag_factory.__version__ import __version__
        console.print(f"RAG Factory CLI version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode with verbose output",
    ),
) -> None:
    """RAG Factory CLI - Development tool for testing RAG strategies."""
    if debug:
        console.print("[dim]Debug mode enabled[/dim]")


# Import and register commands
try:
    from rag_factory.cli.commands import (
        benchmark,
        check_consistency,
        config,
        gui,
        index,
        query,
        repl,
        strategies,
        validate_pipeline,
        validate_e2e,
    )

    app.command(name="gui")(gui.gui_command)
    app.command(name="index")(index.index_command)
    app.command(name="query")(query.query_command)
    app.command(name="strategies")(strategies.list_strategies)
    app.command(name="config")(config.validate_config)
    app.command(name="benchmark")(benchmark.run_benchmark)
    app.command(name="repl")(repl.start_repl)
    app.command(name="validate-pipeline")(validate_pipeline.validate_pipeline)
    app.command(name="check-consistency")(check_consistency.check_consistency)
    app.command(name="validate-e2e")(validate_e2e.validate_e2e)


except ImportError as e:
    console.print(f"[red]Error importing commands: {e}[/red]")
    sys.exit(1)


if __name__ == "__main__":
    app()
