"""GUI command for launching the RAG Factory GUI application.

This module provides a CLI command to launch the tkinter-based GUI
for testing RAG strategy pairs.
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from rag_factory.gui.main_window import RAGFactoryGUI

console = Console()


def gui_command(
    config: str = typer.Option(
        "config/services.yaml",
        "--config",
        "-c",
        help="Path to services configuration file",
    ),
    strategies_dir: str = typer.Option(
        "strategies",
        "--strategies",
        "-s",
        help="Directory containing strategy YAML files",
    ),
    alembic_config: str = typer.Option(
        "alembic.ini",
        "--alembic-config",
        "-a",
        help="Path to Alembic configuration file",
    ),
) -> None:
    """Launch the RAG Factory GUI for strategy pair testing.
    
    This command starts a graphical user interface that allows you to:
    - Load and preview strategy pair configurations
    - Index text and files using selected strategies
    - Query indexed data and view retrieval results
    - Monitor operation status and manage indexed data
    
    The GUI initializes the ServiceRegistry and StrategyPairManager internally
    using the provided configuration paths.
    
    Examples:
    
        Launch GUI with default configuration:
        $ rag-factory gui
        
        Launch with custom config and strategies directory:
        $ rag-factory gui --config my_config.yaml --strategies my_strategies/
        
        Launch with custom Alembic config:
        $ rag-factory gui --alembic-config custom_alembic.ini
    """
    try:
        # Validate paths
        config_path = Path(config)
        strategies_path = Path(strategies_dir)
        alembic_path = Path(alembic_config)
        
        # Note: We don't fail if paths don't exist - the GUI will handle this gracefully
        # and show appropriate error messages to the user
        
        console.print("[cyan]Initializing RAG Factory GUI...[/cyan]")
        
        # Create and run GUI
        # The GUI will initialize ServiceRegistry and StrategyPairManager internally
        console.print("[green]Launching GUI...[/green]\n")
        app = RAGFactoryGUI(
            config_path=str(config_path),
            strategies_dir=str(strategies_path),
            alembic_config=str(alembic_path)
        )
        app.run()
        
        console.print("\n[cyan]GUI closed. Goodbye![/cyan]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]GUI launch cancelled by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        if "--debug" in sys.argv:
            raise
        console.print(f"[red]Error launching GUI: {e}[/red]")
        raise typer.Exit(1)

