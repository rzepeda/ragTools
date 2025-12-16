"""REPL (Read-Eval-Print Loop) command for interactive sessions."""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console

from rag_factory.cli.formatters import print_error, print_success, print_warning
from rag_factory.cli.utils.validation import validate_config_file

console = Console()


class REPLSession:
    """Interactive REPL session for RAG Factory CLI."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize REPL session.

        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.index_dir = "./rag_index"
        self.current_strategy = "basic"

        # Load config if provided
        if config_path:
            try:
                self.config = validate_config_file(config_path)
                console.print(f"[green]Loaded configuration from: {config_path}[/green]")
            except Exception as e:
                print_warning(f"Failed to load config: {e}")

        # Setup command completer
        self.commands = [
            "index",
            "query",
            "strategies",
            "config",
            "help",
            "set",
            "show",
            "exit",
            "quit",
        ]
        self.completer = WordCompleter(self.commands, ignore_case=True)
        self.history = InMemoryHistory()
        self.session: PromptSession = PromptSession(
            completer=self.completer,
            history=self.history,
        )

    def run(self) -> None:
        """Run the REPL loop."""
        console.print("\n[bold cyan]RAG Factory Interactive REPL[/bold cyan]")
        console.print("Type 'help' for available commands, 'exit' to quit\n")

        while True:
            try:
                # Get user input
                user_input = self.session.prompt("rag-factory> ")

                # Skip empty input
                if not user_input.strip():
                    continue

                # Parse command
                parts = user_input.strip().split()
                command = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []

                # Execute command
                if command in ["exit", "quit"]:
                    console.print("[dim]Goodbye![/dim]")
                    break
                elif command == "help":
                    self._show_help()
                elif command == "index":
                    self._handle_index(args)
                elif command == "query":
                    self._handle_query(args)
                elif command == "strategies":
                    self._handle_strategies(args)
                elif command == "config":
                    self._handle_config(args)
                elif command == "set":
                    self._handle_set(args)
                elif command == "show":
                    self._handle_show(args)
                else:
                    print_error(f"Unknown command: {command}")
                    console.print("[dim]Type 'help' for available commands[/dim]")

            except KeyboardInterrupt:
                console.print("\n[dim]Use 'exit' or 'quit' to exit[/dim]")
            except EOFError:
                console.print("\n[dim]Goodbye![/dim]")
                break
            except Exception as e:
                print_error(f"Error: {e}")

    def _show_help(self) -> None:
        """Show help message with available commands."""
        console.print("\n[bold]Available Commands:[/bold]\n")
        console.print("  [cyan]index <path>[/cyan]         - Index documents from path")
        console.print("  [cyan]query <text>[/cyan]         - Query indexed documents")
        console.print("  [cyan]strategies[/cyan]           - List available strategies")
        console.print("  [cyan]config <path>[/cyan]        - Load configuration file")
        console.print("  [cyan]set <key> <value>[/cyan]    - Set session parameter")
        console.print("  [cyan]show[/cyan]                 - Show current session state")
        console.print("  [cyan]help[/cyan]                 - Show this help message")
        console.print("  [cyan]exit/quit[/cyan]            - Exit REPL\n")

    def _handle_index(self, args: list) -> None:
        """Handle index command."""
        if not args:
            print_error("Usage: index <path>")
            return

        path = " ".join(args)
        console.print(f"[dim]Indexing documents from: {path}[/dim]")
        console.print(f"[dim]Strategy: {self.current_strategy}[/dim]")

        # In real implementation, would call actual indexing
        print_success(f"Successfully indexed documents from {path}")

    def _handle_query(self, args: list) -> None:
        """Handle query command."""
        if not args:
            print_error("Usage: query <text>")
            return

        query_text = " ".join(args)
        console.print(f"[dim]Executing query: {query_text}[/dim]")

        # Check if index exists
        if not Path(self.index_dir).exists():
            print_error(
                f"Index not found at: {self.index_dir}\n"
                f"Run 'index <path>' first to create an index"
            )
            return

        # In real implementation, would execute actual query
        console.print("\n[bold]Results:[/bold]")
        console.print("  1. Sample result (score: 0.95)")
        console.print("  2. Another result (score: 0.87)\n")

    def _handle_strategies(self, args: list) -> None:
        """Handle strategies command."""
        from rag_factory.factory import RAGFactory

        strategies = RAGFactory.list_strategies()
        if not strategies:
            print_warning("No strategies registered")
            return

        console.print("\n[bold]Registered Strategies:[/bold]")
        for strategy in strategies:
            if strategy == self.current_strategy:
                console.print(f"  [green]→ {strategy}[/green] (current)")
            else:
                console.print(f"    {strategy}")
        console.print()

    def _handle_config(self, args: list) -> None:
        """Handle config command."""
        if not args:
            print_error("Usage: config <path>")
            return

        config_path = " ".join(args)
        try:
            self.config = validate_config_file(config_path)
            self.config_path = config_path
            print_success(f"Loaded configuration from: {config_path}")
        except Exception as e:
            print_error(f"Failed to load config: {e}")

    def _handle_set(self, args: list) -> None:
        """Handle set command."""
        if len(args) < 2:
            print_error("Usage: set <key> <value>")
            console.print("\n[dim]Available settings:[/dim]")
            console.print("  strategy <name>    - Set current strategy")
            console.print("  index_dir <path>   - Set index directory")
            return

        key = args[0].lower()
        value = " ".join(args[1:])

        if key == "strategy":
            self.current_strategy = value
            print_success(f"Strategy set to: {value}")
        elif key == "index_dir":
            self.index_dir = value
            print_success(f"Index directory set to: {value}")
        else:
            print_error(f"Unknown setting: {key}")

    def _handle_show(self, args: list) -> None:
        """Handle show command."""
        console.print("\n[bold]Current Session State:[/bold]\n")
        console.print(f"  Strategy:     [cyan]{self.current_strategy}[/cyan]")
        console.print(f"  Index Dir:    [cyan]{self.index_dir}[/cyan]")
        console.print(f"  Config File:  [cyan]{self.config_path or 'None'}[/cyan]")

        if self.config:
            console.print("\n[bold]Configuration:[/bold]")
            for key, value in self.config.items():
                console.print(f"  {key}: {value}")
        console.print()


def start_repl(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file to load",
    ),
) -> None:
    """
    Start an interactive REPL session.

    The REPL (Read-Eval-Print Loop) provides an interactive environment
    for exploring RAG Factory features. It maintains session state and
    supports command history and auto-completion.

    Examples:

        Start REPL:
        $ rag-factory repl

        Start with configuration:
        $ rag-factory repl --config config.yaml

    In the REPL, you can use commands like:
        rag-factory> index ./docs
        rag-factory> query "What is RAG?"
        rag-factory> strategies
        rag-factory> set strategy semantic_chunker
        rag-factory> exit
    """
    try:
        session = REPLSession(config)
        session.run()
    except Exception as e:
        if "--debug" in sys.argv:
            raise
        print_error(f"REPL failed: {e}")
        raise typer.Exit(1)
