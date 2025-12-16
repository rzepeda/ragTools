"""Configuration validation command."""

import sys
from typing import Any, Dict, List

import typer
from rich.console import Console
from rich.syntax import Syntax

from rag_factory.cli.formatters import print_error, print_success, print_warning
from rag_factory.cli.utils import validate_config_file

console = Console()


def _validate_strategy_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate strategy-specific configuration fields.

    Args:
        config: Configuration dictionary to validate

    Returns:
        List[str]: List of validation errors (empty if valid)
    """
    errors = []

    # Check for required fields
    if "strategy_name" not in config:
        errors.append("Missing required field: 'strategy_name'")

    # Validate chunk_size if present
    if "chunk_size" in config:
        chunk_size = config["chunk_size"]
        if not isinstance(chunk_size, int):
            errors.append(f"'chunk_size' must be an integer, got {type(chunk_size).__name__}")
        elif chunk_size <= 0:
            errors.append(f"'chunk_size' must be positive, got {chunk_size}")

    # Validate chunk_overlap if present
    if "chunk_overlap" in config:
        chunk_overlap = config["chunk_overlap"]
        if not isinstance(chunk_overlap, int):
            errors.append(f"'chunk_overlap' must be an integer, got {type(chunk_overlap).__name__}")
        elif chunk_overlap < 0:
            errors.append(f"'chunk_overlap' must be non-negative, got {chunk_overlap}")

    # Validate top_k if present
    if "top_k" in config:
        top_k = config["top_k"]
        if not isinstance(top_k, int):
            errors.append(f"'top_k' must be an integer, got {type(top_k).__name__}")
        elif top_k <= 0:
            errors.append(f"'top_k' must be positive, got {top_k}")

    # Check for chunk_overlap >= chunk_size
    if "chunk_size" in config and "chunk_overlap" in config:
        if config["chunk_overlap"] >= config["chunk_size"]:
            errors.append(
                f"'chunk_overlap' ({config['chunk_overlap']}) must be less than "
                f"'chunk_size' ({config['chunk_size']})"
            )

    return errors


def validate_config(
    config_path: str = typer.Argument(
        ...,
        help="Path to configuration file to validate (YAML or JSON)",
    ),
    show_config: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Display the configuration file contents",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Enable strict validation (treat warnings as errors)",
    ),
) -> None:
    """
    Validate a configuration file for correctness.

    This command checks configuration files for syntax errors, missing
    required fields, invalid values, and other issues. It helps ensure
    your configuration is correct before using it with other commands.

    Examples:

        Validate a configuration file:
        $ rag-factory config config.yaml

        Validate and show contents:
        $ rag-factory config config.yaml --show

        Strict validation:
        $ rag-factory config config.yaml --strict
    """
    try:
        console.print(f"[bold]Validating configuration file:[/bold] {config_path}\n")

        # Load and validate file format
        config = validate_config_file(config_path)

        console.print("[green]✓[/green] File format is valid")

        # Show configuration if requested
        if show_config:
            import json
            syntax = Syntax(
                json.dumps(config, indent=2),
                "json",
                theme="monokai",
                line_numbers=True,
            )
            console.print("\n[bold]Configuration contents:[/bold]")
            console.print(syntax)
            console.print()

        # Validate configuration structure
        warnings = []
        errors = _validate_strategy_config(config)

        # Check for common issues (warnings)
        if "chunk_size" in config and config["chunk_size"] > 2048:
            warnings.append(
                f"Large chunk_size ({config['chunk_size']}) may impact performance. "
                "Consider using a smaller value (512-1024)."
            )

        if "top_k" in config and config["top_k"] > 100:
            warnings.append(
                f"Large top_k ({config['top_k']}) may impact performance. "
                "Consider using a smaller value (5-20)."
            )

        # Display validation results
        if errors:
            console.print("\n[red]✗ Validation failed with errors:[/red]\n")
            for i, error in enumerate(errors, 1):
                console.print(f"  {i}. [red]{error}[/red]")

            print_error(
                f"Found {len(errors)} error(s) in configuration file.\n"
                f"Please fix the errors and try again."
            )
            raise typer.Exit(1)

        if warnings:
            console.print("\n[yellow]⚠ Validation warnings:[/yellow]\n")
            for i, warning in enumerate(warnings, 1):
                console.print(f"  {i}. [yellow]{warning}[/yellow]")

            if strict:
                print_error(
                    f"Strict mode enabled: {len(warnings)} warning(s) treated as errors."
                )
                raise typer.Exit(1)
            else:
                print_warning(
                    f"Found {len(warnings)} warning(s). "
                    f"Use --strict to treat warnings as errors."
                )

        if not errors and not warnings:
            console.print("[green]✓[/green] No issues found")

        # Display summary
        console.print()
        print_success(
            "Configuration file is valid and ready to use."
        )

        console.print(
            f"\n[dim]Use this config with: rag-factory index --config {config_path}[/dim]"
        )

    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)
    except Exception as e:
        if "--debug" in sys.argv:
            raise
        print_error(f"Validation failed: {e}")
        raise typer.Exit(1)
