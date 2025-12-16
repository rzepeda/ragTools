"""General output formatting utilities."""

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def format_success(message: str, title: Optional[str] = None) -> Panel:
    """
    Format a success message in a green panel.

    Args:
        message: Success message text
        title: Optional panel title

    Returns:
        Panel: Rich panel with formatted success message
    """
    text = Text(message, style="green")
    return Panel(
        text,
        title=title or "Success",
        border_style="green",
        padding=(1, 2),
    )


def format_error(message: str, title: Optional[str] = None) -> Panel:
    """
    Format an error message in a red panel.

    Args:
        message: Error message text
        title: Optional panel title

    Returns:
        Panel: Rich panel with formatted error message
    """
    text = Text(message, style="red")
    return Panel(
        text,
        title=title or "Error",
        border_style="red",
        padding=(1, 2),
    )


def format_warning(message: str, title: Optional[str] = None) -> Panel:
    """
    Format a warning message in a yellow panel.

    Args:
        message: Warning message text
        title: Optional panel title

    Returns:
        Panel: Rich panel with formatted warning message
    """
    text = Text(message, style="yellow")
    return Panel(
        text,
        title=title or "Warning",
        border_style="yellow",
        padding=(1, 2),
    )


def print_success(message: str, title: Optional[str] = None) -> None:
    """Print a formatted success message."""
    console.print(format_success(message, title))


def print_error(message: str, title: Optional[str] = None) -> None:
    """Print a formatted error message."""
    console.print(format_error(message, title))


def print_warning(message: str, title: Optional[str] = None) -> None:
    """Print a formatted warning message."""
    console.print(format_warning(message, title))
