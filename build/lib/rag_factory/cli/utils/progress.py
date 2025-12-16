"""Progress bar utilities using Rich."""

from contextlib import contextmanager
from typing import Iterator, Optional

from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def create_progress_bar(
    description: str = "Processing",
    total: Optional[float] = None,
    show_time: bool = True,
) -> Progress:
    """
    Create a Rich progress bar with default styling.

    Args:
        description: Task description to display
        total: Total number of items (None for indeterminate progress)
        show_time: Whether to show elapsed and remaining time

    Returns:
        Progress: Configured Rich progress bar instance
    """
    columns: list[ProgressColumn] = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ]

    if show_time:
        columns.extend([
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ])

    return Progress(*columns)


@contextmanager
def progress_context(
    description: str = "Processing",
    total: Optional[float] = None,
    show_time: bool = True,
) -> Iterator[tuple[Progress, TaskID]]:
    """
    Context manager for progress bar operations.

    Args:
        description: Task description to display
        total: Total number of items (None for indeterminate progress)
        show_time: Whether to show elapsed and remaining time

    Yields:
        tuple[Progress, TaskID]: Progress bar instance and task ID

    Example:
        >>> with progress_context("Indexing documents", total=100) as (progress, task):
        ...     for i in range(100):
        ...         # Do work
        ...         progress.update(task, advance=1)
    """
    progress = create_progress_bar(description, total, show_time)
    with progress:
        task_id = progress.add_task(description, total=total)
        yield progress, task_id
