"""Query command for searching indexed documents."""

import sys
import time
from typing import Optional

import typer
from rich.console import Console

from rag_factory.cli.formatters import print_error, print_warning
from rag_factory.cli.formatters.results import format_query_results
from rag_factory.cli.utils import validate_config_file
from rag_factory.cli.utils.validation import parse_strategy_list

console = Console()


def query_command(
    query: str = typer.Argument(
        ...,
        help="Query string to search for in indexed documents",
    ),
    strategies: str = typer.Option(
        "basic",
        "--strategies",
        "-s",
        help="Comma-separated list of strategies to apply (e.g., 'reranking,query_expansion')",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of top results to return",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (YAML or JSON)",
    ),
    index_dir: str = typer.Option(
        "./rag_index",
        "--index",
        "-i",
        help="Directory containing the index",
    ),
    show_scores: bool = typer.Option(
        True,
        "--show-scores/--no-scores",
        help="Show relevance scores in results",
    ),
) -> None:
    """
    Query indexed documents using specified strategies.

    This command searches the indexed documents and returns relevant results.
    You can apply multiple strategies like reranking and query expansion.

    Examples:

        Simple query:
        $ rag-factory query "What are action items?"

        Query with multiple strategies:
        $ rag-factory query "database design" --strategies reranking,query_expansion

        Query with custom top-k:
        $ rag-factory query "machine learning" --top-k 10

        Query with configuration file:
        $ rag-factory query "neural networks" --config config.yaml
    """
    try:
        # Parse strategies
        strategy_list = parse_strategy_list(strategies)
        console.print(f"[cyan]Query:[/cyan] {query}")
        console.print(f"[cyan]Strategies:[/cyan] {', '.join(strategy_list)}")
        console.print(f"[cyan]Top-K:[/cyan] {top_k}")

        # Load configuration if provided
        if config:
            console.print(f"[dim]Loading configuration from: {config}[/dim]")
            strategy_config = validate_config_file(config)
        else:
            strategy_config = {}

        # Check if index exists
        from pathlib import Path
        index_path = Path(index_dir)
        if not index_path.exists():
            print_error(
                f"Index directory not found: {index_dir}\n\n"
                f"Please run 'rag-factory index' first to create an index."
            )
            raise typer.Exit(1)

        console.print(f"\n[bold]Executing query...[/bold]")

        # Execute query (mock implementation)
        start_time = time.time()

        # In real implementation, this would:
        # 1. Load the index
        # 2. Apply each strategy in sequence
        # 3. Return ranked results
        results = [
            {
                "text": f"This is a sample result {i+1} for query: {query}",
                "score": 0.95 - (i * 0.1),
                "metadata": {
                    "source": f"document_{i+1}.txt",
                    "chunk_id": f"chunk_{i+1}",
                },
            }
            for i in range(min(top_k, 5))
        ]

        elapsed_time = time.time() - start_time

        # Display results
        console.print("\n")
        results_table = format_query_results(results, query, strategy_list[0])
        console.print(results_table)

        # Show timing information
        console.print(f"\n[dim]Query executed in {elapsed_time * 1000:.2f}ms[/dim]")
        console.print(f"[dim]Returned {len(results)} result(s)[/dim]")

        # Show warnings if no results
        if not results:
            print_warning(
                "No results found for your query.\n"
                "Try:\n"
                "  - Using different keywords\n"
                "  - Broadening your search\n"
                "  - Checking if documents are properly indexed"
            )

    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)
    except KeyboardInterrupt:
        print_error("Query cancelled by user")
        raise typer.Exit(130)
    except Exception as e:
        if "--debug" in sys.argv:
            raise
        print_error(f"Query failed: {e}")
        raise typer.Exit(1)
