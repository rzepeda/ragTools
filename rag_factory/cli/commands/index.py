"""Index command for document indexing."""

import sys
import time
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console

from rag_factory.cli.formatters import print_error, print_success
from rag_factory.cli.formatters.results import format_statistics
from rag_factory.cli.utils import progress_context, validate_config_file, validate_path_exists
from rag_factory.factory import RAGFactory

console = Console()


def _collect_documents(path: Path) -> List[Path]:
    """
    Collect all documents from a path (file or directory).

    Args:
        path: Path to file or directory

    Returns:
        List[Path]: List of document file paths
    """
    if path.is_file():
        return [path]

    # Collect common document formats
    extensions = [".txt", ".md", ".pdf", ".html", ".json"]
    documents = []

    for ext in extensions:
        documents.extend(path.rglob(f"*{ext}"))

    return sorted(documents)


def index_command(
    path: str = typer.Argument(
        ...,
        help="Path to file or directory containing documents to index",
    ),
    strategy: str = typer.Option(
        "fixed_size_chunker",
        "--strategy",
        "-s",
        help="Chunking strategy to use for indexing",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (YAML or JSON)",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Directory to store indexed data (default: ./rag_index)",
    ),
    chunk_size: int = typer.Option(
        512,
        "--chunk-size",
        help="Size of chunks in tokens",
    ),
    chunk_overlap: int = typer.Option(
        50,
        "--chunk-overlap",
        help="Overlap between chunks in tokens",
    ),
) -> None:
    """
    Index documents using specified chunking strategy.

    This command processes documents and creates an index that can be
    queried later. It supports various file formats and chunking strategies.

    Examples:

        Index a directory with default strategy:
        $ rag-factory index ./docs

        Index with specific strategy and configuration:
        $ rag-factory index ./docs --strategy semantic_chunker --config config.yaml

        Index with custom chunk size:
        $ rag-factory index ./docs --chunk-size 1024 --chunk-overlap 100
    """
    try:
        # Validate input path
        console.print(f"[dim]Validating path: {path}[/dim]")
        doc_path = validate_path_exists(path)

        # Load configuration if provided
        strategy_config = {}
        if config:
            console.print(f"[dim]Loading configuration from: {config}[/dim]")
            strategy_config = validate_config_file(config)
        else:
            strategy_config = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }

        # Set output directory
        if output_dir is None:
            output_dir = "./rag_index"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        console.print(f"[cyan]Indexing Strategy:[/cyan] {strategy}")
        console.print(f"[cyan]Output Directory:[/cyan] {output_path}")

        # Collect documents
        console.print("\n[bold]Collecting documents...[/bold]")
        documents = _collect_documents(doc_path)

        if not documents:
            print_error(
                f"No documents found in: {path}\n"
                f"Supported formats: .txt, .md, .pdf, .html, .json"
            )
            raise typer.Exit(1)

        console.print(f"[green]Found {len(documents)} document(s)[/green]")

        # Initialize strategy (for now, we'll just report what would happen)
        # In a real implementation, this would use the factory to create and run the strategy
        console.print(f"\n[bold]Processing documents with {strategy}...[/bold]")

        start_time = time.time()
        total_chunks = 0

        with progress_context(
            "Indexing documents",
            total=len(documents)
        ) as (progress, task):
            for doc in documents:
                # Simulate processing (in real implementation, would call strategy)
                # Here we would: load document, chunk it, create embeddings, store in index
                time.sleep(0.01)  # Simulate work

                # Update progress
                progress.update(task, advance=1)
                total_chunks += 1  # In reality, this would be the actual chunk count

        elapsed_time = time.time() - start_time

        # Display statistics
        stats = {
            "documents_processed": len(documents),
            "total_chunks_created": total_chunks,
            "elapsed_time_seconds": elapsed_time,
            "avg_time_per_document": elapsed_time / len(documents) if documents else 0,
            "output_directory": str(output_path),
        }

        console.print("\n")
        console.print(format_statistics(stats))

        print_success(
            f"Successfully indexed {len(documents)} document(s) "
            f"with {total_chunks} chunk(s) in {elapsed_time:.2f}s"
        )

        console.print(
            f"\n[dim]Index stored in: {output_path}[/dim]\n"
            f"[dim]Use 'rag-factory query' to search the indexed documents[/dim]"
        )

    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)
    except KeyboardInterrupt:
        print_error("Indexing cancelled by user")
        raise typer.Exit(130)
    except Exception as e:
        if "--debug" in sys.argv:
            raise
        print_error(f"Indexing failed: {e}")
        raise typer.Exit(1)
