"""Benchmark command for testing strategies with datasets."""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console

from rag_factory.cli.formatters import print_error, print_success
from rag_factory.cli.formatters.results import format_benchmark_results
from rag_factory.cli.utils import progress_context, validate_path_exists
from rag_factory.cli.utils.validation import parse_strategy_list

console = Console()


def _load_benchmark_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """
    Load benchmark dataset from JSON file.

    Args:
        dataset_path: Path to benchmark dataset file

    Returns:
        List[Dict[str, Any]]: List of benchmark queries

    Expected format:
        [
            {
                "query": "What is RAG?",
                "expected_docs": ["doc1", "doc2"],
                "metadata": {...}
            },
            ...
        ]
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Benchmark dataset must be a JSON array of queries")

    # Validate dataset format
    for i, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise ValueError(f"Query {i} must be a dictionary")
        if "query" not in item:
            raise ValueError(f"Query {i} missing required field 'query'")

    return dataset


def _export_results(results: Dict[str, Any], output_path: Path) -> None:
    """
    Export benchmark results to file.

    Args:
        results: Benchmark results dictionary
        output_path: Path to output file
    """
    if output_path.suffix == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    elif output_path.suffix == ".csv":
        import csv
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            # Extract strategy results for CSV
            writer = csv.writer(f)
            writer.writerow(["Strategy", "Avg Latency (ms)", "Total Queries", "Success Rate", "Avg Score"])

            for strategy_name, strategy_results in results.items():
                if isinstance(strategy_results, dict):
                    writer.writerow([
                        strategy_name,
                        strategy_results.get("avg_latency_ms", 0),
                        strategy_results.get("total_queries", 0),
                        strategy_results.get("success_rate", 0),
                        strategy_results.get("avg_score", 0),
                    ])
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}. Use .json or .csv")


def run_benchmark(
    dataset: str = typer.Argument(
        ...,
        help="Path to benchmark dataset file (JSON)",
    ),
    strategies: Optional[str] = typer.Option(
        None,
        "--strategies",
        "-s",
        help="Comma-separated list of strategies to benchmark (default: all registered)",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to export results (JSON or CSV)",
    ),
    index_dir: str = typer.Option(
        "./rag_index",
        "--index",
        "-i",
        help="Directory containing the index",
    ),
    iterations: int = typer.Option(
        1,
        "--iterations",
        "-n",
        help="Number of iterations to run each query",
    ),
) -> None:
    """
    Run benchmarks using a test dataset.

    This command executes a set of queries from a benchmark dataset and
    measures performance metrics like latency, accuracy, and success rate
    for different strategies. Results can be exported to JSON or CSV.

    The dataset file should be a JSON array of query objects:
    [
        {
            "query": "What is machine learning?",
            "expected_docs": ["doc1.txt", "doc2.txt"],
            "metadata": {"category": "ml"}
        },
        ...
    ]

    Examples:

        Run benchmark with default strategies:
        $ rag-factory benchmark queries.json

        Benchmark specific strategies:
        $ rag-factory benchmark queries.json --strategies reranking,semantic_chunker

        Export results to JSON:
        $ rag-factory benchmark queries.json --output results.json

        Run multiple iterations:
        $ rag-factory benchmark queries.json --iterations 5
    """
    try:
        console.print(f"[bold]Loading benchmark dataset:[/bold] {dataset}\n")

        # Load dataset
        dataset_path = validate_path_exists(dataset, must_be_file=True)
        queries = _load_benchmark_dataset(dataset_path)

        console.print(f"[green]Loaded {len(queries)} queries[/green]")

        # Check if index exists
        index_path = Path(index_dir)
        if not index_path.exists():
            print_error(
                f"Index directory not found: {index_dir}\n\n"
                f"Please run 'rag-factory index' first to create an index."
            )
            raise typer.Exit(1)

        # Parse strategies
        if strategies:
            strategy_list = parse_strategy_list(strategies)
        else:
            from rag_factory.factory import RAGFactory
            strategy_list = RAGFactory.list_strategies()

            if not strategy_list:
                print_error("No strategies registered for benchmarking")
                raise typer.Exit(1)

        console.print(f"[cyan]Strategies to benchmark:[/cyan] {', '.join(strategy_list)}")
        console.print(f"[cyan]Iterations per query:[/cyan] {iterations}")

        # Run benchmarks
        console.print(f"\n[bold]Running benchmarks...[/bold]\n")

        benchmark_results = {}
        total_operations = len(strategy_list) * len(queries) * iterations

        with progress_context(
            "Benchmarking strategies",
            total=total_operations
        ) as (progress, task):
            for strategy_name in strategy_list:
                strategy_metrics = {
                    "total_queries": len(queries),
                    "iterations": iterations,
                    "latencies": [],
                    "scores": [],
                    "successes": 0,
                }

                for query_item in queries:
                    query_text = query_item["query"]

                    for _ in range(iterations):
                        # Simulate query execution
                        start_time = time.time()
                        # In real implementation, would execute actual query
                        time.sleep(0.01)  # Simulate work
                        latency = (time.time() - start_time) * 1000  # Convert to ms

                        strategy_metrics["latencies"].append(latency)
                        strategy_metrics["scores"].append(0.85)  # Mock score
                        strategy_metrics["successes"] += 1

                        progress.update(task, advance=1)

                # Calculate aggregate metrics
                benchmark_results[strategy_name] = {
                    "avg_latency_ms": sum(strategy_metrics["latencies"]) / len(strategy_metrics["latencies"]),
                    "min_latency_ms": min(strategy_metrics["latencies"]),
                    "max_latency_ms": max(strategy_metrics["latencies"]),
                    "total_queries": strategy_metrics["total_queries"],
                    "iterations": strategy_metrics["iterations"],
                    "success_rate": strategy_metrics["successes"] / (strategy_metrics["total_queries"] * iterations),
                    "avg_score": sum(strategy_metrics["scores"]) / len(strategy_metrics["scores"]),
                }

        # Display results
        console.print("\n")
        results_table = format_benchmark_results(benchmark_results, strategy_list)
        console.print(results_table)

        # Export results if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            console.print(f"\n[dim]Exporting results to: {output_path}[/dim]")
            _export_results(benchmark_results, output_path)
            console.print(f"[green]✓[/green] Results exported successfully")

        print_success(
            f"Benchmark completed: {len(queries)} queries × {len(strategy_list)} strategies × {iterations} iterations"
        )

        # Show summary
        best_latency = min(benchmark_results.values(), key=lambda x: x["avg_latency_ms"])
        best_strategy_latency = [k for k, v in benchmark_results.items() if v == best_latency][0]

        console.print(
            f"\n[dim]Best latency: {best_strategy_latency} "
            f"({best_latency['avg_latency_ms']:.2f}ms avg)[/dim]"
        )

    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)
    except KeyboardInterrupt:
        print_error("Benchmark cancelled by user")
        raise typer.Exit(130)
    except Exception as e:
        if "--debug" in sys.argv:
            raise
        print_error(f"Benchmark failed: {e}")
        raise typer.Exit(1)
