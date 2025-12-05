"""
Example usage of the RAG Factory Evaluation Framework.

This script demonstrates how to:
1. Load an evaluation dataset
2. Define metrics
3. Run benchmarks on RAG strategies
4. Analyze and compare results
5. Export results in various formats
"""

from pathlib import Path
from rag_factory.evaluation.datasets import DatasetLoader
from rag_factory.evaluation.metrics import (
    PrecisionAtK,
    RecallAtK,
    MeanReciprocalRank,
    NDCG,
    HitRateAtK,
)
from rag_factory.evaluation.benchmarks import BenchmarkRunner, BenchmarkConfig
from rag_factory.evaluation.analysis import StatisticalAnalyzer, StrategyComparator
from rag_factory.evaluation.exporters import CSVExporter, JSONExporter, HTMLExporter

# For demonstration, we'll create a mock RAG strategy
from rag_factory.strategies.base import IRAGStrategy, Chunk, StrategyConfig, PreparedData
from typing import List, Dict, Any
import random


class MockRAGStrategy(IRAGStrategy):
    """
    Mock RAG strategy for demonstration purposes.

    In real usage, you would use your actual RAG strategy implementation.
    """

    def __init__(self, name: str = "MockStrategy"):
        self.name = name
        self.config = None

    def initialize(self, config: StrategyConfig) -> None:
        self.config = config

    def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
        # Mock implementation
        return PreparedData(chunks=[], embeddings=[], index_metadata={})

    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        """
        Mock retrieval - returns random document IDs.
        In real usage, this would be your actual retrieval logic.
        """
        # Simulate retrieval with random documents
        available_docs = [f"doc{i}" for i in range(1, 15)]
        random.shuffle(available_docs)

        results = []
        for i, doc_id in enumerate(available_docs[:top_k]):
            chunk = Chunk(
                text=f"Content of {doc_id}",
                metadata={"source": doc_id},
                score=1.0 - (i * 0.1),  # Decreasing scores
                source_id=doc_id,
                chunk_id=f"{doc_id}_chunk_0"
            )
            results.append(chunk)

        return results

    async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
        return self.retrieve(query, top_k)

    def process_query(self, query: str, context: List[Chunk]) -> str:
        return f"Mock answer for: {query}"


def main():
    """Main evaluation workflow."""

    print("="*80)
    print("RAG Factory Evaluation Framework - Example Usage")
    print("="*80)
    print()

    # Step 1: Load evaluation dataset
    print("Step 1: Loading evaluation dataset...")
    loader = DatasetLoader()
    dataset_path = Path(__file__).parent / "sample_dataset.json"
    dataset = loader.load(dataset_path)
    print(f"  Loaded dataset: {dataset.name}")
    print(f"  Number of examples: {len(dataset)}")
    print()

    # Step 2: Define evaluation metrics
    print("Step 2: Defining evaluation metrics...")
    metrics = [
        PrecisionAtK(k=3),
        PrecisionAtK(k=5),
        RecallAtK(k=3),
        RecallAtK(k=5),
        MeanReciprocalRank(),
        NDCG(k=5),
        HitRateAtK(k=10),
    ]
    print(f"  Configured {len(metrics)} metrics:")
    for metric in metrics:
        print(f"    - {metric.name}: {metric.description}")
    print()

    # Step 3: Configure and run benchmark
    print("Step 3: Running benchmark...")
    config = BenchmarkConfig(
        metrics=metrics,
        top_k=10,
        verbose=True
    )

    runner = BenchmarkRunner(config)

    # Create mock strategy (in real usage, use your actual strategy)
    strategy = MockRAGStrategy("BaselineStrategy")

    # Run benchmark
    result = runner.run(strategy, dataset, strategy_name="Baseline")
    print()

    # Step 4: Display results
    print("Step 4: Benchmark Results")
    print("-"*80)
    print(f"Strategy: {result.strategy_name}")
    print(f"Dataset: {result.dataset_name}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    print()
    print("Aggregate Metrics:")
    for metric_name, value in sorted(result.aggregate_metrics.items()):
        print(f"  {metric_name:.<40} {value:.4f}")
    print()

    # Step 5: Compare multiple strategies (demo with 2 mock strategies)
    print("Step 5: Comparing multiple strategies...")
    strategy1 = MockRAGStrategy("Strategy_A")
    strategy2 = MockRAGStrategy("Strategy_B")

    results = runner.compare_strategies(
        [strategy1, strategy2],
        dataset,
        strategy_names=["Strategy_A", "Strategy_B"]
    )

    # Use comparator to analyze
    comparator = StrategyComparator()
    comparison = comparator.compare(results)

    print()
    print(comparator.generate_report(comparison))
    print()

    # Step 6: Statistical analysis
    print("Step 6: Statistical Analysis")
    print("-"*80)
    analyzer = StatisticalAnalyzer(confidence_level=0.95)

    # Get metric values for comparison
    metric_name = "precision@5"
    if metric_name in results[0].aggregate_metrics and metric_name in results[1].aggregate_metrics:
        # For demo, create mock distributions
        baseline_values = [results[0].aggregate_metrics[metric_name]] * 5
        comparison_values = [results[1].aggregate_metrics[metric_name]] * 5

        analysis = analyzer.analyze_metric(
            baseline_values,
            comparison_values,
            metric_name
        )

        print(f"Metric: {analysis['metric_name']}")
        print(f"  Baseline mean: {analysis['baseline']['mean']:.4f}")
        print(f"  Comparison mean: {analysis['comparison']['mean']:.4f}")
        print(f"  Improvement: {analysis['improvement_pct']:.2f}%")
        print(f"  Effect size: {analysis['effect_size_interpretation']}")
        print()

    # Step 7: Export results
    print("Step 7: Exporting results...")
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Export to JSON
    json_exporter = JSONExporter()
    json_exporter.export(result, str(output_dir / "benchmark_results.json"))
    print(f"  Exported to JSON: {output_dir / 'benchmark_results.json'}")

    # Export to CSV
    csv_exporter = CSVExporter()
    csv_exporter.export(result, str(output_dir / "benchmark_results.csv"))
    print(f"  Exported to CSV: {output_dir / 'benchmark_results.csv'}")

    # Export comparison to HTML
    html_exporter = HTMLExporter()
    html_exporter.export_comparison(
        results,
        str(output_dir / "comparison_report.html"),
        title="Strategy Comparison Report"
    )
    print(f"  Exported comparison to HTML: {output_dir / 'comparison_report.html'}")
    print()

    print("="*80)
    print("Evaluation complete! Check the output directory for exported results.")
    print("="*80)


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
