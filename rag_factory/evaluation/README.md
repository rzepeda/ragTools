# RAG Factory Evaluation Framework

A comprehensive evaluation framework for assessing and comparing Retrieval-Augmented Generation (RAG) strategies.

## Quick Start

```python
from rag_factory.evaluation import BenchmarkRunner
from rag_factory.evaluation.metrics import PrecisionAtK, RecallAtK
from rag_factory.evaluation.datasets import DatasetLoader

# Load evaluation dataset
loader = DatasetLoader()
dataset = loader.load("path/to/dataset.json")

# Configure metrics
metrics = [PrecisionAtK(k=5), RecallAtK(k=5)]

# Run benchmark
runner = BenchmarkRunner(metrics=metrics)
results = runner.run(my_strategy, dataset)

# View results
print(results.aggregate_metrics)
```

## Features

### 📊 Comprehensive Metrics

**Retrieval Metrics:**
- Precision@K, Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Hit Rate@K

**Quality Metrics:**
- Semantic Similarity
- Faithfulness (context grounding)
- Answer Relevance
- Answer Completeness

**Performance Metrics:**
- Latency (avg, P50, P95, P99)
- Throughput (queries/second)

**Cost Metrics:**
- Token usage
- API costs
- Cost efficiency

### 📁 Dataset Management

Load datasets from multiple formats:

```python
loader = DatasetLoader()

# JSON format
dataset = loader.load("data.json")

# JSONL format (one JSON object per line)
dataset = loader.load("data.jsonl")

# CSV format
dataset = loader.load("data.csv")
```

Dataset format:
```json
{
  "name": "my_dataset",
  "examples": [
    {
      "query_id": "q1",
      "query": "What is machine learning?",
      "ground_truth_answer": "ML is...",
      "relevant_doc_ids": ["doc1", "doc2"],
      "relevance_scores": {"doc1": 3, "doc2": 2}
    }
  ]
}
```

### 🔬 Statistical Analysis

```python
from rag_factory.evaluation.analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(confidence_level=0.95)

# Paired t-test
result = analyzer.paired_t_test(baseline_scores, new_scores)
print(f"p-value: {result.p_value}")
print(f"Significant: {result.significant}")
print(result.interpretation)

# Confidence intervals
ci_low, ci_high = analyzer.confidence_interval(scores)
print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
```

### 📈 Strategy Comparison

```python
from rag_factory.evaluation.analysis import StrategyComparator

# Compare multiple strategies
results = runner.compare_strategies(
    [strategy1, strategy2, strategy3],
    dataset,
    ["Baseline", "Enhanced", "Optimized"]
)

# Generate comparison report
comparator = StrategyComparator()
comparison = comparator.compare(results)
print(comparator.generate_report(comparison))
```

### 💾 Export Results

```python
from rag_factory.evaluation.exporters import (
    CSVExporter, JSONExporter, HTMLExporter
)

# Export to CSV
CSVExporter().export(results, "metrics.csv")

# Export to JSON
JSONExporter().export(results, "metrics.json")

# Export to HTML report
HTMLExporter().export(results, "report.html")

# Export comparison
HTMLExporter().export_comparison(
    [result1, result2],
    "comparison.html"
)
```

## Advanced Usage

### Custom Metrics

Create your own metrics by extending `IMetric`:

```python
from rag_factory.evaluation.metrics.base import IMetric, MetricResult, MetricType

class MyCustomMetric(IMetric):
    def __init__(self):
        super().__init__("my_metric", MetricType.RETRIEVAL)

    def compute(self, **kwargs) -> MetricResult:
        # Your computation logic
        value = compute_value(kwargs)

        return MetricResult(
            name=self.name,
            value=value,
            metadata={"details": "..."}
        )

    @property
    def description(self) -> str:
        return "Description of my custom metric"

# Use it
runner = BenchmarkRunner(metrics=[MyCustomMetric()])
```

### Checkpointing

For long-running evaluations, enable checkpointing:

```python
from rag_factory.evaluation.benchmarks import BenchmarkConfig

config = BenchmarkConfig(
    metrics=metrics,
    enable_checkpointing=True,
    checkpoint_interval=50,  # Save every 50 queries
    cache_dir=".benchmark_cache"
)

runner = BenchmarkRunner(config)
results = runner.run(strategy, dataset)

# Resume from checkpoint if interrupted
results = runner.run(
    strategy,
    dataset,
    resume_from=".benchmark_cache/checkpoints/checkpoint.json"
)
```

### Dataset Statistics

```python
from rag_factory.evaluation.datasets.statistics import DatasetStatistics

stats = DatasetStatistics(dataset)
summary = stats.compute()

print(stats.format_summary(summary))

# Check for quality issues
issues = stats.get_quality_issues()
for issue in issues:
    print(f"⚠️  {issue}")
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For quality metrics (semantic similarity)
pip install sentence-transformers
```

## Examples

See `examples/evaluation/` for complete examples:
- `sample_dataset.json`: Example evaluation dataset
- `evaluation_example.py`: End-to-end usage example

## Architecture

```
evaluation/
├── metrics/          # All evaluation metrics
│   ├── base.py       # Base metric interface
│   ├── retrieval.py  # Precision, Recall, MRR, NDCG, Hit Rate
│   ├── quality.py    # Semantic similarity, Faithfulness, etc.
│   ├── performance.py # Latency, Throughput
│   └── cost.py       # Token usage, API costs
├── datasets/         # Dataset management
│   ├── schema.py     # Data structures
│   ├── loader.py     # Multi-format loading
│   └── statistics.py # Dataset analysis
├── benchmarks/       # Benchmark execution
│   ├── config.py     # Configuration
│   └── runner.py     # Benchmark runner
├── analysis/         # Statistical analysis
│   ├── statistics.py # T-tests, effect sizes
│   └── comparison.py # Strategy comparison
└── exporters/        # Result export
    ├── csv_exporter.py
    ├── json_exporter.py
    └── html_exporter.py
```

## Best Practices

1. **Start with Standard Metrics**: Use Precision@K, Recall@K, and MRR for initial evaluation
2. **Use Multiple K Values**: Evaluate at different cutoffs (k=3, 5, 10)
3. **Statistical Testing**: Always compare with baseline using t-tests
4. **Export Results**: Save results for future reference and regression detection
5. **Dataset Quality**: Ensure high-quality relevance judgments in your dataset

## Troubleshooting

**Issue**: "sentence-transformers not installed"
- Solution: `pip install sentence-transformers` for quality metrics

**Issue**: "Dataset not found"
- Solution: Use absolute paths or check file location

**Issue**: "Empty dataset warning"
- Solution: Ensure dataset has at least one example

## Contributing

To add a new metric:

1. Create a class inheriting from `IMetric`
2. Implement `compute()` method
3. Implement `description` property
4. Add tests in `tests/unit/evaluation/`

## Documentation

- Full API documentation: See docstrings in each module
- Story documentation: `/docs/stories/epic-08/story-8.2-evaluation-framework.md`
- Completion summary: `/docs/stories/epic-08/story-8.2-COMPLETION-SUMMARY.md`

## License

Part of RAG Factory - see main project license.
