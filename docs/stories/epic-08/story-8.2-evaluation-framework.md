# Story 8.2: Create Evaluation Framework

**Story ID:** 8.2
**Epic:** Epic 8 - Observability & Quality Assurance
**Story Points:** 13
**Priority:** High
**Dependencies:** Epic 4 (RAG Strategies to evaluate), Story 8.1 (Logging for evaluation results)

---

## User Story

**As a** developer
**I want** to evaluate and compare RAG strategies
**So that** I can choose the best approach for my use case

---

## Detailed Requirements

### Functional Requirements

1. **Evaluation Metrics System**
   - **Retrieval Metrics**:
     - Precision@K: Percentage of relevant docs in top-K results
     - Recall@K: Percentage of relevant docs retrieved
     - Mean Reciprocal Rank (MRR): Position of first relevant result
     - Normalized Discounted Cumulative Gain (NDCG): Graded relevance ranking
     - Hit Rate@K: Percentage of queries with at least one relevant doc
   - **Quality Metrics**:
     - Answer Accuracy: Semantic similarity to ground truth
     - Answer Completeness: Coverage of key information
     - Faithfulness: Answer grounded in retrieved context
     - Relevance: How relevant the answer is to the query
   - **Performance Metrics**:
     - End-to-end Latency: Total response time
     - Retrieval Latency: Time to retrieve documents
     - Generation Latency: Time to generate answer
     - Throughput: Queries per second
   - **Cost Metrics**:
     - Token Usage: Input/output tokens consumed
     - API Costs: Total cost per query
     - Cost per Accurate Answer: Cost efficiency metric

2. **Test Dataset Management**
   - Load datasets from multiple formats (JSON, CSV, JSONL)
   - Dataset schema: query, ground_truth_answer, relevant_doc_ids, metadata
   - Support for multiple test datasets
   - Dataset versioning and tracking
   - Train/test/validation splits
   - Synthetic dataset generation utilities
   - Dataset statistics and analysis

3. **Benchmarking Suite**
   - Run evaluations on multiple strategies
   - Parallel strategy evaluation
   - Configurable evaluation runs
   - Support for A/B testing
   - Baseline strategy comparison
   - Progress tracking for long runs
   - Checkpoint/resume capability
   - Results caching

4. **Results Visualization**
   - Web-based results dashboard
   - Comparison tables for multiple strategies
   - Performance charts (latency, accuracy)
   - Cost analysis visualizations
   - Metric heatmaps
   - Query-level result inspection
   - Error analysis views
   - Interactive filtering and sorting

5. **Results Export**
   - Export to CSV format
   - Export to JSON format
   - Export to HTML reports
   - Export to PDF reports (optional)
   - Configurable export fields
   - Summary statistics in exports

6. **Statistical Significance Testing**
   - Paired t-tests for metric comparison
   - Bootstrap confidence intervals
   - Effect size calculation (Cohen's d)
   - P-value computation
   - Multiple comparison correction (Bonferroni)
   - Significance indicators in results

7. **Advanced Features**
   - Custom metric definitions
   - Metric composition (weighted averages)
   - Metric aggregation at different levels
   - Time-based metric tracking
   - Regression testing (compare against previous runs)
   - Alert thresholds for metric degradation

### Non-Functional Requirements

1. **Performance**
   - Evaluate 100 queries in <5 minutes (per strategy)
   - Support concurrent strategy evaluation
   - Efficient metric calculation
   - Minimal memory overhead

2. **Accuracy**
   - Metric calculations verified against known benchmarks
   - Consistent results across runs
   - Proper handling of edge cases

3. **Usability**
   - Clear configuration API
   - Intuitive command-line interface
   - Comprehensive documentation
   - Example datasets and notebooks
   - Error messages with actionable guidance

4. **Extensibility**
   - Easy to add new metrics
   - Pluggable evaluators
   - Custom visualization components
   - Integration with existing tools (MLflow, Weights & Biases)

5. **Reproducibility**
   - Deterministic evaluation
   - Seed control for randomness
   - Full configuration logging
   - Version tracking for datasets and strategies

---

## Acceptance Criteria

### AC1: Metrics System
- [ ] All retrieval metrics implemented (Precision@K, Recall@K, MRR, NDCG, Hit Rate)
- [ ] Quality metrics implemented (Accuracy, Completeness, Faithfulness, Relevance)
- [ ] Performance metrics tracked (Latency, Throughput)
- [ ] Cost metrics calculated (Token usage, API costs)
- [ ] Metrics validated against known test cases

### AC2: Dataset Management
- [ ] Load datasets from JSON, CSV, JSONL
- [ ] Dataset schema validation
- [ ] Support multiple datasets
- [ ] Dataset statistics available
- [ ] Synthetic dataset generator working

### AC3: Benchmarking
- [ ] Evaluate multiple strategies
- [ ] Parallel evaluation supported
- [ ] Progress tracking implemented
- [ ] Results caching working
- [ ] Checkpoint/resume functional

### AC4: Visualization
- [ ] Web dashboard accessible
- [ ] Comparison tables rendered
- [ ] Performance charts displayed
- [ ] Query-level inspection working
- [ ] Interactive filtering functional

### AC5: Export Capabilities
- [ ] Export to CSV working
- [ ] Export to JSON working
- [ ] Export to HTML reports
- [ ] Summary statistics included
- [ ] Configurable export fields

### AC6: Statistical Testing
- [ ] T-tests implemented
- [ ] Confidence intervals calculated
- [ ] Effect sizes computed
- [ ] P-values displayed
- [ ] Significance indicators shown

### AC7: Testing & Quality
- [ ] Unit tests for all metrics (>90% coverage)
- [ ] Integration tests with real strategies
- [ ] Benchmark performance tests pass
- [ ] Example datasets included
- [ ] Documentation complete

---

## Technical Specifications

### File Structure
```
rag_factory/
├── evaluation/
│   ├── __init__.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── base.py                  # Base metric interface
│   │   ├── retrieval.py             # Retrieval metrics
│   │   ├── quality.py               # Answer quality metrics
│   │   ├── performance.py           # Performance metrics
│   │   ├── cost.py                  # Cost metrics
│   │   └── registry.py              # Metric registry
│   │
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── loader.py                # Dataset loading
│   │   ├── schema.py                # Dataset schema
│   │   ├── splitter.py              # Train/test splits
│   │   ├── generator.py             # Synthetic data generation
│   │   └── statistics.py            # Dataset statistics
│   │
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── runner.py                # Benchmark runner
│   │   ├── config.py                # Benchmark configuration
│   │   ├── parallel.py              # Parallel execution
│   │   ├── checkpoint.py            # Checkpointing
│   │   └── cache.py                 # Results caching
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── statistics.py            # Statistical tests
│   │   ├── comparison.py            # Strategy comparison
│   │   ├── aggregation.py           # Metric aggregation
│   │   └── regression.py            # Regression detection
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── dashboard.py             # Web dashboard
│   │   ├── plots.py                 # Plotting utilities
│   │   ├── tables.py                # Table generation
│   │   └── templates/               # HTML templates
│   │       ├── dashboard.html
│   │       ├── comparison.html
│   │       └── report.html
│   │
│   └── exporters/
│       ├── __init__.py
│       ├── csv_exporter.py          # CSV export
│       ├── json_exporter.py         # JSON export
│       ├── html_exporter.py         # HTML export
│       └── pdf_exporter.py          # PDF export (optional)
│
tests/
├── unit/
│   └── evaluation/
│       ├── test_retrieval_metrics.py
│       ├── test_quality_metrics.py
│       ├── test_dataset_loader.py
│       ├── test_benchmark_runner.py
│       ├── test_statistics.py
│       └── test_exporters.py
│
├── integration/
│   └── evaluation/
│       ├── test_evaluation_integration.py
│       └── test_benchmark_integration.py
│
examples/
└── evaluation/
    ├── sample_dataset.json
    ├── evaluation_example.py
    └── benchmark_notebook.ipynb
```

### Dependencies
```python
# requirements.txt additions
pandas==2.1.4                  # Data manipulation
numpy==1.24.0                  # Numerical operations
scipy==1.11.0                  # Statistical functions
matplotlib==3.8.2              # Visualization
plotly==5.18.0                 # Interactive plots
seaborn==0.13.0                # Statistical plots
scikit-learn==1.3.2            # ML metrics and utilities
jinja2==3.1.2                  # Template rendering
tabulate==0.9.0                # Table formatting
tqdm==4.66.1                   # Progress bars

# Optional
sentence-transformers==2.3.1   # For semantic similarity
rouge-score==0.1.2             # For text evaluation
bert-score==0.3.13             # For semantic evaluation
```

### Base Metric Interface
```python
# rag_factory/evaluation/metrics/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class MetricType(Enum):
    """Types of evaluation metrics."""
    RETRIEVAL = "retrieval"
    QUALITY = "quality"
    PERFORMANCE = "performance"
    COST = "cost"

@dataclass
class MetricResult:
    """Result from a metric computation."""
    name: str
    value: float
    metadata: Dict[str, Any]
    query_id: Optional[str] = None

class IMetric(ABC):
    """
    Abstract base class for evaluation metrics.

    All metrics should inherit from this class and implement
    the compute method.
    """

    def __init__(self, name: str, metric_type: MetricType):
        self.name = name
        self.metric_type = metric_type

    @abstractmethod
    def compute(self, **kwargs) -> MetricResult:
        """
        Compute the metric value.

        Args:
            **kwargs: Metric-specific inputs

        Returns:
            MetricResult with computed value and metadata
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of the metric."""
        pass

    @property
    def higher_is_better(self) -> bool:
        """Whether higher values are better (default: True)."""
        return True
```

### Retrieval Metrics
```python
# rag_factory/evaluation/metrics/retrieval.py
from typing import List, Set
import numpy as np
from .base import IMetric, MetricResult, MetricType

class PrecisionAtK(IMetric):
    """
    Precision@K: Proportion of retrieved documents that are relevant.

    Formula: Precision@K = (# relevant docs in top K) / K
    """

    def __init__(self, k: int = 5):
        super().__init__(f"precision@{k}", MetricType.RETRIEVAL)
        self.k = k

    def compute(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        query_id: Optional[str] = None
    ) -> MetricResult:
        """
        Compute Precision@K.

        Args:
            retrieved_ids: List of retrieved document IDs (ordered by rank)
            relevant_ids: Set of relevant document IDs
            query_id: Optional query identifier

        Returns:
            MetricResult with precision value
        """
        top_k = retrieved_ids[:self.k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        precision = relevant_in_top_k / self.k if self.k > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=precision,
            metadata={
                "k": self.k,
                "relevant_in_top_k": relevant_in_top_k,
                "total_relevant": len(relevant_ids)
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return f"Proportion of top-{self.k} retrieved documents that are relevant"

class RecallAtK(IMetric):
    """
    Recall@K: Proportion of relevant documents retrieved in top K.

    Formula: Recall@K = (# relevant docs in top K) / (# total relevant docs)
    """

    def __init__(self, k: int = 5):
        super().__init__(f"recall@{k}", MetricType.RETRIEVAL)
        self.k = k

    def compute(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        query_id: Optional[str] = None
    ) -> MetricResult:
        """Compute Recall@K."""
        if not relevant_ids:
            return MetricResult(
                name=self.name,
                value=0.0,
                metadata={"k": self.k, "total_relevant": 0},
                query_id=query_id
            )

        top_k = retrieved_ids[:self.k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        recall = relevant_in_top_k / len(relevant_ids)

        return MetricResult(
            name=self.name,
            value=recall,
            metadata={
                "k": self.k,
                "relevant_in_top_k": relevant_in_top_k,
                "total_relevant": len(relevant_ids)
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return f"Proportion of relevant documents retrieved in top-{self.k}"

class MeanReciprocalRank(IMetric):
    """
    Mean Reciprocal Rank (MRR): Position of first relevant document.

    Formula: MRR = 1 / rank of first relevant document
    """

    def __init__(self):
        super().__init__("mrr", MetricType.RETRIEVAL)

    def compute(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        query_id: Optional[str] = None
    ) -> MetricResult:
        """Compute MRR."""
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                mrr = 1.0 / rank
                return MetricResult(
                    name=self.name,
                    value=mrr,
                    metadata={"first_relevant_rank": rank},
                    query_id=query_id
                )

        # No relevant document found
        return MetricResult(
            name=self.name,
            value=0.0,
            metadata={"first_relevant_rank": None},
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return "Reciprocal rank of first relevant document"

class NDCG(IMetric):
    """
    Normalized Discounted Cumulative Gain (NDCG).

    Considers graded relevance and position discount.
    Formula: NDCG@K = DCG@K / IDCG@K
    """

    def __init__(self, k: int = 10):
        super().__init__(f"ndcg@{k}", MetricType.RETRIEVAL)
        self.k = k

    def compute(
        self,
        retrieved_ids: List[str],
        relevance_scores: Dict[str, float],  # doc_id -> relevance (0-1 or 0-3)
        query_id: Optional[str] = None
    ) -> MetricResult:
        """
        Compute NDCG@K.

        Args:
            retrieved_ids: List of retrieved document IDs (ordered)
            relevance_scores: Dictionary mapping doc IDs to relevance scores
            query_id: Optional query identifier

        Returns:
            MetricResult with NDCG value
        """
        def dcg(scores: List[float]) -> float:
            """Calculate DCG."""
            return sum(
                (2**score - 1) / np.log2(rank + 2)
                for rank, score in enumerate(scores)
            )

        # Get relevance scores for retrieved docs
        retrieved_scores = [
            relevance_scores.get(doc_id, 0.0)
            for doc_id in retrieved_ids[:self.k]
        ]

        # Calculate DCG
        dcg_value = dcg(retrieved_scores)

        # Calculate IDCG (ideal DCG with perfect ranking)
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:self.k]
        idcg_value = dcg(ideal_scores)

        # Calculate NDCG
        ndcg_value = dcg_value / idcg_value if idcg_value > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=ndcg_value,
            metadata={
                "k": self.k,
                "dcg": dcg_value,
                "idcg": idcg_value
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return f"Normalized Discounted Cumulative Gain at top-{self.k}"
```

### Quality Metrics
```python
# rag_factory/evaluation/metrics/quality.py
from typing import Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .base import IMetric, MetricResult, MetricType

class SemanticSimilarity(IMetric):
    """
    Semantic similarity between generated answer and ground truth.

    Uses sentence embeddings to compute cosine similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__("semantic_similarity", MetricType.QUALITY)
        self.model = SentenceTransformer(model_name)

    def compute(
        self,
        generated_answer: str,
        ground_truth: str,
        query_id: Optional[str] = None
    ) -> MetricResult:
        """
        Compute semantic similarity.

        Args:
            generated_answer: Generated answer text
            ground_truth: Ground truth answer text
            query_id: Optional query identifier

        Returns:
            MetricResult with similarity score (0-1)
        """
        # Generate embeddings
        embeddings = self.model.encode([generated_answer, ground_truth])

        # Compute cosine similarity
        similarity = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]

        return MetricResult(
            name=self.name,
            value=float(similarity),
            metadata={
                "model": self.model.get_sentence_embedding_dimension()
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return "Semantic similarity between generated and ground truth answers"

class Faithfulness(IMetric):
    """
    Faithfulness: Whether answer is grounded in retrieved context.

    Measures if claims in answer can be verified from context.
    """

    def __init__(self):
        super().__init__("faithfulness", MetricType.QUALITY)

    def compute(
        self,
        answer: str,
        context: List[str],
        query_id: Optional[str] = None
    ) -> MetricResult:
        """
        Compute faithfulness score.

        Args:
            answer: Generated answer
            context: Retrieved context documents
            query_id: Optional query identifier

        Returns:
            MetricResult with faithfulness score
        """
        # Simplified implementation
        # In production, use NLI model or LLM-based verification

        # Check if answer phrases appear in context
        answer_lower = answer.lower()
        context_text = " ".join(context).lower()

        # Split answer into sentences
        sentences = [s.strip() for s in answer.split('.') if s.strip()]

        grounded_count = 0
        for sentence in sentences:
            # Check if sentence words appear in context
            words = sentence.split()
            word_coverage = sum(1 for word in words if word in context_text)
            if word_coverage / len(words) > 0.5:  # 50% word overlap
                grounded_count += 1

        faithfulness_score = grounded_count / len(sentences) if sentences else 0.0

        return MetricResult(
            name=self.name,
            value=faithfulness_score,
            metadata={
                "total_sentences": len(sentences),
                "grounded_sentences": grounded_count
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return "Degree to which answer is grounded in retrieved context"
```

### Dataset Management
```python
# rag_factory/evaluation/datasets/schema.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set

@dataclass
class EvaluationExample:
    """Single evaluation example."""
    query_id: str
    query: str
    ground_truth_answer: Optional[str] = None
    relevant_doc_ids: Set[str] = field(default_factory=set)
    relevance_scores: Dict[str, float] = field(default_factory=dict)  # doc_id -> score
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationDataset:
    """Collection of evaluation examples."""
    name: str
    examples: List[EvaluationExample]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> EvaluationExample:
        return self.examples[idx]

# rag_factory/evaluation/datasets/loader.py
import json
import csv
from pathlib import Path
from typing import Union
from .schema import EvaluationDataset, EvaluationExample

class DatasetLoader:
    """
    Load evaluation datasets from various formats.

    Supports: JSON, JSONL, CSV

    Example:
        loader = DatasetLoader()
        dataset = loader.load("path/to/dataset.json")
    """

    def load(self, path: Union[str, Path]) -> EvaluationDataset:
        """Load dataset from file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        if path.suffix == '.json':
            return self._load_json(path)
        elif path.suffix == '.jsonl':
            return self._load_jsonl(path)
        elif path.suffix == '.csv':
            return self._load_csv(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    def _load_json(self, path: Path) -> EvaluationDataset:
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)

        examples = [
            EvaluationExample(
                query_id=ex.get("query_id", f"q_{i}"),
                query=ex["query"],
                ground_truth_answer=ex.get("ground_truth_answer"),
                relevant_doc_ids=set(ex.get("relevant_doc_ids", [])),
                relevance_scores=ex.get("relevance_scores", {}),
                metadata=ex.get("metadata", {})
            )
            for i, ex in enumerate(data.get("examples", []))
        ]

        return EvaluationDataset(
            name=data.get("name", path.stem),
            examples=examples,
            metadata=data.get("metadata", {})
        )

    def _load_jsonl(self, path: Path) -> EvaluationDataset:
        """Load from JSONL file."""
        examples = []

        with open(path) as f:
            for i, line in enumerate(f):
                ex = json.loads(line)
                examples.append(
                    EvaluationExample(
                        query_id=ex.get("query_id", f"q_{i}"),
                        query=ex["query"],
                        ground_truth_answer=ex.get("ground_truth_answer"),
                        relevant_doc_ids=set(ex.get("relevant_doc_ids", [])),
                        relevance_scores=ex.get("relevance_scores", {}),
                        metadata=ex.get("metadata", {})
                    )
                )

        return EvaluationDataset(
            name=path.stem,
            examples=examples
        )

    def _load_csv(self, path: Path) -> EvaluationDataset:
        """Load from CSV file."""
        examples = []

        with open(path) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                relevant_ids = row.get("relevant_doc_ids", "").split(",")
                relevant_ids = set(id.strip() for id in relevant_ids if id.strip())

                examples.append(
                    EvaluationExample(
                        query_id=row.get("query_id", f"q_{i}"),
                        query=row["query"],
                        ground_truth_answer=row.get("ground_truth_answer"),
                        relevant_doc_ids=relevant_ids
                    )
                )

        return EvaluationDataset(
            name=path.stem,
            examples=examples
        )
```

### Benchmark Runner
```python
# rag_factory/evaluation/benchmarks/runner.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time
from tqdm import tqdm
from ..datasets.schema import EvaluationDataset, EvaluationExample
from ..metrics.base import IMetric, MetricResult
from ...strategies.base import IRAGStrategy

@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    strategy_name: str
    dataset_name: str
    query_results: List[Dict[str, Any]]
    aggregate_metrics: Dict[str, float]
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class BenchmarkRunner:
    """
    Run benchmarks on RAG strategies.

    Features:
    - Evaluate multiple strategies on datasets
    - Track progress
    - Compute metrics
    - Cache results

    Example:
        runner = BenchmarkRunner(metrics=[precision, recall])
        results = runner.run(strategy, dataset)
    """

    def __init__(self, metrics: List[IMetric]):
        self.metrics = metrics

    def run(
        self,
        strategy: IRAGStrategy,
        dataset: EvaluationDataset,
        strategy_name: Optional[str] = None
    ) -> BenchmarkResult:
        """
        Run benchmark on a strategy.

        Args:
            strategy: RAG strategy to evaluate
            dataset: Evaluation dataset
            strategy_name: Optional name for strategy

        Returns:
            BenchmarkResult with all metrics
        """
        strategy_name = strategy_name or strategy.__class__.__name__

        start_time = time.time()
        query_results = []

        # Iterate through dataset with progress bar
        for example in tqdm(dataset.examples, desc=f"Evaluating {strategy_name}"):
            query_result = self._evaluate_query(strategy, example)
            query_results.append(query_result)

        # Aggregate metrics
        aggregate_metrics = self._aggregate_metrics(query_results)

        execution_time = time.time() - start_time

        return BenchmarkResult(
            strategy_name=strategy_name,
            dataset_name=dataset.name,
            query_results=query_results,
            aggregate_metrics=aggregate_metrics,
            execution_time=execution_time
        )

    def _evaluate_query(
        self,
        strategy: IRAGStrategy,
        example: EvaluationExample
    ) -> Dict[str, Any]:
        """Evaluate a single query."""
        # Execute strategy
        start_time = time.time()
        retrieved_docs = strategy.retrieve(example.query, top_k=10)
        latency = (time.time() - start_time) * 1000  # ms

        # Compute all metrics
        metric_results = {}
        for metric in self.metrics:
            try:
                result = metric.compute(
                    retrieved_ids=[doc.source_id for doc in retrieved_docs],
                    relevant_ids=example.relevant_doc_ids,
                    query_id=example.query_id
                )
                metric_results[result.name] = result.value
            except Exception as e:
                metric_results[metric.name] = None

        return {
            "query_id": example.query_id,
            "query": example.query,
            "latency_ms": latency,
            "results_count": len(retrieved_docs),
            "metrics": metric_results
        }

    def _aggregate_metrics(
        self,
        query_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate metrics across all queries."""
        aggregated = {}

        # Get all metric names
        metric_names = set()
        for result in query_results:
            metric_names.update(result["metrics"].keys())

        # Compute averages
        for metric_name in metric_names:
            values = [
                result["metrics"][metric_name]
                for result in query_results
                if result["metrics"].get(metric_name) is not None
            ]
            if values:
                aggregated[metric_name] = sum(values) / len(values)

        # Add performance metrics
        latencies = [r["latency_ms"] for r in query_results]
        aggregated["avg_latency_ms"] = sum(latencies) / len(latencies)
        aggregated["p95_latency_ms"] = sorted(latencies)[int(len(latencies) * 0.95)]

        return aggregated
```

---

## Unit Tests

### Test File Location
`tests/unit/evaluation/test_retrieval_metrics.py`
`tests/unit/evaluation/test_quality_metrics.py`
`tests/unit/evaluation/test_dataset_loader.py`

### Test Cases

#### TC8.2.1: Retrieval Metrics Tests
```python
import pytest
from rag_factory.evaluation.metrics.retrieval import (
    PrecisionAtK, RecallAtK, MeanReciprocalRank, NDCG
)

def test_precision_at_k():
    """Test Precision@K calculation."""
    metric = PrecisionAtK(k=5)

    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc5"}

    result = metric.compute(retrieved, relevant)

    # 3 out of 5 are relevant
    assert result.value == 0.6
    assert result.metadata["relevant_in_top_k"] == 3

def test_recall_at_k():
    """Test Recall@K calculation."""
    metric = RecallAtK(k=5)

    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc6", "doc7"}  # 4 total relevant

    result = metric.compute(retrieved, relevant)

    # 2 out of 4 relevant docs retrieved
    assert result.value == 0.5
    assert result.metadata["relevant_in_top_k"] == 2

def test_mrr():
    """Test Mean Reciprocal Rank."""
    metric = MeanReciprocalRank()

    # First relevant at position 3
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    relevant = {"doc3"}

    result = metric.compute(retrieved, relevant)

    assert result.value == pytest.approx(1/3)
    assert result.metadata["first_relevant_rank"] == 3

def test_mrr_no_relevant():
    """Test MRR when no relevant docs retrieved."""
    metric = MeanReciprocalRank()

    retrieved = ["doc1", "doc2"]
    relevant = {"doc3"}

    result = metric.compute(retrieved, relevant)

    assert result.value == 0.0
    assert result.metadata["first_relevant_rank"] is None

def test_ndcg():
    """Test NDCG calculation."""
    metric = NDCG(k=5)

    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevance_scores = {
        "doc1": 3,  # Highly relevant
        "doc2": 0,  # Not relevant
        "doc3": 2,  # Somewhat relevant
        "doc4": 1,  # Slightly relevant
        "doc5": 0
    }

    result = metric.compute(retrieved, relevance_scores)

    # NDCG should be less than 1 since order is not optimal
    assert 0 <= result.value <= 1
    assert "dcg" in result.metadata
    assert "idcg" in result.metadata

def test_perfect_ndcg():
    """Test NDCG with perfect ranking."""
    metric = NDCG(k=3)

    # Perfect order: most relevant first
    retrieved = ["doc1", "doc2", "doc3"]
    relevance_scores = {
        "doc1": 3,
        "doc2": 2,
        "doc3": 1
    }

    result = metric.compute(retrieved, relevance_scores)

    # Perfect ranking should give NDCG = 1.0
    assert result.value == pytest.approx(1.0, abs=0.01)
```

#### TC8.2.2: Dataset Loader Tests
```python
import pytest
import json
import csv
from pathlib import Path
from rag_factory.evaluation.datasets.loader import DatasetLoader
from rag_factory.evaluation.datasets.schema import EvaluationDataset

@pytest.fixture
def temp_json_dataset(tmp_path):
    """Create a temporary JSON dataset."""
    data = {
        "name": "test_dataset",
        "examples": [
            {
                "query_id": "q1",
                "query": "What is machine learning?",
                "ground_truth_answer": "ML is...",
                "relevant_doc_ids": ["doc1", "doc2"]
            },
            {
                "query_id": "q2",
                "query": "What is AI?",
                "relevant_doc_ids": ["doc3"]
            }
        ]
    }

    path = tmp_path / "dataset.json"
    with open(path, 'w') as f:
        json.dump(data, f)

    return path

def test_load_json_dataset(temp_json_dataset):
    """Test loading JSON dataset."""
    loader = DatasetLoader()
    dataset = loader.load(temp_json_dataset)

    assert isinstance(dataset, EvaluationDataset)
    assert dataset.name == "test_dataset"
    assert len(dataset) == 2

    # Check first example
    ex1 = dataset[0]
    assert ex1.query_id == "q1"
    assert ex1.query == "What is machine learning?"
    assert "doc1" in ex1.relevant_doc_ids
    assert "doc2" in ex1.relevant_doc_ids

def test_load_nonexistent_file():
    """Test loading non-existent file raises error."""
    loader = DatasetLoader()

    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent.json")

def test_load_unsupported_format(tmp_path):
    """Test loading unsupported format raises error."""
    path = tmp_path / "data.txt"
    path.write_text("test")

    loader = DatasetLoader()

    with pytest.raises(ValueError, match="Unsupported format"):
        loader.load(path)
```

#### TC8.2.3: Benchmark Runner Tests
```python
import pytest
from unittest.mock import Mock
from rag_factory.evaluation.benchmarks.runner import BenchmarkRunner
from rag_factory.evaluation.metrics.retrieval import PrecisionAtK
from rag_factory.evaluation.datasets.schema import EvaluationDataset, EvaluationExample
from rag_factory.strategies.base import Chunk

@pytest.fixture
def mock_strategy():
    """Create a mock RAG strategy."""
    strategy = Mock()
    strategy.retrieve.return_value = [
        Chunk(text="doc1", metadata={}, score=0.9, source_id="doc1", chunk_id="c1"),
        Chunk(text="doc2", metadata={}, score=0.8, source_id="doc2", chunk_id="c2")
    ]
    return strategy

@pytest.fixture
def simple_dataset():
    """Create a simple evaluation dataset."""
    examples = [
        EvaluationExample(
            query_id="q1",
            query="test query 1",
            relevant_doc_ids={"doc1", "doc3"}
        ),
        EvaluationExample(
            query_id="q2",
            query="test query 2",
            relevant_doc_ids={"doc2"}
        )
    ]
    return EvaluationDataset(name="test_dataset", examples=examples)

def test_benchmark_runner_initialization():
    """Test benchmark runner initialization."""
    metrics = [PrecisionAtK(k=5)]
    runner = BenchmarkRunner(metrics)

    assert len(runner.metrics) == 1

def test_run_benchmark(mock_strategy, simple_dataset):
    """Test running a benchmark."""
    metrics = [PrecisionAtK(k=5)]
    runner = BenchmarkRunner(metrics)

    result = runner.run(mock_strategy, simple_dataset, "TestStrategy")

    assert result.strategy_name == "TestStrategy"
    assert result.dataset_name == "test_dataset"
    assert len(result.query_results) == 2
    assert "precision@5" in result.aggregate_metrics
    assert result.execution_time > 0

def test_aggregate_metrics(mock_strategy, simple_dataset):
    """Test metric aggregation."""
    metrics = [PrecisionAtK(k=5)]
    runner = BenchmarkRunner(metrics)

    result = runner.run(mock_strategy, simple_dataset)

    # Should have aggregated metrics
    assert "precision@5" in result.aggregate_metrics
    assert "avg_latency_ms" in result.aggregate_metrics
    assert isinstance(result.aggregate_metrics["precision@5"], float)
```

---

## Integration Tests

### Test File Location
`tests/integration/evaluation/test_evaluation_integration.py`

### Test Scenarios

#### IS8.2.1: End-to-End Evaluation
```python
import pytest
from rag_factory.evaluation.benchmarks.runner import BenchmarkRunner
from rag_factory.evaluation.metrics.retrieval import PrecisionAtK, RecallAtK, MeanReciprocalRank
from rag_factory.evaluation.datasets.loader import DatasetLoader
from rag_factory.strategies.base import IRAGStrategy, Chunk

class DummyStrategy(IRAGStrategy):
    """Dummy strategy for testing."""

    def initialize(self, config):
        pass

    def prepare_data(self, documents):
        pass

    def retrieve(self, query: str, top_k: int):
        # Return dummy results
        return [
            Chunk(f"doc{i}", {}, 0.9-i*0.1, f"doc{i}", f"c{i}")
            for i in range(1, top_k+1)
        ]

    async def aretrieve(self, query: str, top_k: int):
        return self.retrieve(query, top_k)

@pytest.mark.integration
def test_complete_evaluation_pipeline(tmp_path):
    """Test complete evaluation pipeline."""
    # Create dataset
    import json
    dataset_path = tmp_path / "test_dataset.json"
    data = {
        "name": "test",
        "examples": [
            {
                "query_id": "q1",
                "query": "test query 1",
                "relevant_doc_ids": ["doc1", "doc2"]
            },
            {
                "query_id": "q2",
                "query": "test query 2",
                "relevant_doc_ids": ["doc2", "doc3"]
            }
        ]
    }
    with open(dataset_path, 'w') as f:
        json.dump(data, f)

    # Load dataset
    loader = DatasetLoader()
    dataset = loader.load(dataset_path)

    # Setup evaluation
    metrics = [
        PrecisionAtK(k=5),
        RecallAtK(k=5),
        MeanReciprocalRank()
    ]
    runner = BenchmarkRunner(metrics)

    # Run evaluation
    strategy = DummyStrategy()
    result = runner.run(strategy, dataset, "DummyStrategy")

    # Verify results
    assert result.strategy_name == "DummyStrategy"
    assert len(result.query_results) == 2
    assert "precision@5" in result.aggregate_metrics
    assert "recall@5" in result.aggregate_metrics
    assert "mrr" in result.aggregate_metrics

    # All metrics should have valid values
    for metric_value in result.aggregate_metrics.values():
        assert 0 <= metric_value <= 1

@pytest.mark.integration
def test_multi_strategy_comparison():
    """Test comparing multiple strategies."""
    # Implementation for comparing strategies
    pass
```

---

## Definition of Done

- [ ] Base metric interface defined
- [ ] All retrieval metrics implemented (Precision, Recall, MRR, NDCG)
- [ ] Quality metrics implemented (Semantic Similarity, Faithfulness)
- [ ] Performance and cost metrics implemented
- [ ] Dataset loader supports JSON, JSONL, CSV
- [ ] Dataset schema validation working
- [ ] Benchmark runner implemented
- [ ] Parallel evaluation supported
- [ ] Progress tracking working
- [ ] Results visualization dashboard complete
- [ ] Export to CSV/JSON working
- [ ] Statistical significance tests implemented
- [ ] All unit tests pass (>90% coverage)
- [ ] All integration tests pass
- [ ] Example datasets included
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] No linting errors

---

## Notes for Developers

1. **Start with Metrics**: Implement retrieval metrics first, they're the foundation

2. **Dataset Format**: Use a simple, extensible JSON format for datasets

3. **Metric Interface**: IMetric interface allows easy addition of custom metrics

4. **Parallel Evaluation**: Use multiprocessing for evaluating multiple strategies

5. **Caching**: Cache evaluation results to avoid re-running expensive operations

6. **Progress Tracking**: Use tqdm for progress bars - improves UX significantly

7. **Statistical Tests**: Use scipy for statistical tests, well-tested library

8. **Visualization**: Start with simple tables, enhance with charts later

9. **Quality Metrics**: For faithfulness and relevance, consider using LLM-based evaluation

10. **Benchmarking**: Include standard datasets (MS MARCO, Natural Questions) for comparison

11. **Export**: Support multiple formats for easy integration with other tools

12. **Regression Testing**: Track metrics over time to detect performance degradation

13. **Documentation**: Provide Jupyter notebooks with examples - helps adoption
