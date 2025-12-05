# Epic 8: Observability & Quality Assurance - Verification Guide

This document provides a comprehensive checklist for verifying the implementation of Epic 8 stories.

## Pre-Implementation Checklist

- [ ] Epic 4 (Priority RAG Strategies) has working strategies to monitor
- [ ] Epic 3 (Services Infrastructure) is complete
- [ ] Development environment set up
- [ ] All dependencies installed (structlog, prometheus-client, pandas, etc.)
- [ ] Test data prepared for evaluation

---

## Story 8.1: Monitoring & Logging System - Verification

### Implementation Checklist

#### Core Components

- [ ] `rag_factory/observability/logging/logger.py` - Logger implementation
  - [ ] `RAGLogger` class created
  - [ ] `LogContext` dataclass defined
  - [ ] `LogLevel` enum defined
  - [ ] Context manager for operations working
  - [ ] PII sanitization implemented
  - [ ] Structured logging with JSON output

- [ ] `rag_factory/observability/logging/config.py` - Logging configuration
  - [ ] Configuration dataclass created
  - [ ] File-based logging setup
  - [ ] Console logging setup
  - [ ] Log rotation configuration
  - [ ] Retention policy configuration

- [ ] `rag_factory/observability/logging/formatters.py` - Log formatters
  - [ ] JSON formatter implemented
  - [ ] Text formatter implemented
  - [ ] Timestamp formatting working
  - [ ] Context inclusion working

- [ ] `rag_factory/observability/logging/filters.py` - Log filters
  - [ ] PII filter implemented
  - [ ] Sampling filter (if needed)
  - [ ] Log level filter working

- [ ] `rag_factory/observability/metrics/collector.py` - Metrics collector
  - [ ] `MetricsCollector` class created
  - [ ] `PerformanceMetrics` dataclass defined
  - [ ] `MetricPoint` dataclass defined
  - [ ] Thread-safe operations
  - [ ] Time-series data storage
  - [ ] Metric aggregation working

- [ ] `rag_factory/observability/metrics/performance.py` - Performance metrics
  - [ ] Latency tracking
  - [ ] Throughput calculation
  - [ ] Percentile calculation (p50, p95, p99)
  - [ ] Success rate calculation

- [ ] `rag_factory/observability/metrics/cost.py` - Cost tracking
  - [ ] Token counting
  - [ ] Cost calculation per model
  - [ ] Cost aggregation
  - [ ] Cost per query metrics

- [ ] `rag_factory/observability/monitoring/dashboard.py` - Dashboard
  - [ ] Flask/FastAPI app created
  - [ ] Dashboard templates created
  - [ ] Static assets (CSS, JS) added

- [ ] `rag_factory/observability/monitoring/api.py` - API endpoints
  - [ ] `/api/health` endpoint working
  - [ ] `/api/metrics` endpoint working
  - [ ] `/api/metrics/summary` endpoint working
  - [ ] `/api/metrics/timeseries` endpoint working

#### Unit Tests

- [ ] `tests/unit/observability/test_logger.py`
  - [ ] Test logger initialization
  - [ ] Test operation context success
  - [ ] Test operation context failure
  - [ ] Test query logging
  - [ ] Test metrics logging
  - [ ] Test error logging
  - [ ] Test PII sanitization
  - [ ] Test context metadata
  - [ ] Coverage >85%

- [ ] `tests/unit/observability/test_metrics.py`
  - [ ] Test collector initialization
  - [ ] Test record successful query
  - [ ] Test record failed query
  - [ ] Test multiple queries
  - [ ] Test success rate calculation
  - [ ] Test latency percentiles
  - [ ] Test multiple strategies
  - [ ] Test summary generation
  - [ ] Test reset metrics
  - [ ] Test thread safety
  - [ ] Coverage >85%

- [ ] `tests/unit/observability/test_formatters.py`
  - [ ] Test JSON formatter
  - [ ] Test text formatter
  - [ ] Test timestamp formats
  - [ ] Coverage >80%

- [ ] `tests/unit/observability/test_filters.py`
  - [ ] Test PII filtering
  - [ ] Test query truncation
  - [ ] Test sampling filter
  - [ ] Coverage >80%

#### Integration Tests

- [ ] `tests/integration/observability/test_monitoring_integration.py`
  - [ ] End-to-end logging with real strategy
  - [ ] Error handling integration
  - [ ] Dashboard API tests
  - [ ] Performance overhead test
  - [ ] All tests passing

#### Acceptance Criteria Verification

- [ ] **AC1**: Structured logging working
  - [ ] `structlog` integrated
  - [ ] All operations logged with timestamps
  - [ ] Logs include strategy, query, execution time
  - [ ] Multiple log levels supported
  - [ ] JSON output format

- [ ] **AC2**: Performance metrics tracked
  - [ ] Latency tracked for all operations
  - [ ] Token usage counted
  - [ ] Cost calculated correctly
  - [ ] Throughput (QPS) tracked
  - [ ] Prometheus export format available

- [ ] **AC3**: Error tracking working
  - [ ] All exceptions captured
  - [ ] Stack traces included
  - [ ] Error context preserved
  - [ ] Error categorization working
  - [ ] Error rate tracked per strategy

- [ ] **AC4**: Query analytics working
  - [ ] Query patterns tracked
  - [ ] Response time percentiles calculated
  - [ ] Success/failure rates tracked
  - [ ] Analytics queryable via API

- [ ] **AC5**: Log management working
  - [ ] File-based logging with rotation
  - [ ] In-memory buffer (last 1000 logs)
  - [ ] Export capability working
  - [ ] Retention policy enforced
  - [ ] Log compression (if implemented)

- [ ] **AC6**: Dashboard accessible
  - [ ] Web dashboard loads
  - [ ] Live metrics displayed
  - [ ] Recent logs viewable
  - [ ] Strategy comparison view
  - [ ] Cost tracking displayed
  - [ ] Filtering working

- [ ] **AC7**: Testing complete
  - [ ] Unit tests: 100% pass
  - [ ] Integration tests: 100% pass
  - [ ] Coverage >85%
  - [ ] Performance tests pass
  - [ ] Security review (no PII leaks)

#### Performance Verification

- [ ] Logging overhead <5ms per operation
- [ ] Dashboard response time <200ms
- [ ] Support 100+ concurrent operations
- [ ] Memory usage reasonable (<100MB for 1000 logs)
- [ ] No performance regression in monitored code

#### Manual Testing

- [ ] Test basic logging
  ```python
  logger = RAGLogger()

  with logger.operation("retrieve", strategy="vector_search", query="test") as ctx:
      # Simulate work
      time.sleep(0.1)
      ctx.metadata["results"] = 5

  print(f"Elapsed: {ctx.elapsed_ms():.2f}ms")
  ```

- [ ] Test metrics collection
  ```python
  collector = MetricsCollector()

  collector.record_query(
      strategy="vector_search",
      latency_ms=45.2,
      tokens=150,
      cost=0.001,
      success=True
  )

  metrics = collector.get_metrics("vector_search")
  print(f"Avg latency: {metrics['avg_latency_ms']:.2f}ms")
  ```

- [ ] Test dashboard access
  - Open browser to `http://localhost:8080`
  - Verify metrics displayed
  - Verify logs viewable
  - Test filtering
  - Test strategy comparison

- [ ] Verify PII filtering
  - Log query with sensitive data
  - Verify sensitive data sanitized
  - Check log files

---

## Story 8.2: Evaluation Framework - Verification

### Implementation Checklist

#### Core Components

- [ ] `rag_factory/evaluation/metrics/base.py` - Base metric interface
  - [ ] `IMetric` abstract class defined
  - [ ] `MetricResult` dataclass created
  - [ ] `MetricType` enum defined
  - [ ] Base methods implemented

- [ ] `rag_factory/evaluation/metrics/retrieval.py` - Retrieval metrics
  - [ ] `PrecisionAtK` implemented
  - [ ] `RecallAtK` implemented
  - [ ] `MeanReciprocalRank` implemented
  - [ ] `NDCG` implemented
  - [ ] `HitRate` implemented (optional)

- [ ] `rag_factory/evaluation/metrics/quality.py` - Quality metrics
  - [ ] `SemanticSimilarity` implemented
  - [ ] `Faithfulness` implemented
  - [ ] `Relevance` implemented (optional)
  - [ ] `Completeness` implemented (optional)

- [ ] `rag_factory/evaluation/metrics/performance.py` - Performance metrics
  - [ ] Latency metrics
  - [ ] Throughput metrics
  - [ ] Resource usage metrics

- [ ] `rag_factory/evaluation/metrics/cost.py` - Cost metrics
  - [ ] Token usage tracking
  - [ ] API cost calculation
  - [ ] Cost per query metrics

- [ ] `rag_factory/evaluation/datasets/schema.py` - Dataset schema
  - [ ] `EvaluationExample` dataclass defined
  - [ ] `EvaluationDataset` dataclass defined
  - [ ] Schema validation implemented

- [ ] `rag_factory/evaluation/datasets/loader.py` - Dataset loader
  - [ ] `DatasetLoader` class created
  - [ ] JSON loading working
  - [ ] JSONL loading working
  - [ ] CSV loading working
  - [ ] Error handling for invalid formats

- [ ] `rag_factory/evaluation/datasets/generator.py` - Synthetic data generator
  - [ ] Synthetic dataset generation working
  - [ ] Configurable generation parameters

- [ ] `rag_factory/evaluation/benchmarks/runner.py` - Benchmark runner
  - [ ] `BenchmarkRunner` class created
  - [ ] `BenchmarkResult` dataclass defined
  - [ ] Single strategy evaluation working
  - [ ] Progress tracking with tqdm
  - [ ] Metric aggregation working

- [ ] `rag_factory/evaluation/benchmarks/parallel.py` - Parallel execution
  - [ ] Parallel strategy evaluation working
  - [ ] Worker pool management
  - [ ] Result aggregation

- [ ] `rag_factory/evaluation/benchmarks/checkpoint.py` - Checkpointing
  - [ ] Checkpoint saving
  - [ ] Resume from checkpoint
  - [ ] Checkpoint validation

- [ ] `rag_factory/evaluation/benchmarks/cache.py` - Results caching
  - [ ] Cache implementation
  - [ ] Cache key generation
  - [ ] Cache invalidation

- [ ] `rag_factory/evaluation/analysis/statistics.py` - Statistical tests
  - [ ] T-test implementation
  - [ ] Confidence intervals
  - [ ] Effect size (Cohen's d)
  - [ ] P-value calculation
  - [ ] Multiple comparison correction

- [ ] `rag_factory/evaluation/visualization/dashboard.py` - Visualization dashboard
  - [ ] Dashboard implementation
  - [ ] Comparison tables
  - [ ] Performance charts
  - [ ] Interactive filtering

- [ ] `rag_factory/evaluation/exporters/csv_exporter.py` - CSV export
  - [ ] CSV export working
  - [ ] Configurable fields
  - [ ] Summary statistics included

- [ ] `rag_factory/evaluation/exporters/json_exporter.py` - JSON export
  - [ ] JSON export working
  - [ ] Proper formatting
  - [ ] Metadata included

- [ ] `rag_factory/evaluation/exporters/html_exporter.py` - HTML export
  - [ ] HTML report generation
  - [ ] Styling applied
  - [ ] Charts embedded

#### Unit Tests

- [ ] `tests/unit/evaluation/test_retrieval_metrics.py`
  - [ ] Test Precision@K calculation
  - [ ] Test Recall@K calculation
  - [ ] Test MRR calculation
  - [ ] Test MRR with no relevant docs
  - [ ] Test NDCG calculation
  - [ ] Test perfect NDCG (score = 1.0)
  - [ ] Coverage >90%

- [ ] `tests/unit/evaluation/test_quality_metrics.py`
  - [ ] Test semantic similarity
  - [ ] Test faithfulness calculation
  - [ ] Test with empty inputs
  - [ ] Coverage >90%

- [ ] `tests/unit/evaluation/test_dataset_loader.py`
  - [ ] Test JSON loading
  - [ ] Test JSONL loading
  - [ ] Test CSV loading
  - [ ] Test nonexistent file error
  - [ ] Test unsupported format error
  - [ ] Test schema validation
  - [ ] Coverage >90%

- [ ] `tests/unit/evaluation/test_benchmark_runner.py`
  - [ ] Test runner initialization
  - [ ] Test single strategy evaluation
  - [ ] Test metric aggregation
  - [ ] Test with mock strategy
  - [ ] Coverage >90%

- [ ] `tests/unit/evaluation/test_statistics.py`
  - [ ] Test t-test calculation
  - [ ] Test confidence intervals
  - [ ] Test effect size
  - [ ] Test significance detection
  - [ ] Coverage >90%

- [ ] `tests/unit/evaluation/test_exporters.py`
  - [ ] Test CSV export
  - [ ] Test JSON export
  - [ ] Test HTML export
  - [ ] Test export with custom fields
  - [ ] Coverage >85%

#### Integration Tests

- [ ] `tests/integration/evaluation/test_evaluation_integration.py`
  - [ ] End-to-end evaluation pipeline
  - [ ] Multi-strategy comparison
  - [ ] Statistical testing integration
  - [ ] Export integration
  - [ ] All tests passing

#### Acceptance Criteria Verification

- [ ] **AC1**: Metrics system complete
  - [ ] All retrieval metrics implemented
  - [ ] Quality metrics implemented
  - [ ] Performance metrics implemented
  - [ ] Cost metrics implemented
  - [ ] Metrics validated against test cases

- [ ] **AC2**: Dataset management working
  - [ ] Load JSON datasets
  - [ ] Load JSONL datasets
  - [ ] Load CSV datasets
  - [ ] Schema validation
  - [ ] Multiple datasets supported
  - [ ] Dataset statistics available

- [ ] **AC3**: Benchmarking working
  - [ ] Evaluate single strategy
  - [ ] Evaluate multiple strategies
  - [ ] Parallel evaluation
  - [ ] Progress tracking
  - [ ] Results caching
  - [ ] Checkpoint/resume

- [ ] **AC4**: Visualization working
  - [ ] Dashboard accessible
  - [ ] Comparison tables rendered
  - [ ] Charts displayed
  - [ ] Query-level inspection
  - [ ] Interactive filtering

- [ ] **AC5**: Export working
  - [ ] CSV export
  - [ ] JSON export
  - [ ] HTML reports
  - [ ] Summary statistics
  - [ ] Configurable fields

- [ ] **AC6**: Statistical testing working
  - [ ] T-tests implemented
  - [ ] Confidence intervals
  - [ ] Effect sizes
  - [ ] P-values displayed
  - [ ] Significance indicators

- [ ] **AC7**: Testing complete
  - [ ] Unit tests: 100% pass
  - [ ] Integration tests: 100% pass
  - [ ] Coverage >90%
  - [ ] Example datasets included
  - [ ] Documentation complete

#### Performance Verification

- [ ] Evaluate 100 queries in <5 minutes
- [ ] Parallel evaluation scales with workers
- [ ] Memory usage reasonable
- [ ] Statistical tests complete quickly (<1s)
- [ ] Export completes in reasonable time

#### Manual Testing

- [ ] Test dataset loading
  ```python
  loader = DatasetLoader()
  dataset = loader.load("examples/evaluation/sample_dataset.json")

  print(f"Dataset: {dataset.name}")
  print(f"Examples: {len(dataset)}")
  print(f"First query: {dataset[0].query}")
  ```

- [ ] Test metric calculation
  ```python
  precision = PrecisionAtK(k=5)
  result = precision.compute(
      retrieved_ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
      relevant_ids={"doc1", "doc3", "doc5"}
  )
  print(f"Precision@5: {result.value:.4f}")
  ```

- [ ] Test benchmark runner
  ```python
  metrics = [PrecisionAtK(k=5), RecallAtK(k=10), MeanReciprocalRank()]
  runner = BenchmarkRunner(metrics)

  result = runner.run(strategy, dataset, "VectorSearch")

  print(f"\nResults for {result.strategy_name}:")
  for metric, value in result.aggregate_metrics.items():
      print(f"  {metric}: {value:.4f}")
  ```

- [ ] Test statistical comparison
  ```python
  analyzer = StatisticalAnalyzer()
  comparison = analyzer.compare_strategies(
      baseline_result,
      treatment_result,
      metric="precision@5"
  )

  print(f"Improvement: {comparison['improvement_percent']:.1f}%")
  print(f"P-value: {comparison['p_value']:.4f}")
  print(f"Significant: {comparison['is_significant']}")
  ```

- [ ] Test export
  ```python
  exporter = CSVExporter()
  exporter.export_results(result, "results/evaluation.csv")

  # Verify CSV created and readable
  ```

- [ ] Test dashboard
  - Open browser to dashboard URL
  - Verify results displayed
  - Test comparison view
  - Test filtering
  - Test export from UI

---

## Integration Verification (Both Stories)

### Combined System Testing

- [ ] Test monitoring during evaluation
  ```python
  # Setup monitoring
  logger = RAGLogger()
  metrics_collector = MetricsCollector()

  # Setup evaluation
  loader = DatasetLoader()
  dataset = loader.load("test_dataset.json")
  runner = BenchmarkRunner(metrics)

  # Run evaluation with monitoring
  with logger.operation("evaluation", strategy="test") as ctx:
      result = runner.run(strategy, dataset)
      ctx.metadata["queries_evaluated"] = len(dataset)

  # Verify both systems working
  assert len(result.query_results) > 0
  assert metrics_collector.get_summary()["total_queries"] > 0
  ```

- [ ] Verify logs capture evaluation metrics
- [ ] Verify dashboard shows evaluation results
- [ ] Test end-to-end: monitor evaluation, view results, export

### Quality Verification

- [ ] Run evaluation on sample strategies
- [ ] Verify metrics detect quality differences
- [ ] Statistical tests show significance when appropriate
- [ ] Logging captures all evaluation operations
- [ ] No errors during long evaluation runs

### Performance Verification

- [ ] Logging overhead acceptable during evaluation
- [ ] Evaluation completes in expected time
- [ ] Memory usage remains bounded
- [ ] Dashboard responsive with evaluation data
- [ ] Export handles large result sets

---

## Code Quality Verification

### Code Review Checklist

- [ ] All code follows project style guide
- [ ] No linting errors (flake8, pylint)
- [ ] Type hints on all public methods
- [ ] Docstrings on all public classes and methods
- [ ] No TODOs or FIXME comments remaining
- [ ] Error handling comprehensive
- [ ] Logging appropriate
- [ ] Thread safety verified where needed

### Documentation Verification

- [ ] All story documents complete
- [ ] README.md comprehensive
- [ ] VERIFICATION.md complete
- [ ] Code examples working
- [ ] API documentation complete
- [ ] Configuration examples provided
- [ ] Dashboard usage documented

### Testing Verification

- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] Test coverage >85% (Story 8.1)
- [ ] Test coverage >90% (Story 8.2)
- [ ] Performance benchmarks documented
- [ ] Edge cases tested
- [ ] Example datasets provided

---

## Deployment Readiness Checklist

### Pre-Deployment

- [ ] All acceptance criteria met
- [ ] All tests passing
- [ ] Performance benchmarks met
- [ ] Code reviewed
- [ ] Documentation complete
- [ ] Configuration validated
- [ ] Example datasets included

### Configuration

- [ ] Logging configuration created
  - [ ] Log levels set appropriately
  - [ ] File rotation configured
  - [ ] Retention policies set
  - [ ] PII filtering enabled

- [ ] Monitoring configuration created
  - [ ] Dashboard port configured
  - [ ] Metrics exporters configured
  - [ ] Authentication set up (if needed)

- [ ] Evaluation configuration created
  - [ ] Default datasets specified
  - [ ] Metrics configured
  - [ ] Parallel execution settings
  - [ ] Export paths configured

### Deployment Steps

- [ ] Install dependencies
  ```bash
  pip install structlog prometheus-client pandas matplotlib plotly
  ```

- [ ] Verify imports
  ```python
  from rag_factory.observability.logging import RAGLogger
  from rag_factory.observability.metrics import MetricsCollector
  from rag_factory.evaluation.benchmarks import BenchmarkRunner
  from rag_factory.evaluation.metrics import PrecisionAtK
  ```

- [ ] Start monitoring dashboard (if configured)
  ```bash
  python -m rag_factory.observability.monitoring.dashboard
  ```

- [ ] Verify dashboard accessible
- [ ] Run smoke tests

### Post-Deployment

- [ ] Smoke tests pass
- [ ] Monitoring dashboard accessible
- [ ] Logs being written correctly
- [ ] Metrics being collected
- [ ] Evaluation framework functional
- [ ] Performance metrics tracked
- [ ] No errors in logs

---

## Sign-Off

### Story 8.1: Monitoring & Logging System
- [ ] Developer: _____________________ Date: _______
- [ ] Reviewer: ______________________ Date: _______
- [ ] QA: ___________________________ Date: _______

### Story 8.2: Evaluation Framework
- [ ] Developer: _____________________ Date: _______
- [ ] Reviewer: ______________________ Date: _______
- [ ] QA: ___________________________ Date: _______

### Epic 8 Final Sign-Off
- [ ] Tech Lead: _____________________ Date: _______
- [ ] Product Owner: _________________ Date: _______

---

## Known Issues and Limitations

### Document Known Issues Here
| Issue | Severity | Story | Status | Notes |
|-------|----------|-------|--------|-------|
|       |          |       |        |       |

---

## Performance Benchmarks

### Monitoring System (Story 8.1)

Record actual performance metrics:

| Metric | Target | Actual | Pass/Fail |
|--------|--------|--------|-----------|
| Logging overhead | <5ms | ___ms | ___ |
| Dashboard response | <200ms | ___ms | ___ |
| Memory usage (1K logs) | <100MB | ___MB | ___ |
| Concurrent operations | 100+ | ___ | ___ |

### Evaluation Framework (Story 8.2)

Record actual performance metrics:

| Metric | Target | Actual | Pass/Fail |
|--------|--------|--------|-----------|
| 100 query evaluation | <5 min | ___min | ___ |
| Parallel speedup (4 workers) | ~3x | ___x | ___ |
| Statistical test time | <1s | ___s | ___ |
| Export time (1000 results) | <5s | ___s | ___ |

---

## Future Improvements

### Post-Epic Enhancements

**Monitoring & Logging**:
- [ ] Anomaly detection in metrics
- [ ] Automated alerting
- [ ] Distributed tracing with OpenTelemetry
- [ ] Integration with external systems (Datadog, ELK)
- [ ] Real-time dashboard updates (WebSocket)
- [ ] Mobile dashboard view

**Evaluation Framework**:
- [ ] LLM-as-judge for quality evaluation
- [ ] Multi-modal evaluation (images, tables)
- [ ] Human-in-the-loop evaluation
- [ ] Automated hyperparameter tuning
- [ ] Cost optimization recommendations
- [ ] MLflow/Weights & Biases integration
- [ ] Real-time A/B testing
- [ ] Regression detection alerts

---

## References

- Story 8.1: `story-8.1-monitoring-logging-system.md`
- Story 8.2: `story-8.2-evaluation-framework.md`
- Epic README: `README.md`
- Epic 8 documentation: `docs/epics/epic-08-observability.md`

---

## Additional Resources

### Example Datasets

Create these example datasets before testing:

**examples/evaluation/sample_dataset.json**:
```json
{
  "name": "sample_evaluation",
  "examples": [
    {
      "query_id": "q1",
      "query": "What is machine learning?",
      "ground_truth_answer": "Machine learning is a subset of AI...",
      "relevant_doc_ids": ["doc1", "doc2", "doc3"]
    },
    {
      "query_id": "q2",
      "query": "Explain neural networks",
      "ground_truth_answer": "Neural networks are...",
      "relevant_doc_ids": ["doc4", "doc5"]
    }
  ]
}
```

### Testing Scripts

Create helper scripts for verification:

**scripts/verify_monitoring.py**:
```python
"""Verify monitoring system is working."""
from rag_factory.observability.logging import RAGLogger
from rag_factory.observability.metrics import MetricsCollector

def verify_monitoring():
    logger = RAGLogger()
    collector = MetricsCollector()

    # Test logging
    with logger.operation("test", strategy="verify"):
        pass

    # Test metrics
    collector.record_query("verify", 10.0, success=True)

    metrics = collector.get_metrics("verify")
    assert metrics["total_queries"] == 1

    print("✓ Monitoring system verified")

if __name__ == "__main__":
    verify_monitoring()
```

**scripts/verify_evaluation.py**:
```python
"""Verify evaluation framework is working."""
from rag_factory.evaluation.datasets import DatasetLoader
from rag_factory.evaluation.metrics import PrecisionAtK
from rag_factory.evaluation.benchmarks import BenchmarkRunner

def verify_evaluation():
    # Test dataset loading
    loader = DatasetLoader()
    # Load example dataset

    # Test metrics
    precision = PrecisionAtK(k=5)
    result = precision.compute(
        retrieved_ids=["1", "2", "3", "4", "5"],
        relevant_ids={"1", "3", "5"}
    )
    assert result.value == 0.6

    print("✓ Evaluation framework verified")

if __name__ == "__main__":
    verify_evaluation()
```

Run these scripts as part of deployment verification.
