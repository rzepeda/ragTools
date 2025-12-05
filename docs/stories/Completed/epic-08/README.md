# Epic 8: Observability & Quality Assurance

## Overview

Epic 8 focuses on building comprehensive monitoring, logging, and evaluation frameworks to ensure system quality and enable continuous improvement. This epic provides the observability infrastructure needed to understand, debug, and optimize RAG system performance.

**Key Components**:
1. **Monitoring & Logging System** - Track all operations with comprehensive metrics
2. **Evaluation Framework** - Systematically evaluate and compare RAG strategies

## Epic Goal

Build comprehensive monitoring, logging, and evaluation frameworks to ensure system quality and enable continuous improvement through data-driven decision making.

## Epic Story Points

**Total: 21 points**
- Story 8.1: Build Monitoring & Logging System (8 points)
- Story 8.2: Create Evaluation Framework (13 points)

## Dependencies

- **Epic 4**: Priority RAG Strategies (need working strategies to monitor and evaluate)
- **Epic 3**: Services Infrastructure (LLM and Embedding services for quality metrics)

## Stories

### Story 8.1: Build Monitoring & Logging System (8 points)

**Status**: Not Started

**Objective**: Implement comprehensive logging and monitoring for debugging and optimization

**Key Features**:
- Structured logging with `structlog`
- Performance metrics tracking (latency, cost, token usage)
- Error tracking with full stack traces
- Query analytics and aggregation
- Real-time monitoring dashboard
- Export to external systems (ELK, Datadog, CloudWatch)
- OpenTelemetry and Prometheus integration

**Technical Components**:
- RAG Logger with context management
- Metrics Collector with time-series support
- Web-based monitoring dashboard
- Log aggregation and rotation
- PII filtering and sanitization
- Integration with external monitoring tools

**Acceptance Criteria**:
- ✅ All strategy executions logged with timestamps
- ✅ Performance metrics tracked (latency, tokens, cost)
- ✅ Error tracking with stack traces
- ✅ Web dashboard accessible
- ✅ Log rotation and retention working
- ✅ Logging overhead <5ms per operation
- ✅ All tests passing (>85% coverage)

### Story 8.2: Create Evaluation Framework (13 points)

**Status**: Not Started

**Objective**: Systematically evaluate and compare RAG strategies for informed decision-making

**Key Features**:
- Comprehensive metrics (Precision@K, Recall@K, MRR, NDCG)
- Quality metrics (Semantic Similarity, Faithfulness, Relevance)
- Performance and cost metrics
- Test dataset management (JSON, CSV, JSONL)
- Benchmarking suite with parallel execution
- Results visualization dashboard
- Statistical significance testing
- Export to CSV/JSON/HTML

**Technical Components**:
- Metric system with pluggable metrics
- Dataset loader and validator
- Benchmark runner with progress tracking
- Statistical analysis tools
- Visualization dashboard
- Results exporters (CSV, JSON, HTML)
- Regression testing framework

**Acceptance Criteria**:
- ✅ All retrieval and quality metrics implemented
- ✅ Dataset management working (JSON, CSV, JSONL)
- ✅ Benchmarking with parallel execution
- ✅ Web dashboard for results visualization
- ✅ Statistical significance tests implemented
- ✅ Export to multiple formats working
- ✅ Evaluate 100 queries in <5 minutes
- ✅ All tests passing (>90% coverage)

## Sprint Planning

### Sprint 4 (8 points)
- Story 8.1: Monitoring & Logging System (8 points)
- Can combine with Epic 4 Story 4.3 (Query Expansion)

### Sprint 8 (13 points)
- Story 8.2: Evaluation Framework (13 points)
- Can combine with final testing and optimization

## Technical Stack

### Monitoring & Logging
- **structlog**: Structured logging framework
- **python-json-logger**: JSON log formatting
- **prometheus-client**: Metrics export
- **flask**: Web dashboard (or FastAPI)
- **opentelemetry**: Distributed tracing
- **psutil**: System metrics

### Evaluation Framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **scipy**: Statistical functions
- **matplotlib/plotly**: Visualization
- **seaborn**: Statistical plots
- **scikit-learn**: ML metrics and utilities
- **sentence-transformers**: Semantic similarity
- **tqdm**: Progress bars

## Success Criteria

### Performance Metrics
- [ ] Retrieval accuracy > 85% on benchmark datasets
- [ ] Query latency < 2 seconds (p95)
- [ ] Cost per query < $0.02
- [ ] Logging overhead < 5ms per operation
- [ ] Dashboard response time < 200ms

### Code Quality
- [ ] Test coverage > 80% for observability components
- [ ] Test coverage > 90% for evaluation framework
- [ ] All critical paths logged
- [ ] Error rates tracked per strategy
- [ ] No performance regression from monitoring

### System Metrics
- [ ] All strategy executions logged
- [ ] Performance metrics tracked continuously
- [ ] Evaluation framework validates strategy improvements
- [ ] Can compare multiple strategies quantitatively
- [ ] Dashboard accessible and intuitive

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Observability System                      │
└─────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
    ┌───────────────────────┐      ┌───────────────────────┐
    │   Monitoring &        │      │   Evaluation          │
    │   Logging (8.1)       │      │   Framework (8.2)     │
    └───────────┬───────────┘      └───────────┬───────────┘
                │                               │
    ┌───────────┴──────────┐      ┌────────────┴──────────┐
    │                      │      │                       │
    ▼                      ▼      ▼                       ▼
┌─────────┐      ┌──────────┐ ┌─────────┐      ┌──────────┐
│ Logger  │      │ Metrics  │ │ Dataset │      │Benchmark │
│         │      │Collector │ │ Loader  │      │ Runner   │
└────┬────┘      └────┬─────┘ └────┬────┘      └────┬─────┘
     │                │            │                 │
     ▼                ▼            ▼                 ▼
┌─────────────────────────────────────────────────────────┐
│              RAG Strategies (Epic 4, 5, 6, 7)           │
│  - Vector Search                                         │
│  - Re-ranking                                            │
│  - Query Expansion                                       │
│  - Agentic RAG                                           │
└─────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Dashboard/Reports   │
                    │   - Metrics view      │
                    │   - Comparison view   │
                    │   - Error analysis    │
                    └───────────────────────┘
```

## Implementation Order

1. **Story 8.1: Monitoring & Logging** (FIRST)
   - Core infrastructure for observability
   - Needed to track evaluation runs
   - Provides debugging capabilities
   - Enables performance monitoring

2. **Story 8.2: Evaluation Framework** (SECOND)
   - Uses logging infrastructure from 8.1
   - Validates strategy improvements
   - Enables data-driven decisions
   - Provides regression testing

## Configuration Example

```yaml
# config.yaml for Epic 8

# Monitoring & Logging (Story 8.1)
observability:
  logging:
    level: "INFO"  # DEBUG | INFO | WARNING | ERROR | CRITICAL
    format: "json"  # json | text
    output:
      - type: "file"
        path: "logs/rag.log"
        rotation: "daily"  # daily | size
        max_size_mb: 100
        retention_days: 30
      - type: "console"
        enabled: true

    filters:
      sanitize_pii: true
      max_query_length: 500

    sampling:
      enabled: false
      rate: 0.1  # Sample 10% of logs

  metrics:
    enabled: true
    collectors:
      - "performance"  # Latency, throughput
      - "cost"         # Token usage, API costs
      - "quality"      # Success rates, errors

    exporters:
      - type: "prometheus"
        port: 9090
      - type: "statsd"
        host: "localhost"
        port: 8125

  monitoring:
    dashboard:
      enabled: true
      host: "0.0.0.0"
      port: 8080
      auth:
        enabled: false
        username: "admin"
        password: "changeme"

# Evaluation Framework (Story 8.2)
evaluation:
  datasets:
    default: "examples/evaluation/benchmark_dataset.json"
    custom_datasets:
      - name: "domain_specific"
        path: "data/evaluation/domain.jsonl"
      - name: "stress_test"
        path: "data/evaluation/stress.csv"

  metrics:
    retrieval:
      - precision@5
      - recall@10
      - mrr
      - ndcg@10

    quality:
      - semantic_similarity
      - faithfulness
      - relevance

    performance:
      - latency_ms
      - throughput_qps

    cost:
      - token_usage
      - api_cost

  benchmarking:
    parallel_execution: true
    max_workers: 4
    enable_caching: true
    checkpoint_interval: 50  # Save every 50 queries

  statistical_tests:
    enabled: true
    confidence_level: 0.95
    multiple_comparison_correction: "bonferroni"

  visualization:
    dashboard:
      enabled: true
      port: 8081

    export:
      formats:
        - csv
        - json
        - html
      output_dir: "results/evaluation"
```

## Usage Example

```python
from rag_factory.observability.logging.logger import RAGLogger
from rag_factory.observability.metrics.collector import MetricsCollector
from rag_factory.evaluation.benchmarks.runner import BenchmarkRunner
from rag_factory.evaluation.metrics.retrieval import PrecisionAtK, RecallAtK, MeanReciprocalRank
from rag_factory.evaluation.datasets.loader import DatasetLoader

# ========================================
# Part 1: Monitoring & Logging (Story 8.1)
# ========================================

# Initialize logger and metrics collector
logger = RAGLogger()
metrics_collector = MetricsCollector()

# Use in your RAG strategy
def execute_rag_query(query: str, strategy: IRAGStrategy):
    """Execute RAG query with monitoring."""

    # Log operation with context
    with logger.operation("retrieve", strategy="vector_search", query=query) as ctx:
        # Execute strategy
        results = strategy.retrieve(query, top_k=5)

        # Add metadata
        ctx.metadata["results_count"] = len(results)
        ctx.metadata["total_tokens"] = 150  # From LLM call

    # Record metrics
    metrics_collector.record_query(
        strategy="vector_search",
        latency_ms=ctx.elapsed_ms(),
        tokens=150,
        cost=0.001,
        success=True
    )

    return results

# View metrics
metrics = metrics_collector.get_metrics("vector_search")
print(f"Average latency: {metrics['avg_latency_ms']:.2f}ms")
print(f"Success rate: {metrics['success_rate']:.1f}%")
print(f"P95 latency: {metrics['p95_latency_ms']:.2f}ms")

# Get overall summary
summary = metrics_collector.get_summary()
print(f"Total queries: {summary['total_queries']}")
print(f"QPS: {summary['queries_per_second']:.2f}")
print(f"Total cost: ${summary['total_cost']:.4f}")

# ========================================
# Part 2: Evaluation Framework (Story 8.2)
# ========================================

# Load evaluation dataset
loader = DatasetLoader()
dataset = loader.load("examples/evaluation/benchmark_dataset.json")

print(f"Loaded dataset: {dataset.name}")
print(f"Number of examples: {len(dataset)}")

# Setup evaluation metrics
metrics = [
    PrecisionAtK(k=5),
    RecallAtK(k=10),
    MeanReciprocalRank(),
    NDCG(k=10)
]

# Create benchmark runner
runner = BenchmarkRunner(metrics)

# Evaluate a strategy
from rag_factory.strategies.vector_search import VectorSearchStrategy

vector_strategy = VectorSearchStrategy(config)
result = runner.run(vector_strategy, dataset, "VectorSearch")

# View results
print(f"\nStrategy: {result.strategy_name}")
print(f"Dataset: {result.dataset_name}")
print(f"Execution time: {result.execution_time:.2f}s")
print("\nAggregate Metrics:")
for metric_name, value in result.aggregate_metrics.items():
    print(f"  {metric_name}: {value:.4f}")

# Compare multiple strategies
strategies = [
    ("VectorSearch", VectorSearchStrategy(config)),
    ("WithReranking", RerankingStrategy(config)),
    ("WithExpansion", QueryExpansionStrategy(config))
]

results = []
for name, strategy in strategies:
    result = runner.run(strategy, dataset, name)
    results.append(result)

# Export comparison
from rag_factory.evaluation.exporters.csv_exporter import CSVExporter

exporter = CSVExporter()
exporter.export_comparison(results, "results/comparison.csv")

# Statistical significance test
from rag_factory.evaluation.analysis.statistics import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()
comparison = analyzer.compare_strategies(
    results[0],  # Baseline
    results[1],  # Treatment
    metric="precision@5"
)

print(f"\nStatistical Comparison:")
print(f"Baseline: {comparison['baseline_mean']:.4f}")
print(f"Treatment: {comparison['treatment_mean']:.4f}")
print(f"Improvement: {comparison['improvement_percent']:.1f}%")
print(f"P-value: {comparison['p_value']:.4f}")
print(f"Significant: {comparison['is_significant']}")

# ========================================
# Part 3: Dashboard Access
# ========================================

# Monitoring dashboard available at: http://localhost:8080
# - View real-time metrics
# - See recent logs
# - Compare strategy performance
# - Track costs

# Evaluation dashboard available at: http://localhost:8081
# - Compare evaluation results
# - View metric distributions
# - Inspect query-level results
# - Export reports
```

## Testing Strategy

### Unit Tests

**Monitoring & Logging**:
- Test logger initialization and configuration
- Test context managers and operation tracking
- Test PII filtering and sanitization
- Test metrics collection and aggregation
- Test thread safety of collectors
- Mock external dependencies

**Evaluation Framework**:
- Test each metric independently
- Test dataset loading from different formats
- Test benchmark runner with mock strategies
- Test statistical analysis functions
- Test export to different formats
- Validate metric calculations against known results

### Integration Tests

**Monitoring & Logging**:
- Test end-to-end logging with real strategies
- Test dashboard API endpoints
- Test metrics export to Prometheus
- Verify logging overhead is acceptable
- Test error recovery and graceful degradation

**Evaluation Framework**:
- Test complete evaluation pipeline
- Test with real RAG strategies
- Validate quality improvements are detected
- Test parallel evaluation
- Benchmark performance meets requirements

### Performance Tests
- Logging overhead < 5ms per operation
- Evaluate 100 queries in < 5 minutes
- Dashboard response time < 200ms
- Metrics collection has minimal memory overhead
- Statistical tests complete in reasonable time

## Monitoring and Metrics

### Logging Metrics (Story 8.1)
- **Log volume**: Total logs per hour
- **Error rate**: Errors per 1000 operations
- **Logging overhead**: Time spent in logging code
- **Dashboard usage**: Active users, page views

### Evaluation Metrics (Story 8.2)
- **Evaluation runs**: Number of evaluations completed
- **Dataset coverage**: Queries evaluated per dataset
- **Metric stability**: Variance in repeated evaluations
- **Export frequency**: Reports generated per day

### System Health
- Memory usage of collectors and loggers
- Disk space for log storage
- CPU usage during evaluation
- API costs for quality metrics (if using LLM)

## Cost Considerations

### Monitoring & Logging Costs
- **Storage**: Log files and time-series data
  - Estimated: ~100MB-1GB per day
  - Mitigation: Compression, retention policies
- **External Services**: If using Datadog, ELK
  - Estimated: $0-$100+ per month
  - Mitigation: Use local Prometheus + Grafana

### Evaluation Framework Costs
- **Compute**: Running evaluations
  - Minimal for retrieval metrics
  - GPU recommended for semantic similarity
- **API Costs**: Quality metrics using LLMs
  - Estimated: $0.01-0.10 per query evaluated
  - Mitigation: Use local models, batch processing
- **Storage**: Evaluation results
  - Estimated: ~10-100MB per evaluation run
  - Mitigation: Compress old results, periodic cleanup

## Risk Management

### Risks and Mitigations

1. **Performance Risk**: Monitoring adds overhead
   - **Mitigation**: Async logging, sampling, optimization

2. **Storage Risk**: Logs and results consume disk space
   - **Mitigation**: Rotation, compression, retention policies

3. **Quality Risk**: Metrics may not capture real quality
   - **Mitigation**: Multiple metrics, human evaluation, A/B testing

4. **Complexity Risk**: Complex evaluation setup
   - **Mitigation**: Clear documentation, examples, CLI tools

5. **Cost Risk**: LLM-based quality metrics expensive
   - **Mitigation**: Local models, caching, batch processing

## Future Enhancements

### Post-MVP Improvements
- [ ] Anomaly detection in metrics
- [ ] Automated alerting for regressions
- [ ] MLflow integration for experiment tracking
- [ ] Weights & Biases integration
- [ ] Real-time A/B testing framework
- [ ] LLM-as-judge for quality evaluation
- [ ] Multi-modal evaluation (images, tables)
- [ ] Human-in-the-loop evaluation
- [ ] Automated hyperparameter tuning based on metrics
- [ ] Cost optimization recommendations

## References

### Academic Papers
- "Measuring Massive Multitask Language Understanding" (Hendrycks et al., 2020)
- "Beyond Accuracy: Behavioral Testing of NLP Models" (Ribeiro et al., 2020)
- "Evaluation of Retrieval-Augmented Generation" (Lewis et al., 2020)

### Libraries and Tools
- [structlog](https://www.structlog.org/) - Structured logging
- [Prometheus](https://prometheus.io/) - Metrics and monitoring
- [Grafana](https://grafana.com/) - Visualization dashboards
- [OpenTelemetry](https://opentelemetry.io/) - Distributed tracing
- [scikit-learn](https://scikit-learn.org/) - ML metrics
- [RAGAS](https://github.com/explodinggradients/ragas) - RAG evaluation framework (inspiration)

### Best Practices
- [Google SRE Book](https://sre.google/books/) - Monitoring and observability
- [Observability Engineering](https://www.oreilly.com/library/view/observability-engineering/9781492076438/) - Honeycomb
- [Effective Pandas](https://pandas.pydata.org/) - Data analysis

## Getting Help

For questions or issues:
1. Check story documentation in this directory
2. Review code examples in tests and examples
3. See Epic 8 technical specifications
4. Review monitoring dashboard documentation
5. Consult team members for metric definitions and thresholds
6. Check logs for detailed error information
