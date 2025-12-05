# Story 8.1: Build Monitoring & Logging System

**Story ID:** 8.1
**Epic:** Epic 8 - Observability & Quality Assurance
**Story Points:** 8
**Priority:** High
**Dependencies:** Epic 4 (RAG Strategies to monitor)

---

## User Story

**As a** developer
**I want** comprehensive logging and monitoring
**So that** I can debug and optimize RAG performance

---

## Detailed Requirements

### Functional Requirements

1. **Structured Logging System**
   - Use `structlog` for structured, machine-readable logs
   - Log all strategy executions with timestamps
   - Include context: strategy name, query, execution time, result metadata
   - Support multiple log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
   - JSON format for easy parsing and analysis

2. **Performance Metrics Tracking**
   - **Latency Metrics**: Track execution time for each strategy operation
   - **Token Usage**: Count input/output tokens for LLM calls
   - **Cost Tracking**: Calculate API costs based on token usage and model pricing
   - **Throughput**: Queries per second, documents processed per second
   - **Resource Usage**: Memory, CPU utilization

3. **Error Tracking & Stack Traces**
   - Capture all exceptions with full stack traces
   - Include context: strategy name, query, input parameters
   - Error categorization: API errors, validation errors, system errors
   - Error rate tracking per strategy
   - Automatic error aggregation and deduplication

4. **Query Analytics**
   - Track query patterns and frequencies
   - Response time distribution (p50, p95, p99)
   - Success/failure rates per strategy
   - Query result quality metrics (if available)
   - User session tracking

5. **Log Aggregation & Storage**
   - File-based logging with rotation (daily/size-based)
   - In-memory buffer for recent logs
   - Export to external systems: ELK, Datadog, CloudWatch
   - Configurable retention policies
   - Compression for archived logs

6. **Real-Time Monitoring Dashboard**
   - Web-based dashboard using Flask/FastAPI
   - Live metrics display: QPS, latency, error rate
   - Recent logs view with filtering
   - Strategy performance comparison
   - Cost tracking over time
   - Alerts for anomalies

7. **Observability Integration**
   - OpenTelemetry support for distributed tracing
   - Prometheus metrics export
   - Grafana dashboard templates
   - Integration with existing monitoring tools

### Non-Functional Requirements

1. **Performance**
   - Logging overhead <5ms per operation
   - Dashboard response time <200ms
   - Support 1000+ queries/second with minimal impact
   - Async logging to avoid blocking main thread

2. **Reliability**
   - No log loss under normal conditions
   - Graceful degradation if logging fails
   - Automatic recovery from transient errors
   - Buffer overflow protection

3. **Scalability**
   - Handle logs from multiple strategies concurrently
   - Support horizontal scaling (multiple instances)
   - Efficient log aggregation across distributed systems
   - Configurable log sampling for high-volume scenarios

4. **Security**
   - Sanitize sensitive data (API keys, PII) from logs
   - Secure dashboard access with authentication
   - Encrypted log transmission to external systems
   - Audit log for sensitive operations

5. **Usability**
   - Easy configuration via config files
   - Clear documentation and examples
   - Intuitive dashboard interface
   - Export capabilities (CSV, JSON)

---

## Acceptance Criteria

### AC1: Structured Logging
- [ ] `structlog` integrated and configured
- [ ] All strategy executions logged with timestamps
- [ ] Logs include strategy name, query, execution time
- [ ] Multiple log levels supported (DEBUG to CRITICAL)
- [ ] JSON format output for machine parsing

### AC2: Performance Metrics
- [ ] Latency tracked for all operations
- [ ] Token usage counted for LLM calls
- [ ] Cost calculated based on token usage and pricing
- [ ] Throughput metrics (QPS) tracked
- [ ] Metrics exportable to Prometheus format

### AC3: Error Tracking
- [ ] All exceptions captured with stack traces
- [ ] Error context includes strategy and query info
- [ ] Error categorization implemented
- [ ] Error rate tracked per strategy
- [ ] Automatic error deduplication working

### AC4: Query Analytics
- [ ] Query patterns tracked
- [ ] Response time percentiles calculated (p50, p95, p99)
- [ ] Success/failure rates tracked per strategy
- [ ] Analytics queryable via API

### AC5: Log Management
- [ ] File-based logging with rotation
- [ ] In-memory buffer for recent logs (last 1000)
- [ ] Export to at least one external system (file/ELK)
- [ ] Configurable retention policy
- [ ] Log compression for archives

### AC6: Monitoring Dashboard
- [ ] Web-based dashboard accessible
- [ ] Live metrics displayed (QPS, latency, errors)
- [ ] Recent logs viewable with filtering
- [ ] Strategy comparison view
- [ ] Cost tracking displayed

### AC7: Testing & Quality
- [ ] Unit tests for all logging components (>85% coverage)
- [ ] Integration tests with real strategies
- [ ] Performance tests show <5ms overhead
- [ ] Dashboard load tests pass
- [ ] Security review completed (no PII leaks)

---

## Technical Specifications

### File Structure
```
rag_factory/
├── observability/
│   ├── __init__.py
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── config.py               # Logging configuration
│   │   ├── logger.py               # Main logger implementation
│   │   ├── formatters.py           # Log formatters (JSON, etc.)
│   │   ├── handlers.py             # Custom log handlers
│   │   ├── filters.py              # PII filtering, sampling
│   │   └── context.py              # Context management for logs
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── collector.py            # Metrics collection
│   │   ├── performance.py          # Performance metrics
│   │   ├── cost.py                 # Cost tracking
│   │   ├── aggregator.py           # Metrics aggregation
│   │   └── exporters.py            # Prometheus, StatsD exporters
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── dashboard.py            # Web dashboard (Flask/FastAPI)
│   │   ├── api.py                  # REST API for metrics/logs
│   │   ├── templates/              # HTML templates
│   │   │   ├── index.html
│   │   │   └── metrics.html
│   │   └── static/                 # CSS, JS files
│   │       ├── dashboard.js
│   │       └── styles.css
│   │
│   └── integrations/
│       ├── __init__.py
│       ├── opentelemetry.py        # OpenTelemetry integration
│       ├── prometheus.py           # Prometheus integration
│       └── elk.py                  # ELK stack integration
│
tests/
├── unit/
│   └── observability/
│       ├── test_logger.py
│       ├── test_metrics.py
│       ├── test_formatters.py
│       ├── test_filters.py
│       ├── test_collector.py
│       └── test_exporters.py
│
├── integration/
│   └── observability/
│       ├── test_logging_integration.py
│       ├── test_monitoring_integration.py
│       └── test_dashboard_integration.py
```

### Dependencies
```python
# requirements.txt additions
structlog==24.1.0              # Structured logging
python-json-logger==2.0.7      # JSON log formatting
prometheus-client==0.19.0      # Prometheus metrics
flask==3.0.0                   # Dashboard (or fastapi)
opentelemetry-api==1.22.0      # OpenTelemetry
opentelemetry-sdk==1.22.0
psutil==5.9.6                  # System metrics
```

### Core Logger Implementation
```python
# rag_factory/observability/logging/logger.py
import structlog
import logging
import time
from typing import Dict, Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class LogContext:
    """Context for a logging session."""
    strategy_name: str
    operation: str
    query: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self.start_time) * 1000

class RAGLogger:
    """
    Centralized logger for RAG operations.

    Features:
    - Structured logging with JSON output
    - Context management for operations
    - Performance tracking
    - Error tracking with stack traces
    - PII filtering

    Example:
        logger = RAGLogger()

        with logger.operation("retrieve", strategy="vector_search", query="test"):
            # Your operation here
            pass
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize logger with configuration."""
        self.config = config or {}
        self._setup_structlog()
        self.log = structlog.get_logger()

    def _setup_structlog(self):
        """Configure structlog."""
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                logging.INFO
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

    @contextmanager
    def operation(
        self,
        operation: str,
        strategy: str,
        query: Optional[str] = None,
        **metadata
    ):
        """
        Context manager for logging an operation.

        Args:
            operation: Type of operation (e.g., "retrieve", "rerank")
            strategy: Strategy name
            query: Optional query string
            **metadata: Additional metadata

        Yields:
            LogContext: Context object with timing and metadata
        """
        context = LogContext(
            strategy_name=strategy,
            operation=operation,
            query=query,
            metadata=metadata
        )

        # Log operation start
        self.log.info(
            "operation_start",
            operation=operation,
            strategy=strategy,
            query=self._sanitize_query(query),
            **metadata
        )

        try:
            yield context

            # Log operation success
            self.log.info(
                "operation_complete",
                operation=operation,
                strategy=strategy,
                duration_ms=context.elapsed_ms(),
                **context.metadata
            )

        except Exception as e:
            # Log operation failure
            self.log.error(
                "operation_failed",
                operation=operation,
                strategy=strategy,
                duration_ms=context.elapsed_ms(),
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

    def log_query(
        self,
        query: str,
        strategy: str,
        results_count: int,
        latency_ms: float,
        success: bool = True,
        **metadata
    ):
        """Log a query execution."""
        self.log.info(
            "query_executed",
            query=self._sanitize_query(query),
            strategy=strategy,
            results_count=results_count,
            latency_ms=latency_ms,
            success=success,
            **metadata
        )

    def log_metrics(
        self,
        strategy: str,
        operation: str,
        metrics: Dict[str, Any]
    ):
        """Log performance metrics."""
        self.log.info(
            "metrics",
            strategy=strategy,
            operation=operation,
            **metrics
        )

    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: LogLevel = LogLevel.ERROR
    ):
        """Log an error with context."""
        log_func = getattr(self.log, severity.value.lower())
        log_func(
            "error_occurred",
            error=str(error),
            error_type=type(error).__name__,
            exc_info=True,
            **context
        )

    def _sanitize_query(self, query: Optional[str]) -> Optional[str]:
        """Sanitize query to remove PII."""
        if not query:
            return None

        # Truncate long queries
        max_length = self.config.get("max_query_length", 500)
        if len(query) > max_length:
            return query[:max_length] + "..."

        # Add more PII filtering as needed
        return query
```

### Metrics Collector
```python
# rag_factory/observability/metrics/collector.py
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
from collections import defaultdict, deque
from threading import Lock
import statistics

@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Performance metrics for a strategy."""
    strategy_name: str
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    errors: List[str] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.total_queries == 0:
            return 0.0
        return self.total_latency_ms / self.total_queries

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_queries == 0:
            return 0.0
        return (self.successful_queries / self.total_queries) * 100

    @property
    def p50_latency(self) -> float:
        """50th percentile latency."""
        return self._percentile(50)

    @property
    def p95_latency(self) -> float:
        """95th percentile latency."""
        return self._percentile(95)

    @property
    def p99_latency(self) -> float:
        """99th percentile latency."""
        return self._percentile(99)

    def _percentile(self, p: int) -> float:
        """Calculate percentile from latencies."""
        if not self.latencies:
            return 0.0
        latencies_list = list(self.latencies)
        return statistics.quantiles(latencies_list, n=100)[p-1] if len(latencies_list) > 1 else latencies_list[0]

class MetricsCollector:
    """
    Centralized metrics collection for RAG operations.

    Features:
    - Performance metrics per strategy
    - Cost tracking
    - Query analytics
    - Thread-safe operations
    - Time-series data

    Example:
        collector = MetricsCollector()
        collector.record_query(
            strategy="vector_search",
            latency_ms=45.2,
            tokens=150,
            cost=0.001,
            success=True
        )
    """

    def __init__(self):
        self._metrics: Dict[str, PerformanceMetrics] = defaultdict(
            lambda: PerformanceMetrics(strategy_name="")
        )
        self._lock = Lock()
        self._time_series: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self._start_time = time.time()

    def record_query(
        self,
        strategy: str,
        latency_ms: float,
        tokens: int = 0,
        cost: float = 0.0,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Record a query execution."""
        with self._lock:
            metrics = self._metrics[strategy]
            metrics.strategy_name = strategy
            metrics.total_queries += 1

            if success:
                metrics.successful_queries += 1
            else:
                metrics.failed_queries += 1
                if error:
                    metrics.errors.append(error)

            metrics.total_latency_ms += latency_ms
            metrics.latencies.append(latency_ms)
            metrics.total_tokens += tokens
            metrics.total_cost += cost

            # Record time-series point
            self._time_series[f"{strategy}_latency"].append(
                MetricPoint(
                    timestamp=datetime.now(),
                    value=latency_ms,
                    labels={"strategy": strategy}
                )
            )

    def get_metrics(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics for a strategy or all strategies.

        Args:
            strategy: Strategy name, or None for all strategies

        Returns:
            Dictionary of metrics
        """
        with self._lock:
            if strategy:
                if strategy not in self._metrics:
                    return {}
                return self._format_metrics(self._metrics[strategy])
            else:
                return {
                    name: self._format_metrics(metrics)
                    for name, metrics in self._metrics.items()
                }

    def _format_metrics(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Format metrics for output."""
        return {
            "strategy": metrics.strategy_name,
            "total_queries": metrics.total_queries,
            "successful_queries": metrics.successful_queries,
            "failed_queries": metrics.failed_queries,
            "success_rate": metrics.success_rate,
            "avg_latency_ms": metrics.avg_latency_ms,
            "p50_latency_ms": metrics.p50_latency,
            "p95_latency_ms": metrics.p95_latency,
            "p99_latency_ms": metrics.p99_latency,
            "total_tokens": metrics.total_tokens,
            "total_cost": metrics.total_cost,
            "error_count": len(metrics.errors)
        }

    def get_time_series(
        self,
        metric_name: str,
        duration_minutes: int = 60
    ) -> List[MetricPoint]:
        """Get time-series data for a metric."""
        with self._lock:
            if metric_name not in self._time_series:
                return []

            cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
            return [
                point for point in self._time_series[metric_name]
                if point.timestamp >= cutoff_time
            ]

    def reset_metrics(self, strategy: Optional[str] = None):
        """Reset metrics for a strategy or all strategies."""
        with self._lock:
            if strategy:
                if strategy in self._metrics:
                    del self._metrics[strategy]
            else:
                self._metrics.clear()
                self._time_series.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get overall summary of all metrics."""
        with self._lock:
            total_queries = sum(m.total_queries for m in self._metrics.values())
            total_successful = sum(m.successful_queries for m in self._metrics.values())
            total_failed = sum(m.failed_queries for m in self._metrics.values())
            total_cost = sum(m.total_cost for m in self._metrics.values())
            uptime_seconds = time.time() - self._start_time

            return {
                "uptime_seconds": uptime_seconds,
                "total_queries": total_queries,
                "successful_queries": total_successful,
                "failed_queries": total_failed,
                "overall_success_rate": (total_successful / total_queries * 100) if total_queries > 0 else 0,
                "total_cost": total_cost,
                "queries_per_second": total_queries / uptime_seconds if uptime_seconds > 0 else 0,
                "strategies_count": len(self._metrics)
            }
```

### Dashboard API
```python
# rag_factory/observability/monitoring/api.py
from flask import Flask, jsonify, request, render_template
from typing import Dict, Any
from ..logging.logger import RAGLogger
from ..metrics.collector import MetricsCollector

app = Flask(__name__)

# Global instances (should be injected in production)
logger = RAGLogger()
metrics_collector = MetricsCollector()

@app.route('/')
def index():
    """Dashboard home page."""
    return render_template('index.html')

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get metrics for all or specific strategy."""
    strategy = request.args.get('strategy')
    metrics = metrics_collector.get_metrics(strategy)
    return jsonify(metrics)

@app.route('/api/metrics/summary', methods=['GET'])
def get_summary():
    """Get overall metrics summary."""
    summary = metrics_collector.get_summary()
    return jsonify(summary)

@app.route('/api/metrics/timeseries', methods=['GET'])
def get_timeseries():
    """Get time-series data for a metric."""
    metric_name = request.args.get('metric', 'latency')
    duration = int(request.args.get('duration', 60))

    data = metrics_collector.get_time_series(metric_name, duration)

    return jsonify([
        {
            'timestamp': point.timestamp.isoformat(),
            'value': point.value,
            'labels': point.labels
        }
        for point in data
    ])

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "version": "1.0.0"})

def start_dashboard(host: str = '0.0.0.0', port: int = 8080):
    """Start the monitoring dashboard."""
    app.run(host=host, port=port, debug=False)
```

---

## Unit Tests

### Test File Location
`tests/unit/observability/test_logger.py`
`tests/unit/observability/test_metrics.py`

### Test Cases

#### TC8.1.1: Logger Tests
```python
import pytest
import structlog
from rag_factory.observability.logging.logger import RAGLogger, LogContext, LogLevel

@pytest.fixture
def logger():
    return RAGLogger()

def test_logger_initialization(logger):
    """Test logger initializes correctly."""
    assert logger.log is not None
    assert isinstance(logger.log, structlog.stdlib.BoundLogger) or hasattr(logger.log, 'info')

def test_operation_context_success(logger, caplog):
    """Test successful operation logging."""
    with logger.operation("retrieve", strategy="vector_search", query="test query") as ctx:
        assert ctx.strategy_name == "vector_search"
        assert ctx.operation == "retrieve"
        assert ctx.query == "test query"

        # Simulate work
        import time
        time.sleep(0.01)

    # Check elapsed time
    assert ctx.elapsed_ms() >= 10

def test_operation_context_failure(logger):
    """Test operation logging with exception."""
    with pytest.raises(ValueError):
        with logger.operation("retrieve", strategy="vector_search") as ctx:
            raise ValueError("Test error")

def test_log_query(logger):
    """Test query logging."""
    logger.log_query(
        query="test query",
        strategy="vector_search",
        results_count=5,
        latency_ms=45.2,
        success=True
    )
    # Verify log was created (would check caplog in real implementation)

def test_log_metrics(logger):
    """Test metrics logging."""
    metrics = {
        "latency_ms": 45.2,
        "tokens": 150,
        "cost": 0.001
    }
    logger.log_metrics(
        strategy="vector_search",
        operation="retrieve",
        metrics=metrics
    )

def test_log_error(logger):
    """Test error logging."""
    error = ValueError("Test error")
    context = {"strategy": "vector_search", "query": "test"}

    logger.log_error(error, context, LogLevel.ERROR)

def test_query_sanitization(logger):
    """Test PII sanitization in queries."""
    long_query = "x" * 1000
    sanitized = logger._sanitize_query(long_query)

    assert len(sanitized) <= 503  # 500 + "..."
    assert sanitized.endswith("...")

def test_context_metadata(logger):
    """Test metadata in context."""
    with logger.operation(
        "retrieve",
        strategy="test",
        custom_field="value",
        num_field=123
    ) as ctx:
        ctx.metadata["result_count"] = 5

    assert ctx.metadata["result_count"] == 5
```

#### TC8.1.2: Metrics Collector Tests
```python
import pytest
from rag_factory.observability.metrics.collector import MetricsCollector, PerformanceMetrics

@pytest.fixture
def collector():
    return MetricsCollector()

def test_collector_initialization(collector):
    """Test collector initializes correctly."""
    assert len(collector._metrics) == 0
    assert collector._start_time > 0

def test_record_successful_query(collector):
    """Test recording a successful query."""
    collector.record_query(
        strategy="vector_search",
        latency_ms=45.2,
        tokens=150,
        cost=0.001,
        success=True
    )

    metrics = collector.get_metrics("vector_search")
    assert metrics["total_queries"] == 1
    assert metrics["successful_queries"] == 1
    assert metrics["failed_queries"] == 0
    assert metrics["avg_latency_ms"] == 45.2
    assert metrics["total_tokens"] == 150
    assert metrics["total_cost"] == 0.001

def test_record_failed_query(collector):
    """Test recording a failed query."""
    collector.record_query(
        strategy="vector_search",
        latency_ms=30.0,
        success=False,
        error="Connection timeout"
    )

    metrics = collector.get_metrics("vector_search")
    assert metrics["failed_queries"] == 1
    assert metrics["error_count"] == 1

def test_multiple_queries(collector):
    """Test recording multiple queries."""
    for i in range(10):
        collector.record_query(
            strategy="vector_search",
            latency_ms=40.0 + i,
            tokens=100,
            cost=0.001,
            success=True
        )

    metrics = collector.get_metrics("vector_search")
    assert metrics["total_queries"] == 10
    assert metrics["successful_queries"] == 10
    assert metrics["total_tokens"] == 1000

def test_success_rate_calculation(collector):
    """Test success rate calculation."""
    # 7 successful, 3 failed
    for i in range(10):
        collector.record_query(
            strategy="test",
            latency_ms=50.0,
            success=(i < 7)
        )

    metrics = collector.get_metrics("test")
    assert metrics["success_rate"] == 70.0

def test_latency_percentiles(collector):
    """Test latency percentile calculations."""
    latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for latency in latencies:
        collector.record_query(
            strategy="test",
            latency_ms=latency,
            success=True
        )

    metrics = collector.get_metrics("test")
    assert metrics["p50_latency_ms"] >= 40
    assert metrics["p50_latency_ms"] <= 60
    assert metrics["p95_latency_ms"] >= 90

def test_multiple_strategies(collector):
    """Test metrics for multiple strategies."""
    collector.record_query(strategy="strategy_a", latency_ms=10, success=True)
    collector.record_query(strategy="strategy_b", latency_ms=20, success=True)
    collector.record_query(strategy="strategy_a", latency_ms=15, success=True)

    all_metrics = collector.get_metrics()

    assert len(all_metrics) == 2
    assert "strategy_a" in all_metrics
    assert "strategy_b" in all_metrics
    assert all_metrics["strategy_a"]["total_queries"] == 2
    assert all_metrics["strategy_b"]["total_queries"] == 1

def test_get_summary(collector):
    """Test overall summary."""
    collector.record_query(strategy="a", latency_ms=10, cost=0.001, success=True)
    collector.record_query(strategy="b", latency_ms=20, cost=0.002, success=True)
    collector.record_query(strategy="a", latency_ms=15, cost=0.001, success=False)

    summary = collector.get_summary()

    assert summary["total_queries"] == 3
    assert summary["successful_queries"] == 2
    assert summary["failed_queries"] == 1
    assert summary["total_cost"] == 0.004
    assert summary["strategies_count"] == 2

def test_reset_metrics(collector):
    """Test resetting metrics."""
    collector.record_query(strategy="test", latency_ms=10, success=True)

    collector.reset_metrics("test")
    metrics = collector.get_metrics("test")

    assert metrics == {}

def test_thread_safety(collector):
    """Test thread-safe operations."""
    import threading

    def record_queries():
        for _ in range(100):
            collector.record_query(
                strategy="test",
                latency_ms=50,
                success=True
            )

    threads = [threading.Thread(target=record_queries) for _ in range(10)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    metrics = collector.get_metrics("test")
    assert metrics["total_queries"] == 1000
```

---

## Integration Tests

### Test File Location
`tests/integration/observability/test_monitoring_integration.py`

### Test Scenarios

#### IS8.1.1: End-to-End Logging
```python
import pytest
import time
from rag_factory.observability.logging.logger import RAGLogger
from rag_factory.observability.metrics.collector import MetricsCollector

@pytest.mark.integration
def test_complete_query_logging():
    """Test complete query execution with logging."""
    logger = RAGLogger()
    collector = MetricsCollector()

    # Simulate a query execution
    start_time = time.time()

    with logger.operation("retrieve", strategy="vector_search", query="test query") as ctx:
        # Simulate retrieval
        time.sleep(0.05)  # 50ms
        results_count = 5
        ctx.metadata["results_count"] = results_count

    latency_ms = ctx.elapsed_ms()

    # Record metrics
    collector.record_query(
        strategy="vector_search",
        latency_ms=latency_ms,
        tokens=150,
        cost=0.001,
        success=True
    )

    # Verify metrics
    metrics = collector.get_metrics("vector_search")
    assert metrics["total_queries"] == 1
    assert metrics["avg_latency_ms"] >= 50

@pytest.mark.integration
def test_error_handling_with_logging():
    """Test error handling integrates with logging."""
    logger = RAGLogger()
    collector = MetricsCollector()

    try:
        with logger.operation("retrieve", strategy="test_strategy") as ctx:
            raise ValueError("Simulated error")
    except ValueError:
        pass

    # Record failed query
    collector.record_query(
        strategy="test_strategy",
        latency_ms=ctx.elapsed_ms(),
        success=False,
        error="ValueError: Simulated error"
    )

    metrics = collector.get_metrics("test_strategy")
    assert metrics["failed_queries"] == 1
    assert metrics["error_count"] == 1

@pytest.mark.integration
def test_dashboard_api():
    """Test dashboard API endpoints."""
    from rag_factory.observability.monitoring.api import app

    collector = MetricsCollector()

    # Record some data
    collector.record_query(strategy="test", latency_ms=50, success=True)

    # Test API
    with app.test_client() as client:
        # Test health endpoint
        response = client.get('/api/health')
        assert response.status_code == 200
        assert response.json["status"] == "healthy"

        # Test metrics endpoint
        response = client.get('/api/metrics')
        assert response.status_code == 200

        # Test summary endpoint
        response = client.get('/api/metrics/summary')
        assert response.status_code == 200
        assert "total_queries" in response.json

@pytest.mark.integration
def test_performance_overhead():
    """Test that logging overhead is minimal."""
    logger = RAGLogger()

    iterations = 1000
    start_time = time.time()

    for i in range(iterations):
        with logger.operation("test", strategy="benchmark"):
            pass  # No actual work

    total_time = time.time() - start_time
    avg_overhead = (total_time / iterations) * 1000  # ms

    # Overhead should be < 5ms per operation
    assert avg_overhead < 5.0, f"Logging overhead {avg_overhead:.2f}ms exceeds 5ms"
```

---

## Definition of Done

- [ ] Structured logging with structlog implemented
- [ ] All strategy executions logged with timestamps
- [ ] Performance metrics tracked (latency, tokens, cost)
- [ ] Error tracking with stack traces working
- [ ] Query analytics implemented
- [ ] File-based logging with rotation
- [ ] In-memory log buffer (last 1000)
- [ ] Metrics collector fully implemented
- [ ] Dashboard API endpoints working
- [ ] Web dashboard accessible
- [ ] All unit tests pass (>85% coverage)
- [ ] All integration tests pass
- [ ] Performance tests show <5ms overhead
- [ ] PII filtering implemented
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] No linting errors

---

## Notes for Developers

1. **Start with Logger**: Implement the logger first, then metrics collector

2. **Structured Logs**: Use structlog for all logging - makes parsing easier

3. **Context Managers**: Use context managers for operation tracking - cleaner code

4. **Thread Safety**: MetricsCollector must be thread-safe - use locks

5. **PII Filtering**: Always sanitize queries before logging - privacy is critical

6. **Performance**: Async logging or use a queue to avoid blocking operations

7. **Dashboard**: Start with simple Flask API, can enhance UI later

8. **Prometheus**: Export metrics in Prometheus format for easy Grafana integration

9. **Log Rotation**: Use `logging.handlers.RotatingFileHandler` for file rotation

10. **Testing**: Test performance overhead - logging should not impact production

11. **Configuration**: Make everything configurable - log levels, retention, formats

12. **Error Handling**: Logging failures should never crash the application
