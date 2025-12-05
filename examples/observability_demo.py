"""
Demonstration of RAG Factory Observability System

This script demonstrates the key features of the monitoring and logging system.
"""

import time
import random
from rag_factory import RAGLogger, MetricsCollector
from rag_factory.observability.metrics.cost import CostCalculator
from rag_factory.observability.metrics.performance import PerformanceMonitor


def simulate_retrieval_query(query: str, strategy: str = "vector_search"):
    """Simulate a retrieval operation with logging and metrics."""
    logger = RAGLogger()
    collector = MetricsCollector()
    cost_calculator = CostCalculator()

    # Track the operation with logging
    with logger.operation("retrieve", strategy=strategy, query=query) as ctx:
        # Simulate retrieval work
        time.sleep(random.uniform(0.02, 0.1))  # 20-100ms

        # Simulate results
        results_count = random.randint(3, 10)
        tokens = random.randint(100, 300)

        ctx.metadata["results_count"] = results_count
        ctx.metadata["tokens"] = tokens

    # Calculate cost (using GPT-3.5-turbo pricing)
    cost = cost_calculator.calculate_cost(
        model="gpt-3.5-turbo",
        input_tokens=tokens,
        output_tokens=0
    )

    # Record metrics
    collector.record_query(
        strategy=strategy,
        latency_ms=ctx.elapsed_ms(),
        tokens=tokens,
        cost=cost,
        success=True
    )

    return {
        "results_count": results_count,
        "latency_ms": ctx.elapsed_ms(),
        "tokens": tokens,
        "cost": cost
    }


def simulate_failed_query(query: str, strategy: str = "vector_search"):
    """Simulate a failed query."""
    logger = RAGLogger()
    collector = MetricsCollector()

    try:
        with logger.operation("retrieve", strategy=strategy, query=query) as ctx:
            time.sleep(random.uniform(0.01, 0.03))
            raise ValueError("Simulated database connection error")
    except ValueError as e:
        # Log error
        logger.log_error(e, {"strategy": strategy, "query": query})

        # Record failed query metrics
        collector.record_query(
            strategy=strategy,
            latency_ms=ctx.elapsed_ms(),
            success=False,
            error=str(e)
        )


def demo_basic_logging():
    """Demonstrate basic logging features."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Logging")
    print("="*60)

    logger = RAGLogger()

    # Simple log messages
    logger.info("Starting RAG Factory demo", component="demo")
    logger.debug("Debug message example", details="Some debug info")
    logger.warning("This is a warning", severity="low")

    # Operation logging with timing
    print("\n✓ Logged various messages at different levels")


def demo_metrics_collection():
    """Demonstrate metrics collection."""
    print("\n" + "="*60)
    print("DEMO 2: Metrics Collection")
    print("="*60)

    collector = MetricsCollector()

    # Simulate 20 queries across different strategies
    strategies = ["vector_search", "bm25", "hybrid"]

    print(f"\nSimulating 20 queries across {len(strategies)} strategies...")

    for i in range(20):
        strategy = random.choice(strategies)

        # 80% success rate
        if random.random() < 0.8:
            result = simulate_retrieval_query(f"query {i}", strategy)
            print(f"  ✓ Query {i+1}: {strategy} ({result['latency_ms']:.1f}ms)")
        else:
            simulate_failed_query(f"query {i}", strategy)
            print(f"  ✗ Query {i+1}: {strategy} (failed)")

    # Get metrics for each strategy
    print("\n" + "-"*60)
    print("Metrics Summary:")
    print("-"*60)

    for strategy in strategies:
        metrics = collector.get_metrics(strategy)
        if metrics:
            print(f"\n{strategy}:")
            print(f"  Total Queries:    {metrics['total_queries']}")
            print(f"  Success Rate:     {metrics['success_rate']:.1f}%")
            print(f"  Avg Latency:      {metrics['avg_latency_ms']:.2f}ms")
            print(f"  P95 Latency:      {metrics['p95_latency_ms']:.2f}ms")
            print(f"  Total Cost:       ${metrics['total_cost']:.6f}")
            print(f"  Errors:           {metrics['error_count']}")

    # Overall summary
    summary = collector.get_summary()
    print("\n" + "-"*60)
    print("Overall Summary:")
    print("-"*60)
    print(f"Total Queries:      {summary['total_queries']}")
    print(f"Success Rate:       {summary['overall_success_rate']:.1f}%")
    print(f"Total Cost:         ${summary['total_cost']:.6f}")
    print(f"Queries/Second:     {summary['queries_per_second']:.2f}")
    print(f"Avg Latency:        {summary['avg_latency_ms']:.2f}ms")
    print(f"P95 Latency:        {summary['p95_latency_ms']:.2f}ms")


def demo_cost_calculation():
    """Demonstrate cost calculation."""
    print("\n" + "="*60)
    print("DEMO 3: Cost Calculation")
    print("="*60)

    calculator = CostCalculator()

    # Calculate costs for different models
    print("\nCost examples:")

    # GPT-4
    cost = calculator.calculate_cost("gpt-4", input_tokens=1000, output_tokens=500)
    print(f"\nGPT-4 (1K input, 500 output):     ${cost:.6f}")

    # GPT-3.5-turbo
    cost = calculator.calculate_cost("gpt-3.5-turbo", input_tokens=1000, output_tokens=500)
    print(f"GPT-3.5-turbo (1K input, 500 out): ${cost:.6f}")

    # Embeddings
    cost = calculator.calculate_embedding_cost("text-embedding-3-small", tokens=5000)
    print(f"Embedding (5K tokens):             ${cost:.6f}")

    # Claude
    cost = calculator.calculate_cost("claude-3-haiku", input_tokens=1000, output_tokens=500)
    print(f"Claude-3-haiku (1K input, 500 out): ${cost:.6f}")

    # List available models
    print(f"\nTotal models supported: {len(calculator.list_available_models())}")


def demo_performance_monitoring():
    """Demonstrate performance monitoring."""
    print("\n" + "="*60)
    print("DEMO 4: Performance Monitoring")
    print("="*60)

    monitor = PerformanceMonitor()

    # Track some operations
    operations = ["embedding_generation", "vector_search", "reranking"]

    print("\nTracking system performance during operations...")

    for operation in operations:
        for _ in range(3):
            with monitor.track(operation):
                time.sleep(random.uniform(0.01, 0.05))

    # Get statistics
    print("\n" + "-"*60)
    print("Performance Statistics:")
    print("-"*60)

    for operation in operations:
        stats = monitor.get_stats(operation)
        print(f"\n{operation}:")
        print(f"  Executions:      {stats['executions']}")
        print(f"  Avg Duration:    {stats['avg_duration_ms']:.2f}ms")
        print(f"  Min Duration:    {stats['min_duration_ms']:.2f}ms")
        print(f"  Max Duration:    {stats['max_duration_ms']:.2f}ms")
        print(f"  Avg CPU:         {stats['avg_cpu_percent']:.1f}%")
        print(f"  Avg Memory:      {stats['avg_memory_percent']:.1f}%")

    # Current system stats
    sys_stats = monitor.get_current_system_stats()
    print("\n" + "-"*60)
    print("Current System Stats:")
    print("-"*60)
    print(f"CPU:           {sys_stats['cpu_percent']:.1f}%")
    print(f"Memory:        {sys_stats['memory_percent']:.1f}%")
    print(f"Memory Used:   {sys_stats['memory_used_mb']:.0f} MB")
    print(f"Disk Usage:    {sys_stats['disk_percent']:.1f}%")


def demo_pii_filtering():
    """Demonstrate PII filtering."""
    print("\n" + "="*60)
    print("DEMO 5: PII Filtering")
    print("="*60)

    logger = RAGLogger()

    # Test queries with PII
    test_queries = [
        "Contact john.doe@example.com for details",
        "Call me at 555-123-4567",
        "My SSN is 123-45-6789",
    ]

    print("\nOriginal queries → Sanitized queries:")
    print("-"*60)

    for query in test_queries:
        sanitized = logger._sanitize_query(query)
        print(f"{query}")
        print(f"  → {sanitized}\n")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("RAG FACTORY OBSERVABILITY SYSTEM DEMO")
    print("="*60)

    # Run demonstrations
    demo_basic_logging()
    demo_metrics_collection()
    demo_cost_calculation()
    demo_performance_monitoring()
    demo_pii_filtering()

    # Final message
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nTo view the monitoring dashboard, run:")
    print("  from rag_factory.observability.monitoring.api import start_dashboard")
    print("  start_dashboard(host='localhost', port=8080)")
    print("\nThen open http://localhost:8080 in your browser")
    print("\n")


if __name__ == "__main__":
    main()
