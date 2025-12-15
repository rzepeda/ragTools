# Story 15.6: Improve Evaluation Framework Tests

**Story ID:** 15.6  
**Epic:** Epic 15 - Test Coverage Improvements  
**Story Points:** 5  
**Priority:** Low  
**Dependencies:** Epic 9 (Evaluation Framework)

---

## User Story

**As a** developer  
**I want** comprehensive tests for the evaluation framework  
**So that** I can ensure benchmarking and metrics work correctly

---

## Detailed Requirements

### Functional Requirements

> [!NOTE]
> **Current Evaluation Framework Coverage**: 33% (2 out of 6 features tested)
> **Target Coverage**: 60%+
> **Existing Tests**: `test_retrieval_metrics.py`, `test_dataset_loader.py` (partial coverage)
> **Missing**: Benchmarking suite, visualization, statistical testing, export functionality

1. **Benchmarking Suite Tests**
   - Test benchmark execution with datasets
   - Test metric collection (accuracy, latency, cost)
   - Test result aggregation
   - Test comparison across strategies

2. **Results Visualization Tests**
   - Test chart generation (bar, line, scatter)
   - Test table formatting
   - Test export to PNG/SVG
   - Test visualization with missing data

3. **Statistical Testing Tests**
   - Test significance testing (t-test, Mann-Whitney)
   - Test confidence interval calculation
   - Test p-value computation
   - Test result interpretation

4. **Export Functionality Tests**
   - Test CSV export
   - Test JSON export
   - Test export with custom fields
   - Test export error handling

---

## Acceptance Criteria

### AC1: Benchmarking Suite Tests
- [ ] Test file `tests/unit/evaluation/test_benchmark_suite.py` created
- [ ] Test benchmark execution with sample dataset
- [ ] Test metric collection for multiple strategies
- [ ] Test result aggregation across runs
- [ ] Test strategy comparison
- [ ] Test error handling for failed benchmarks
- [ ] Minimum 10 test cases

### AC2: Visualization Tests
- [ ] Test file `tests/unit/evaluation/test_visualization.py` created
- [ ] Test bar chart generation
- [ ] Test line chart generation
- [ ] Test scatter plot generation
- [ ] Test table formatting
- [ ] Test export to PNG
- [ ] Test export to SVG
- [ ] Test visualization with missing data
- [ ] Minimum 10 test cases

### AC3: Statistical Testing Tests
- [ ] Test file `tests/unit/evaluation/test_statistical_testing.py` created
- [ ] Test t-test for strategy comparison
- [ ] Test Mann-Whitney U test
- [ ] Test confidence interval calculation
- [ ] Test p-value computation
- [ ] Test result interpretation (significant/not significant)
- [ ] Test with edge cases (equal means, high variance)
- [ ] Minimum 10 test cases

### AC4: Export Tests
- [ ] Test file `tests/unit/evaluation/test_export.py` created
- [ ] Test CSV export with all fields
- [ ] Test JSON export with nested data
- [ ] Test export with custom field selection
- [ ] Test export error handling (permissions, disk space)
- [ ] Test export file format validation
- [ ] Minimum 8 test cases

### AC5: Integration Tests
- [ ] Test file `tests/integration/evaluation/test_evaluation_integration.py` created
- [ ] Test full benchmark workflow (run → visualize → export)
- [ ] Test comparison of real strategies
- [ ] Test statistical analysis on real results
- [ ] Minimum 6 test cases

### AC6: Test Quality
- [ ] All tests pass (100% success rate)
- [ ] Evaluation framework coverage increases from 33% to 60%+
- [ ] Tests use proper mocking for expensive operations
- [ ] Type hints validated
- [ ] Linting passes

---

## Technical Specifications

### File Structure

```
tests/unit/evaluation/
├── test_dataset_loader.py          # Existing - ✅ Partial coverage
├── test_retrieval_metrics.py       # Existing - ✅ Partial coverage
├── test_benchmark_suite.py         # NEW - No existing tests
├── test_visualization.py           # NEW - No existing tests
├── test_statistical_testing.py     # NEW - No existing tests
└── test_export.py                  # NEW - No existing tests

tests/integration/evaluation/
└── test_evaluation_integration.py  # NEW - No existing integration tests

# Existing benchmark files (for reference):
tests/benchmarks/
├── test_contextual_performance.py
├── test_late_chunking_performance.py
└── test_model_comparison_performance.py
```

> [!TIP]
> The existing benchmark files in `tests/benchmarks/` can provide patterns for performance testing, but they don't test the benchmarking suite itself.

### Benchmark Suite Test Template

```python
"""Unit tests for benchmark suite."""
import pytest
from unittest.mock import Mock, patch
from rag_factory.evaluation.benchmark_suite import BenchmarkSuite

class TestBenchmarkSuite:
    """Test suite for benchmarking functionality."""
    
    def test_run_benchmark_with_dataset(self):
        """Test running benchmark with sample dataset."""
        suite = BenchmarkSuite()
        
        # Mock strategies
        strategy1 = Mock()
        strategy1.retrieve.return_value = ["result1", "result2"]
        
        dataset = [
            {"query": "What is RAG?", "expected": ["doc1"]},
            {"query": "How does it work?", "expected": ["doc2"]}
        ]
        
        results = suite.run_benchmark(
            strategies=[strategy1],
            dataset=dataset,
            metrics=["accuracy", "latency"]
        )
        
        assert len(results) == 1
        assert "accuracy" in results[0]
        assert "latency" in results[0]
    
    def test_compare_strategies(self):
        """Test comparing multiple strategies."""
        suite = BenchmarkSuite()
        
        strategy1 = Mock()
        strategy2 = Mock()
        
        dataset = [{"query": "test", "expected": ["doc1"]}]
        
        results = suite.run_benchmark(
            strategies=[strategy1, strategy2],
            dataset=dataset
        )
        
        comparison = suite.compare(results)
        
        assert len(comparison) == 2
        assert "strategy_name" in comparison[0]
        assert "metrics" in comparison[0]
```

### Visualization Test Template

```python
"""Unit tests for visualization."""
import pytest
from rag_factory.evaluation.visualization import Visualizer

class TestVisualization:
    """Test suite for result visualization."""
    
    def test_generate_bar_chart(self, tmp_path):
        """Test bar chart generation."""
        viz = Visualizer()
        
        data = {
            "strategy1": {"accuracy": 0.85, "latency": 1.2},
            "strategy2": {"accuracy": 0.90, "latency": 1.5}
        }
        
        output_path = tmp_path / "chart.png"
        viz.bar_chart(data, metric="accuracy", output=str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_visualization_with_missing_data(self, tmp_path):
        """Test visualization handles missing data gracefully."""
        viz = Visualizer()
        
        data = {
            "strategy1": {"accuracy": 0.85},
            "strategy2": {"latency": 1.5}  # Missing accuracy
        }
        
        output_path = tmp_path / "chart.png"
        
        # Should not raise error
        viz.bar_chart(data, metric="accuracy", output=str(output_path))
        assert output_path.exists()
```

### Statistical Testing Test Template

```python
"""Unit tests for statistical testing."""
import pytest
import numpy as np
from rag_factory.evaluation.statistics import StatisticalTester

class TestStatisticalTesting:
    """Test suite for statistical analysis."""
    
    def test_t_test_significant_difference(self):
        """Test t-test detects significant difference."""
        tester = StatisticalTester()
        
        # Two clearly different distributions
        results1 = [0.8, 0.82, 0.81, 0.83, 0.79]
        results2 = [0.6, 0.62, 0.61, 0.63, 0.59]
        
        p_value, is_significant = tester.t_test(results1, results2, alpha=0.05)
        
        assert is_significant is True
        assert p_value < 0.05
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        tester = StatisticalTester()
        
        data = [0.8, 0.82, 0.81, 0.83, 0.79]
        
        ci_lower, ci_upper = tester.confidence_interval(data, confidence=0.95)
        
        mean = np.mean(data)
        assert ci_lower < mean < ci_upper
        assert ci_upper - ci_lower > 0  # Non-zero width
```

### Testing Strategy

1. **Unit Tests**
   - Mock expensive operations (actual benchmarking)
   - Test logic in isolation
   - Use small sample datasets

2. **Integration Tests**
   - Run real benchmarks with small datasets
   - Test full workflow
   - Verify file outputs

3. **Visualization Tests**
   - Use temporary directories for output files
   - Verify file creation and size
   - Don't validate exact pixel content (too fragile)

---

## Definition of Done

- [ ] All 4 new unit test files created
- [ ] All 1 new integration test file created
- [ ] All tests pass (100% success rate)
- [ ] Evaluation framework coverage reaches 60%+ (from 33%)
- [ ] Tests use temporary directories for file outputs
- [ ] Type checking passes
- [ ] Linting passes
- [ ] PR merged

---

## Notes

- Current evaluation framework coverage is 33% (2 out of 6 features tested)
- Visualization tests should use temporary directories
- Statistical tests should use known distributions for validation
- This is low priority compared to CLI and database tests
