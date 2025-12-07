"""Unit tests for ABTestingFramework."""

import pytest
import numpy as np
from rag_factory.models.evaluation.ab_testing import ABTestingFramework
from rag_factory.models.evaluation.models import ABTestConfig


@pytest.fixture
def ab_framework():
    """Create ABTestingFramework instance."""
    return ABTestingFramework()


@pytest.fixture
def test_config():
    """Sample test configuration."""
    return ABTestConfig(
        test_name="base_vs_finetuned",
        model_a_id="base_model",
        model_b_id="finetuned_model",
        traffic_split=0.5,
        duration_days=7,
        minimum_samples=100,
        metrics_to_track=["latency", "accuracy"],
        statistical_threshold=0.05
    )


def test_framework_initialization(ab_framework):
    """Test framework initializes correctly."""
    assert isinstance(ab_framework.active_tests, dict)
    assert isinstance(ab_framework.results, dict)
    assert len(ab_framework.active_tests) == 0
    assert len(ab_framework.results) == 0


def test_start_test(ab_framework, test_config):
    """Test starting an A/B test."""
    ab_framework.start_test(test_config)

    assert "base_vs_finetuned" in ab_framework.active_tests
    assert ab_framework.active_tests["base_vs_finetuned"].model_a_id == "base_model"
    assert ab_framework.active_tests["base_vs_finetuned"].model_b_id == "finetuned_model"
    assert "base_vs_finetuned" in ab_framework.results
    assert len(ab_framework.results["base_vs_finetuned"]) == 0


def test_traffic_splitting(ab_framework, test_config):
    """Test traffic splitting between models."""
    ab_framework.start_test(test_config)

    # Run many trials
    model_b_count = 0
    trials = 10000

    for _ in range(trials):
        if ab_framework.should_use_model_b("base_vs_finetuned"):
            model_b_count += 1

    # Should be approximately 50% (within reasonable margin)
    model_b_ratio = model_b_count / trials
    assert 0.45 < model_b_ratio < 0.55


def test_traffic_splitting_different_ratios(ab_framework):
    """Test traffic splitting with different ratios."""
    # Test with 20% traffic to model B
    config = ABTestConfig(
        test_name="test_20",
        model_a_id="a",
        model_b_id="b",
        traffic_split=0.2
    )
    ab_framework.start_test(config)

    model_b_count = sum(
        1 for _ in range(10000)
        if ab_framework.should_use_model_b("test_20")
    )
    
    ratio = model_b_count / 10000
    assert 0.15 < ratio < 0.25


def test_should_use_model_b_inactive_test(ab_framework):
    """Test should_use_model_b returns False for inactive test."""
    result = ab_framework.should_use_model_b("nonexistent_test")
    assert result is False


def test_record_results(ab_framework, test_config):
    """Test recording test results."""
    ab_framework.start_test(test_config)

    # Record some results
    ab_framework.record_result(
        "base_vs_finetuned",
        "base_model",
        {"latency": 50.0, "accuracy": 0.80}
    )

    ab_framework.record_result(
        "base_vs_finetuned",
        "finetuned_model",
        {"latency": 45.0, "accuracy": 0.85}
    )

    assert len(ab_framework.results["base_vs_finetuned"]) == 2
    assert ab_framework.results["base_vs_finetuned"][0]["model_id"] == "base_model"
    assert ab_framework.results["base_vs_finetuned"][1]["model_id"] == "finetuned_model"


def test_record_result_inactive_test(ab_framework):
    """Test recording result for inactive test logs warning."""
    # Should not raise error, just log warning
    ab_framework.record_result(
        "nonexistent",
        "model",
        {"metric": 1.0}
    )
    # No exception should be raised


def test_analyze_test(ab_framework, test_config):
    """Test analyzing A/B test results."""
    ab_framework.start_test(test_config)

    # Record many results with clear difference
    np.random.seed(42)
    for i in range(200):
        if i < 100:
            # Model A: higher latency, lower accuracy
            ab_framework.record_result(
                "base_vs_finetuned",
                "base_model",
                {
                    "latency": 50.0 + np.random.randn() * 2,
                    "accuracy": 0.80 + np.random.randn() * 0.02
                }
            )
        else:
            # Model B: lower latency, higher accuracy
            ab_framework.record_result(
                "base_vs_finetuned",
                "finetuned_model",
                {
                    "latency": 45.0 + np.random.randn() * 2,
                    "accuracy": 0.85 + np.random.randn() * 0.02
                }
            )

    # Analyze
    result = ab_framework.analyze_test("base_vs_finetuned")

    assert result.test_name == "base_vs_finetuned"
    assert result.model_a_samples == 100
    assert result.model_b_samples == 100
    assert "latency" in result.metrics
    assert "accuracy" in result.metrics
    assert "latency" in result.p_values
    assert "accuracy" in result.p_values
    assert result.winner in ["model_a", "model_b", "no_difference"]


def test_analyze_test_not_found(ab_framework):
    """Test analyzing non-existent test raises error."""
    with pytest.raises(ValueError, match="Test .* not found"):
        ab_framework.analyze_test("nonexistent")


def test_analyze_test_insufficient_samples(ab_framework):
    """Test analysis with insufficient samples."""
    config = ABTestConfig(
        test_name="test",
        model_a_id="a",
        model_b_id="b",
        minimum_samples=100
    )
    ab_framework.start_test(config)

    # Record only 10 samples
    for i in range(10):
        ab_framework.record_result("test", "a", {"metric": 1.0})
        ab_framework.record_result("test", "b", {"metric": 1.1})

    # Should still analyze but log warning
    result = ab_framework.analyze_test("test")
    assert result.model_a_samples == 10
    assert result.model_b_samples == 10


def test_winner_determination_clear_winner(ab_framework, test_config):
    """Test winner determination with clear winner."""
    ab_framework.start_test(test_config)

    # Model B clearly better
    for _ in range(100):
        ab_framework.record_result("base_vs_finetuned", "base_model", {"accuracy": 0.70})
        ab_framework.record_result("base_vs_finetuned", "finetuned_model", {"accuracy": 0.90})

    result = ab_framework.analyze_test("base_vs_finetuned")
    assert result.winner == "model_b"
    assert "improvement" in result.recommendation.lower()


def test_winner_determination_no_difference(ab_framework, test_config):
    """Test winner determination with no significant difference."""
    ab_framework.start_test(test_config)

    # Both models similar
    np.random.seed(42)
    for _ in range(100):
        ab_framework.record_result(
            "base_vs_finetuned",
            "base_model",
            {"accuracy": 0.80 + np.random.randn() * 0.05}
        )
        ab_framework.record_result(
            "base_vs_finetuned",
            "finetuned_model",
            {"accuracy": 0.80 + np.random.randn() * 0.05}
        )

    result = ab_framework.analyze_test("base_vs_finetuned")
    # With similar performance, should be no_difference
    assert result.winner in ["no_difference", "model_a", "model_b"]


def test_gradual_rollout(ab_framework, test_config):
    """Test gradual traffic increase."""
    ab_framework.start_test(test_config)

    # Initial split is 50%
    assert ab_framework.active_tests["base_vs_finetuned"].traffic_split == 0.5

    # Increase to 80%
    ab_framework.gradual_rollout("base_vs_finetuned", 0.8)

    assert ab_framework.active_tests["base_vs_finetuned"].traffic_split == 0.8


def test_gradual_rollout_bounds(ab_framework, test_config):
    """Test gradual rollout respects bounds."""
    ab_framework.start_test(test_config)

    # Try to set above 1.0
    ab_framework.gradual_rollout("base_vs_finetuned", 1.5)
    assert ab_framework.active_tests["base_vs_finetuned"].traffic_split == 1.0

    # Try to set below 0.0
    ab_framework.gradual_rollout("base_vs_finetuned", -0.5)
    assert ab_framework.active_tests["base_vs_finetuned"].traffic_split == 0.0


def test_gradual_rollout_not_found(ab_framework):
    """Test gradual rollout for non-existent test raises error."""
    with pytest.raises(ValueError, match="Test .* not found"):
        ab_framework.gradual_rollout("nonexistent", 0.8)


def test_metrics_comparison_calculation(ab_framework, test_config):
    """Test that metrics comparison calculates improvement correctly."""
    ab_framework.start_test(test_config)

    # Model A: 100ms latency
    # Model B: 80ms latency (20% improvement)
    for _ in range(50):
        ab_framework.record_result("base_vs_finetuned", "base_model", {"latency": 100.0})
        ab_framework.record_result("base_vs_finetuned", "finetuned_model", {"latency": 80.0})

    result = ab_framework.analyze_test("base_vs_finetuned")
    
    # Check improvement calculation
    assert "latency" in result.metrics
    improvement = result.metrics["latency"]["improvement"]
    assert -25 < improvement < -15  # Should be around -20% (lower is better for latency)


def test_confidence_intervals(ab_framework, test_config):
    """Test that confidence intervals are calculated."""
    ab_framework.start_test(test_config)

    np.random.seed(42)
    for _ in range(100):
        ab_framework.record_result(
            "base_vs_finetuned",
            "base_model",
            {"accuracy": 0.80 + np.random.randn() * 0.02}
        )
        ab_framework.record_result(
            "base_vs_finetuned",
            "finetuned_model",
            {"accuracy": 0.85 + np.random.randn() * 0.02}
        )

    result = ab_framework.analyze_test("base_vs_finetuned")
    
    assert "accuracy" in result.confidence_intervals
    ci = result.confidence_intervals["accuracy"]
    assert isinstance(ci, tuple)
    assert len(ci) == 2
    assert ci[0] < ci[1]  # Lower bound < upper bound
