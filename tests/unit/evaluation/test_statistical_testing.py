"""
Unit tests for statistical testing.

Tests the StatisticalAnalyzer class for statistical analysis of benchmark results.
"""

import pytest
import numpy as np
from typing import List
from rag_factory.evaluation.analysis.statistics import StatisticalAnalyzer, StatisticalTestResult


class TestStatisticalAnalyzer:
    """Tests for StatisticalAnalyzer class."""

    def test_init_default_confidence(self) -> None:
        """Test initialization with default confidence level."""
        analyzer = StatisticalAnalyzer()
        assert analyzer.confidence_level == 0.95

    def test_init_custom_confidence(self) -> None:
        """Test initialization with custom confidence level."""
        analyzer = StatisticalAnalyzer(confidence_level=0.99)
        assert analyzer.confidence_level == 0.99

    def test_init_invalid_confidence(self) -> None:
        """Test initialization fails with invalid confidence level."""
        with pytest.raises(ValueError):
            StatisticalAnalyzer(confidence_level=1.5)
        
        with pytest.raises(ValueError):
            StatisticalAnalyzer(confidence_level=0.0)

    def test_paired_t_test_significant_difference(self) -> None:
        """Test t-test detects significant difference."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        # Two clearly different distributions
        baseline_scores = [0.6, 0.62, 0.61, 0.63, 0.59, 0.60, 0.61]
        comparison_scores = [0.8, 0.82, 0.81, 0.83, 0.79, 0.80, 0.81]
        
        result = analyzer.paired_t_test(
            baseline_scores=baseline_scores,
            comparison_scores=comparison_scores,
            metric_name="precision"
        )
        
        assert isinstance(result, StatisticalTestResult)
        assert result.test_name == "Paired t-test"
        assert bool(result.significant) is True
        assert result.p_value < 0.05
        assert result.effect_size is not None

    def test_paired_t_test_no_difference(self) -> None:
        """Test t-test does not detect difference when none exists."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        # Two similar distributions
        baseline_scores = [0.8, 0.82, 0.81, 0.83, 0.79]
        comparison_scores = [0.81, 0.83, 0.80, 0.82, 0.80]
        
        result = analyzer.paired_t_test(
            baseline_scores=baseline_scores,
            comparison_scores=comparison_scores,
            metric_name="precision"
        )
        
        assert bool(result.significant) is False
        assert result.p_value >= 0.05

    def test_confidence_interval_contains_mean(self) -> None:
        """Test that confidence interval contains the mean."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        scores = [0.7, 0.75, 0.72, 0.78, 0.73, 0.76, 0.74]
        
        ci_lower, ci_upper = analyzer.confidence_interval(scores)
        
        mean = np.mean(scores)
        assert ci_lower < mean < ci_upper

    def test_confidence_interval_width(self) -> None:
        """Test confidence interval has non-zero width."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        scores = [0.7, 0.75, 0.72, 0.78, 0.73]
        
        ci_lower, ci_upper = analyzer.confidence_interval(scores)
        
        assert ci_upper > ci_lower
        assert ci_upper - ci_lower > 0

    def test_confidence_interval_single_value(self) -> None:
        """Test confidence interval with single value."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        scores = [0.75]
        
        ci_lower, ci_upper = analyzer.confidence_interval(scores)
        
        # With single value, CI will be NaN (not enough data for variance)
        import math
        assert math.isnan(ci_lower)
        assert math.isnan(ci_upper)

    def test_bootstrap_confidence_interval(self) -> None:
        """Test bootstrap confidence interval calculation."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        scores = [0.7, 0.75, 0.72, 0.78, 0.73, 0.76, 0.74, 0.71, 0.77]
        
        ci_lower, ci_upper = analyzer.bootstrap_confidence_interval(
            scores=scores,
            n_bootstrap=1000
        )
        
        mean = np.mean(scores)
        assert ci_lower < mean < ci_upper
        assert ci_upper > ci_lower

    def test_cohens_d_calculation(self) -> None:
        """Test Cohen's d effect size calculation."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        baseline_scores = [0.6, 0.62, 0.61, 0.63, 0.59]
        comparison_scores = [0.8, 0.82, 0.81, 0.83, 0.79]
        
        cohens_d = analyzer._cohens_d(baseline_scores, comparison_scores)
        
        # Should be a large effect size (> 0.8)
        assert cohens_d > 0.8
        assert isinstance(cohens_d, float)

    def test_cohens_d_zero_difference(self) -> None:
        """Test Cohen's d with identical distributions."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        scores = [0.8, 0.82, 0.81, 0.83, 0.79]
        
        cohens_d = analyzer._cohens_d(scores, scores)
        
        # Should be zero or very close to zero
        assert abs(cohens_d) < 0.01

    def test_cohens_d_interpretation_small(self) -> None:
        """Test interpretation of small effect size."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        interpretation = analyzer.interpret_effect_size(0.3)
        
        assert "small" in interpretation.lower()

    def test_cohens_d_interpretation_medium(self) -> None:
        """Test interpretation of medium effect size."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        interpretation = analyzer.interpret_effect_size(0.6)
        
        assert "medium" in interpretation.lower()

    def test_cohens_d_interpretation_large(self) -> None:
        """Test interpretation of large effect size."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        interpretation = analyzer.interpret_effect_size(1.0)
        
        assert "large" in interpretation.lower()

    def test_bonferroni_correction(self) -> None:
        """Test Bonferroni correction for multiple comparisons."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        p_values = [0.01, 0.03, 0.04, 0.06]
        
        significant = analyzer.bonferroni_correction(p_values)
        
        # Only first p-value should be significant (0.01 < 0.05/4 = 0.0125)
        assert bool(significant[0]) is True
        assert bool(significant[1]) is False
        assert bool(significant[2]) is False
        assert bool(significant[3]) is False

    def test_bonferroni_correction_custom_comparisons(self) -> None:
        """Test Bonferroni correction with custom number of comparisons."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        p_values = [0.01, 0.03]
        
        # Use 10 comparisons instead of 2
        significant = analyzer.bonferroni_correction(
            p_values,
            n_comparisons=10
        )
        
        # Both p-values are not significant with stricter threshold (0.05/10 = 0.005)
        assert bool(significant[0]) is False
        assert bool(significant[1]) is False

    def test_analyze_metric_comprehensive(self) -> None:
        """Test comprehensive metric analysis."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        baseline_values = [0.6, 0.62, 0.61, 0.63, 0.59, 0.60]
        comparison_values = [0.8, 0.82, 0.81, 0.83, 0.79, 0.80]
        
        analysis = analyzer.analyze_metric(
            baseline_values=baseline_values,
            comparison_values=comparison_values,
            metric_name="precision"
        )
        
        # Check that all components are present
        assert "t_test" in analysis
        assert "baseline" in analysis
        assert "comparison" in analysis
        assert "effect_size_interpretation" in analysis
        
        # Check t-test result
        assert isinstance(analysis["t_test"], StatisticalTestResult)
        assert bool(analysis["t_test"].significant) is True
        
        # Check confidence intervals (nested in baseline/comparison)
        assert "ci" in analysis["baseline"]
        assert "ci" in analysis["comparison"]
        assert len(analysis["baseline"]["ci"]) == 2
        assert len(analysis["comparison"]["ci"]) == 2

    def test_statistical_test_result_structure(self) -> None:
        """Test StatisticalTestResult structure."""
        result = StatisticalTestResult(
            test_name="t_test",
            statistic=2.5,
            p_value=0.02,
            significant=True,
            confidence_level=0.95,
            effect_size=0.8,
            interpretation="Significant difference detected"
        )
        
        assert result.test_name == "t_test"
        assert result.statistic == 2.5
        assert result.p_value == 0.02
        assert result.significant is True
        assert result.confidence_level == 0.95
        assert result.effect_size == 0.8
        assert "Significant" in result.interpretation

    def test_edge_case_equal_distributions(self) -> None:
        """Test statistical analysis with equal distributions."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        scores = [0.8, 0.8, 0.8, 0.8, 0.8]
        
        # T-test with identical distributions
        import math
        result = analyzer.paired_t_test(scores, scores, "metric")
        # With identical values, p-value will be NaN
        assert math.isnan(result.p_value) or result.p_value >= 0.05
        
        # Confidence interval with zero variance will be NaN
        ci_lower, ci_upper = analyzer.confidence_interval(scores)
        assert math.isnan(ci_lower) or ci_upper - ci_lower < 0.01

    def test_edge_case_high_variance(self) -> None:
        """Test statistical analysis with high variance."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        # High variance distribution
        baseline_scores = [0.1, 0.5, 0.9, 0.2, 0.8]
        comparison_scores = [0.15, 0.55, 0.95, 0.25, 0.85]
        
        result = analyzer.paired_t_test(
            baseline_scores=baseline_scores,
            comparison_scores=comparison_scores,
            metric_name="precision"
        )
        
        # With high variance, small differences may not be significant
        assert isinstance(result, StatisticalTestResult)
        assert result.p_value is not None

    def test_edge_case_small_sample(self) -> None:
        """Test statistical analysis with small sample size."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        # Very small sample
        baseline_scores = [0.6, 0.7]
        comparison_scores = [0.8, 0.9]
        
        result = analyzer.paired_t_test(
            baseline_scores=baseline_scores,
            comparison_scores=comparison_scores,
            metric_name="precision"
        )
        
        # Should still produce valid result
        assert isinstance(result, StatisticalTestResult)
        assert result.p_value is not None
        assert result.effect_size is not None
