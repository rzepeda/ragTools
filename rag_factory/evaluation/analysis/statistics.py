"""
Statistical analysis for benchmark results.

This module provides statistical tests for comparing strategies,
including t-tests, confidence intervals, and effect size calculations.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class StatisticalTestResult:
    """
    Result from a statistical test.

    Attributes:
        test_name: Name of the statistical test
        statistic: Test statistic value
        p_value: P-value from the test
        significant: Whether result is statistically significant
        confidence_level: Confidence level used (e.g., 0.95)
        effect_size: Effect size measure (e.g., Cohen's d)
        interpretation: Human-readable interpretation
    """
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    confidence_level: float
    effect_size: Optional[float] = None
    interpretation: str = ""


class StatisticalAnalyzer:
    """
    Perform statistical analysis on benchmark results.

    Features:
    - Paired t-tests for metric comparison
    - Confidence intervals
    - Effect size calculation (Cohen's d)
    - Bootstrap resampling
    - Multiple comparison correction

    Example:
        >>> analyzer = StatisticalAnalyzer(confidence_level=0.95)
        >>> result = analyzer.paired_t_test(baseline_scores, new_scores)
        >>> if result.significant:
        ...     print(f"Significant improvement! p={result.p_value:.4f}")
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical analyzer.

        Args:
            confidence_level: Confidence level for tests (default: 0.95 for 95% CI)

        Raises:
            ValueError: If confidence_level is not in (0, 1)
        """
        if not 0 < confidence_level < 1:
            raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def paired_t_test(
        self,
        baseline_scores: List[float],
        comparison_scores: List[float],
        metric_name: str = "metric"
    ) -> StatisticalTestResult:
        """
        Perform paired t-test to compare two sets of scores.

        Tests whether the mean difference between paired samples is
        statistically significant.

        Args:
            baseline_scores: Scores from baseline strategy
            comparison_scores: Scores from comparison strategy
            metric_name: Name of the metric being compared

        Returns:
            StatisticalTestResult with test outcomes

        Raises:
            ValueError: If score lists have different lengths or are empty

        Example:
            >>> result = analyzer.paired_t_test([0.7, 0.8, 0.75], [0.75, 0.85, 0.80])
            >>> print(f"p-value: {result.p_value}")
        """
        if len(baseline_scores) != len(comparison_scores):
            raise ValueError("Score lists must have same length")
        if len(baseline_scores) == 0:
            raise ValueError("Score lists cannot be empty")

        # Perform paired t-test
        statistic, p_value = stats.ttest_rel(comparison_scores, baseline_scores)

        # Calculate effect size (Cohen's d)
        effect_size = self._cohens_d(baseline_scores, comparison_scores)

        # Determine significance
        significant = p_value < self.alpha

        # Generate interpretation
        mean_diff = np.mean(comparison_scores) - np.mean(baseline_scores)
        if significant:
            direction = "higher" if mean_diff > 0 else "lower"
            interpretation = (
                f"The comparison strategy has significantly {direction} "
                f"{metric_name} (p={p_value:.4f}, effect size={effect_size:.3f})"
            )
        else:
            interpretation = (
                f"No significant difference in {metric_name} "
                f"(p={p_value:.4f}, effect size={effect_size:.3f})"
            )

        return StatisticalTestResult(
            test_name="Paired t-test",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            interpretation=interpretation
        )

    def confidence_interval(
        self,
        scores: List[float]
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for mean.

        Args:
            scores: List of metric scores

        Returns:
            Tuple of (lower_bound, upper_bound)

        Example:
            >>> ci_low, ci_high = analyzer.confidence_interval([0.7, 0.75, 0.8])
            >>> print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
        """
        if not scores:
            return (0.0, 0.0)

        mean = np.mean(scores)
        sem = stats.sem(scores)
        margin = sem * stats.t.ppf((1 + self.confidence_level) / 2, len(scores) - 1)

        return (mean - margin, mean + margin)

    def bootstrap_confidence_interval(
        self,
        scores: List[float],
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval.

        Uses bootstrap resampling to estimate confidence interval.
        More robust for non-normal distributions.

        Args:
            scores: List of metric scores
            n_bootstrap: Number of bootstrap samples (default: 1000)

        Returns:
            Tuple of (lower_bound, upper_bound)

        Example:
            >>> ci_low, ci_high = analyzer.bootstrap_confidence_interval(scores)
        """
        if not scores:
            return (0.0, 0.0)

        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(sample))

        lower_percentile = ((1 - self.confidence_level) / 2) * 100
        upper_percentile = (1 - (1 - self.confidence_level) / 2) * 100

        ci_low = np.percentile(bootstrap_means, lower_percentile)
        ci_high = np.percentile(bootstrap_means, upper_percentile)

        return (ci_low, ci_high)

    def _cohens_d(
        self,
        baseline_scores: List[float],
        comparison_scores: List[float]
    ) -> float:
        """
        Calculate Cohen's d effect size.

        Cohen's d measures the standardized difference between two means.
        Rules of thumb:
        - Small effect: d = 0.2
        - Medium effect: d = 0.5
        - Large effect: d = 0.8

        Args:
            baseline_scores: Baseline scores
            comparison_scores: Comparison scores

        Returns:
            Cohen's d value
        """
        mean_diff = np.mean(comparison_scores) - np.mean(baseline_scores)
        pooled_std = np.sqrt(
            (np.var(baseline_scores) + np.var(comparison_scores)) / 2
        )

        if pooled_std == 0:
            return 0.0

        return mean_diff / pooled_std

    def bonferroni_correction(
        self,
        p_values: List[float],
        n_comparisons: Optional[int] = None
    ) -> List[bool]:
        """
        Apply Bonferroni correction for multiple comparisons.

        Adjusts significance threshold to control family-wise error rate.

        Args:
            p_values: List of p-values from multiple tests
            n_comparisons: Number of comparisons (defaults to len(p_values))

        Returns:
            List of boolean values indicating significance after correction

        Example:
            >>> p_values = [0.01, 0.03, 0.05, 0.10]
            >>> significant = analyzer.bonferroni_correction(p_values)
            >>> print(significant)  # [True, False, False, False]
        """
        if not p_values:
            return []

        n_comparisons = n_comparisons or len(p_values)
        adjusted_alpha = self.alpha / n_comparisons

        return [p < adjusted_alpha for p in p_values]

    def interpret_effect_size(self, cohens_d: float) -> str:
        """
        Interpret Cohen's d effect size.

        Args:
            cohens_d: Cohen's d value

        Returns:
            String interpretation
        """
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def analyze_metric(
        self,
        baseline_values: List[float],
        comparison_values: List[float],
        metric_name: str
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of a single metric.

        Args:
            baseline_values: Baseline metric values
            comparison_values: Comparison metric values
            metric_name: Name of the metric

        Returns:
            Dictionary with complete analysis

        Example:
            >>> analysis = analyzer.analyze_metric(
            ...     baseline_values=[0.7, 0.75, 0.72],
            ...     comparison_values=[0.78, 0.82, 0.80],
            ...     metric_name="precision@5"
            ... )
            >>> print(analysis["t_test"].interpretation)
        """
        # T-test
        t_test_result = self.paired_t_test(
            baseline_values,
            comparison_values,
            metric_name
        )

        # Confidence intervals
        baseline_ci = self.confidence_interval(baseline_values)
        comparison_ci = self.confidence_interval(comparison_values)

        # Summary statistics
        baseline_mean = np.mean(baseline_values)
        comparison_mean = np.mean(comparison_values)
        improvement = ((comparison_mean - baseline_mean) / baseline_mean) * 100

        return {
            "metric_name": metric_name,
            "baseline": {
                "mean": baseline_mean,
                "std": np.std(baseline_values),
                "ci": baseline_ci,
                "n": len(baseline_values)
            },
            "comparison": {
                "mean": comparison_mean,
                "std": np.std(comparison_values),
                "ci": comparison_ci,
                "n": len(comparison_values)
            },
            "improvement_pct": improvement,
            "t_test": t_test_result,
            "effect_size_interpretation": self.interpret_effect_size(t_test_result.effect_size)
        }
