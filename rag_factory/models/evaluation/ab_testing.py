"""A/B testing framework for embedding models."""

from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

from rag_factory.models.evaluation.models import ABTestConfig, ABTestResult
from datetime import datetime

logger = logging.getLogger(__name__)


class ABTestingFramework:
    """Framework for A/B testing embedding models.
    
    Enables traffic splitting, result tracking, and statistical analysis
    to compare embedding model performance.
    
    Attributes:
        active_tests: Dictionary of active test configurations
        results: Dictionary of test results by test name
    """

    def __init__(self):
        """Initialize A/B testing framework."""
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.results: Dict[str, List[Dict[str, Any]]] = {}

    def start_test(self, config: ABTestConfig) -> None:
        """Start an A/B test.
        
        Args:
            config: Test configuration
        """
        logger.info(f"Starting A/B test: {config.test_name}")

        self.active_tests[config.test_name] = config
        self.results[config.test_name] = []

    def should_use_model_b(self, test_name: str) -> bool:
        """Determine if should use model B for this request (traffic splitting).
        
        Args:
            test_name: Name of the test
            
        Returns:
            True if should use model B, False for model A
        """
        if test_name not in self.active_tests:
            return False

        config = self.active_tests[test_name]
        return np.random.random() < config.traffic_split

    def record_result(
        self,
        test_name: str,
        model_id: str,
        metrics: Dict[str, float],
        version: Optional[str] = None
    ) -> None:
        """Record result from a single request.
        
        Args:
            test_name: Name of the test
            model_id: Model identifier that was used
            metrics: Dictionary of metric_name -> value
            version: Model version (optional)
        """
        if test_name not in self.active_tests:
            logger.warning(f"Test {test_name} not active")
            return

        result = {
            "model_id": model_id,
            "version": version,
            "timestamp": datetime.now(),
            "metrics": metrics
        }

        self.results[test_name].append(result)

    def analyze_test(self, test_name: str) -> ABTestResult:
        """Analyze A/B test results.
        
        Performs statistical analysis including t-tests and confidence intervals
        to determine if there's a significant difference between models.
        
        Args:
            test_name: Name of the test
            
        Returns:
            Test results with statistical analysis
            
        Raises:
            ValueError: If test not found
            ImportError: If scipy is not available
        """
        if not SCIPY_AVAILABLE or stats is None:
            raise ImportError(
                "scipy is required for A/B testing statistical analysis. "
                "Install with: pip install scipy"
            )
            
        logger.info(f"Analyzing A/B test: {test_name}")

        if test_name not in self.active_tests:
            raise ValueError(f"Test {test_name} not found")

        config = self.active_tests[test_name]
        results = self.results[test_name]

        # Filter results by model ID and version
        def filter_results(model_id: str, version: Optional[str]) -> List[Dict[str, Any]]:
            filtered = []
            for r in results:
                if r["model_id"] != model_id:
                    continue
                if version and r.get("version") != version:
                    continue
                filtered.append(r)
            return filtered

        model_a_results = filter_results(config.model_a_id, config.model_a_version)
        model_b_results = filter_results(config.model_b_id, config.model_b_version)

        # Check minimum samples
        if len(model_a_results) < config.minimum_samples or len(model_b_results) < config.minimum_samples:
            logger.warning(
                f"Insufficient samples: A={len(model_a_results)}, B={len(model_b_results)} "
                f"(minimum={config.minimum_samples})"
            )

        # Calculate metrics for each model
        metrics_comparison = {}
        p_values = {}
        confidence_intervals = {}

        for metric_name in config.metrics_to_track:
            # Extract metric values
            model_a_values = [
                r["metrics"].get(metric_name)
                for r in model_a_results
                if metric_name in r["metrics"]
            ]
            model_b_values = [
                r["metrics"].get(metric_name)
                for r in model_b_results
                if metric_name in r["metrics"]
            ]

            if not model_a_values or not model_b_values:
                continue

            # Calculate means
            model_a_mean = np.mean(model_a_values)
            model_b_mean = np.mean(model_b_values)

            metrics_comparison[metric_name] = {
                "model_a": model_a_mean,
                "model_b": model_b_mean,
                "improvement": ((model_b_mean - model_a_mean) / model_a_mean) * 100 if model_a_mean != 0 else 0
            }

            # Statistical test (t-test)
            t_stat, p_value = stats.ttest_ind(model_a_values, model_b_values)
            p_values[metric_name] = p_value

            # Confidence interval for difference
            min_len = min(len(model_a_values), len(model_b_values))
            diff = np.array(model_b_values[:min_len]) - np.array(model_a_values[:min_len])
            
            if len(diff) > 1:
                ci = stats.t.interval(
                    0.95,
                    len(diff) - 1,
                    loc=np.mean(diff),
                    scale=stats.sem(diff)
                )
                confidence_intervals[metric_name] = ci
            else:
                confidence_intervals[metric_name] = (0.0, 0.0)

        # Determine winner
        winner, recommendation = self._determine_winner(
            metrics_comparison,
            p_values,
            config.statistical_threshold
        )

        # Create result object
        result = ABTestResult(
            test_name=test_name,
            model_a_id=config.model_a_id,
            model_b_id=config.model_b_id,
            model_a_version=config.model_a_version,
            model_b_version=config.model_b_version,
            model_a_samples=len(model_a_results),
            model_b_samples=len(model_b_results),
            metrics=metrics_comparison,
            p_values=p_values,
            confidence_intervals=confidence_intervals,
            winner=winner,
            recommendation=recommendation,
            start_time=results[0]["timestamp"] if results else datetime.now(),
            end_time=results[-1]["timestamp"] if results else datetime.now()
        )

        logger.info(f"Test analysis complete. Winner: {winner}")

        return result

    def _determine_winner(
        self,
        metrics: Dict[str, Dict[str, float]],
        p_values: Dict[str, float],
        threshold: float
    ) -> Tuple[Optional[str], str]:
        """Determine winner based on statistical significance.
        
        Args:
            metrics: Metric comparisons
            p_values: P-values for each metric
            threshold: Significance threshold
            
        Returns:
            Tuple of (winner, recommendation)
        """
        significant_improvements = 0
        significant_degradations = 0

        for metric_name, values in metrics.items():
            p_value = p_values.get(metric_name)
            if p_value is None:
                continue

            improvement = values["improvement"]

            # Check statistical significance
            if p_value < threshold:
                if improvement > 0:
                    significant_improvements += 1
                else:
                    significant_degradations += 1

        # Decision logic
        if significant_improvements > significant_degradations:
            return "model_b", "Model B shows statistically significant improvement"
        elif significant_degradations > significant_improvements:
            return "model_a", "Model A performs better, stay with current model"
        else:
            return "no_difference", "No statistically significant difference detected"

    def gradual_rollout(
        self,
        test_name: str,
        new_traffic_split: float
    ) -> None:
        """Gradually increase traffic to model B.
        
        Args:
            test_name: Name of the test
            new_traffic_split: New traffic split (0.0-1.0)
            
        Raises:
            ValueError: If test not found
        """
        if test_name not in self.active_tests:
            raise ValueError(f"Test {test_name} not found")

        config = self.active_tests[test_name]
        config.traffic_split = max(0.0, min(1.0, new_traffic_split))

        logger.info(
            f"Updated traffic split for {test_name}: "
            f"{config.traffic_split * 100:.1f}% to model B"
        )
