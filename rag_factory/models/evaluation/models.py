"""Data models for A/B testing infrastructure."""

from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime


class ABTestConfig(BaseModel):
    """Configuration for A/B testing.
    
    Attributes:
        test_name: Unique name for the test
        model_a_id: Control model identifier
        model_b_id: Experiment model identifier
        traffic_split: Percentage of traffic to model B (0.0-1.0)
        duration_days: Duration of test in days
        minimum_samples: Minimum samples required per model
        metrics_to_track: List of metric names to track
        statistical_threshold: P-value threshold for significance (default 0.05)
    """
    test_name: str
    model_a_id: str
    model_b_id: str
    model_a_version: Optional[str] = None
    model_b_version: Optional[str] = None
    traffic_split: float = 0.5
    duration_days: int = 7
    minimum_samples: int = 1000
    metrics_to_track: List[str] = Field(
        default_factory=lambda: ["latency", "accuracy", "recall@5"]
    )
    statistical_threshold: float = 0.05


class ABTestResult(BaseModel):
    """Results from A/B test.
    
    Attributes:
        test_name: Name of the test
        model_a_id: Control model identifier
        model_b_id: Experiment model identifier
        model_a_samples: Number of samples for model A
        model_b_samples: Number of samples for model B
        metrics: Metric comparisons {metric_name: {model_a: val, model_b: val, improvement: %}}
        p_values: P-values for each metric
        confidence_intervals: 95% confidence intervals for differences
        winner: Winner determination ("model_a", "model_b", or "no_difference")
        recommendation: Human-readable recommendation
        start_time: Test start time
        end_time: Test end time
    """
    test_name: str
    model_a_id: str
    model_b_id: str
    model_a_version: Optional[str] = None
    model_b_version: Optional[str] = None

    # Sample counts
    model_a_samples: int
    model_b_samples: int

    # Metrics
    metrics: Dict[str, Dict[str, float]]

    # Statistical tests
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]

    # Decision
    winner: Optional[str] = None
    recommendation: str = ""

    start_time: datetime
    end_time: datetime
