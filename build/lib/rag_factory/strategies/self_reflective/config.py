"""Configuration for self-reflective RAG strategy."""

from dataclasses import dataclass
from typing import Optional
from .models import RefinementStrategy


@dataclass
class SelfReflectiveConfig:
    """Configuration for self-reflective RAG strategy.
    
    Attributes:
        grade_threshold: Minimum average grade to accept results (1.0-5.0)
        max_retries: Maximum number of retry attempts
        timeout_seconds: Maximum total time for retrieval
        batch_grading_size: Number of results to grade in one LLM call
        grading_prompt: Custom grading prompt template (optional)
        refinement_strategy: Default refinement strategy
    """
    grade_threshold: float = 4.0
    max_retries: int = 2
    timeout_seconds: int = 10
    batch_grading_size: int = 5
    grading_prompt: Optional[str] = None
    refinement_strategy: RefinementStrategy = RefinementStrategy.REFORMULATION
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 1.0 <= self.grade_threshold <= 5.0:
            raise ValueError(
                f"grade_threshold must be between 1.0 and 5.0, got {self.grade_threshold}"
            )
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got {self.max_retries}"
            )
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {self.timeout_seconds}"
            )
        if self.batch_grading_size <= 0:
            raise ValueError(
                f"batch_grading_size must be positive, got {self.batch_grading_size}"
            )
