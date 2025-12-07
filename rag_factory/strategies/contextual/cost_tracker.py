"""
Cost tracking for LLM API usage in context generation.

This module provides cost tracking functionality to monitor and manage
LLM API costs during contextual chunk enrichment.
"""

from typing import Dict, Any
import logging

from .config import ContextualRetrievalConfig

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Tracks LLM API costs for context generation.
    
    Monitors token usage and calculates costs based on configured pricing.
    Provides budget alerts and cost summaries.
    """

    def __init__(self, config: ContextualRetrievalConfig):
        """
        Initialize cost tracker.
        
        Args:
            config: Contextual retrieval configuration with pricing info
        """
        self.config = config
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.chunk_costs: Dict[str, Dict[str, Any]] = {}

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Total cost in USD
        """
        input_cost = (input_tokens / 1000) * self.config.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * self.config.cost_per_1k_output_tokens
        
        return input_cost + output_cost

    def record_chunk_cost(
        self,
        chunk_id: str,
        input_tokens: int,
        output_tokens: int,
        cost: float
    ) -> None:
        """
        Record cost for a chunk.
        
        Args:
            chunk_id: Chunk identifier
            input_tokens: Input tokens used
            output_tokens: Output tokens generated
            cost: Total cost for this chunk
        """
        self.chunk_costs[chunk_id] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost
        }
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        
        # Check budget alert
        if (self.config.budget_alert_threshold and
            self.total_cost >= self.config.budget_alert_threshold):
            logger.warning(
                f"Cost alert: Total cost ${self.total_cost:.2f} "
                f"exceeds threshold ${self.config.budget_alert_threshold:.2f}"
            )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get cost tracking summary.
        
        Returns:
            Dictionary with cost statistics
        """
        avg_cost_per_chunk = (
            self.total_cost / len(self.chunk_costs)
            if self.chunk_costs else 0.0
        )
        
        return {
            "total_chunks": len(self.chunk_costs),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "avg_cost_per_chunk": avg_cost_per_chunk,
            "cost_per_1k_chunks": avg_cost_per_chunk * 1000 if avg_cost_per_chunk > 0 else 0
        }

    def reset(self) -> None:
        """Reset cost tracking for new document."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.chunk_costs = {}

    def check_budget_limit(self, max_cost: float) -> bool:
        """
        Check if cost is within budget.
        
        Args:
            max_cost: Maximum allowed cost
            
        Returns:
            True if within budget, False otherwise
        """
        return self.total_cost <= max_cost
