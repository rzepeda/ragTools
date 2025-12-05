"""
Cost evaluation metrics.

This module provides metrics for evaluating the cost of RAG operations,
including token usage and API costs.
"""

from typing import Optional, Dict, Any
from rag_factory.evaluation.metrics.base import IMetric, MetricResult, MetricType


class TokenUsage(IMetric):
    """
    Token Usage: Number of tokens consumed.

    Tracks input and output tokens used by LLM API calls.

    Args:
        token_type: Type of tokens (default: "total", can be "input" or "output")

    Example:
        >>> metric = TokenUsage(token_type="total")
        >>> result = metric.compute(
        ...     input_tokens=100,
        ...     output_tokens=50
        ... )
        >>> print(f"Total tokens: {result.value}")
    """

    def __init__(self, token_type: str = "total"):
        """
        Initialize Token Usage metric.

        Args:
            token_type: Type of tokens to track ("total", "input", or "output")

        Raises:
            ValueError: If token_type is not valid
        """
        if token_type not in ("total", "input", "output"):
            raise ValueError(f"token_type must be 'total', 'input', or 'output', got {token_type}")
        super().__init__(f"{token_type}_tokens", MetricType.COST)
        self.token_type = token_type

    def compute(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        query_id: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            query_id: Optional query identifier
            **kwargs: Additional metadata

        Returns:
            MetricResult with token count
        """
        if self.token_type == "total":
            token_count = input_tokens + output_tokens
        elif self.token_type == "input":
            token_count = input_tokens
        else:  # output
            token_count = output_tokens

        return MetricResult(
            name=self.name,
            value=float(token_count),
            metadata={
                "token_type": self.token_type,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                **kwargs
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return f"Number of {self.token_type} tokens consumed"

    @property
    def higher_is_better(self) -> bool:
        """Lower token usage is better (cheaper)."""
        return False


class APICost(IMetric):
    """
    API Cost: Estimated cost of API calls.

    Calculates cost based on token usage and pricing.

    Args:
        input_price_per_1k: Price per 1000 input tokens (default: 0.0)
        output_price_per_1k: Price per 1000 output tokens (default: 0.0)

    Example:
        >>> # GPT-4 pricing example
        >>> metric = APICost(
        ...     input_price_per_1k=0.03,
        ...     output_price_per_1k=0.06
        ... )
        >>> result = metric.compute(
        ...     input_tokens=1000,
        ...     output_tokens=500
        ... )
        >>> print(f"Cost: ${result.value:.4f}")
    """

    def __init__(
        self,
        input_price_per_1k: float = 0.0,
        output_price_per_1k: float = 0.0
    ):
        """
        Initialize API Cost metric.

        Args:
            input_price_per_1k: Price per 1000 input tokens
            output_price_per_1k: Price per 1000 output tokens
        """
        super().__init__("api_cost_usd", MetricType.COST)
        self.input_price_per_1k = input_price_per_1k
        self.output_price_per_1k = output_price_per_1k

    def compute(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        query_id: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute API cost.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            query_id: Optional query identifier
            **kwargs: Additional metadata

        Returns:
            MetricResult with cost in USD
        """
        input_cost = (input_tokens / 1000) * self.input_price_per_1k
        output_cost = (output_tokens / 1000) * self.output_price_per_1k
        total_cost = input_cost + output_cost

        return MetricResult(
            name=self.name,
            value=total_cost,
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "input_price_per_1k": self.input_price_per_1k,
                "output_price_per_1k": self.output_price_per_1k,
                "currency": "USD",
                **kwargs
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return "Estimated API cost in USD based on token usage"

    @property
    def higher_is_better(self) -> bool:
        """Lower cost is better."""
        return False


class CostPerQuery(IMetric):
    """
    Cost Per Query: Average cost per query.

    Measures the cost efficiency of the system.

    Example:
        >>> metric = CostPerQuery()
        >>> result = metric.compute(
        ...     total_cost=1.50,
        ...     query_count=100
        ... )
        >>> print(f"Cost per query: ${result.value:.4f}")
    """

    def __init__(self):
        """Initialize Cost Per Query metric."""
        super().__init__("cost_per_query_usd", MetricType.COST)

    def compute(
        self,
        total_cost: float,
        query_count: int,
        **kwargs
    ) -> MetricResult:
        """
        Compute cost per query.

        Args:
            total_cost: Total cost in USD
            query_count: Number of queries
            **kwargs: Additional metadata

        Returns:
            MetricResult with cost per query

        Raises:
            ValueError: If query_count is zero
        """
        if query_count <= 0:
            raise ValueError(f"query_count must be positive, got {query_count}")

        cost_per_query = total_cost / query_count

        return MetricResult(
            name=self.name,
            value=cost_per_query,
            metadata={
                "total_cost": total_cost,
                "query_count": query_count,
                "currency": "USD",
                **kwargs
            },
            query_id=None
        )

    @property
    def description(self) -> str:
        return "Average cost per query in USD"

    @property
    def higher_is_better(self) -> bool:
        """Lower cost per query is better."""
        return False


class CostEfficiency(IMetric):
    """
    Cost Efficiency: Quality per dollar spent.

    Measures how much quality (accuracy) you get per unit cost.
    Higher is better - more bang for your buck.

    Example:
        >>> metric = CostEfficiency()
        >>> result = metric.compute(
        ...     accuracy=0.85,
        ...     cost=0.05
        ... )
        >>> print(f"Efficiency: {result.value:.2f} accuracy/$")
    """

    def __init__(self):
        """Initialize Cost Efficiency metric."""
        super().__init__("cost_efficiency", MetricType.COST)

    def compute(
        self,
        accuracy: float,
        cost: float,
        query_id: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute cost efficiency.

        Args:
            accuracy: Accuracy score (0.0 to 1.0)
            cost: Cost in USD
            query_id: Optional query identifier
            **kwargs: Additional metadata

        Returns:
            MetricResult with efficiency score

        Raises:
            ValueError: If cost is zero
        """
        if cost <= 0:
            raise ValueError(f"cost must be positive, got {cost}")

        efficiency = accuracy / cost

        return MetricResult(
            name=self.name,
            value=efficiency,
            metadata={
                "accuracy": accuracy,
                "cost": cost,
                "unit": "accuracy_per_dollar",
                **kwargs
            },
            query_id=query_id
        )

    @property
    def description(self) -> str:
        return "Quality (accuracy) achieved per dollar spent"

    @property
    def higher_is_better(self) -> bool:
        """Higher efficiency is better."""
        return True
