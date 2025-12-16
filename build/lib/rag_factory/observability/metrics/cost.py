"""Cost tracking for RAG operations."""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Pricing information for a model.

    Attributes:
        input_cost_per_1k_tokens: Cost per 1000 input tokens
        output_cost_per_1k_tokens: Cost per 1000 output tokens
        provider: Provider name (e.g., "openai", "anthropic", "cohere")
        model_name: Model identifier
    """

    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float
    provider: str
    model_name: str


# Default pricing for common models (as of 2024)
DEFAULT_PRICING: Dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4": ModelPricing(0.03, 0.06, "openai", "gpt-4"),
    "gpt-4-turbo": ModelPricing(0.01, 0.03, "openai", "gpt-4-turbo"),
    "gpt-3.5-turbo": ModelPricing(0.0005, 0.0015, "openai", "gpt-3.5-turbo"),
    "text-embedding-ada-002": ModelPricing(
        0.0001, 0.0, "openai", "text-embedding-ada-002"
    ),
    "text-embedding-3-small": ModelPricing(
        0.00002, 0.0, "openai", "text-embedding-3-small"
    ),
    "text-embedding-3-large": ModelPricing(
        0.00013, 0.0, "openai", "text-embedding-3-large"
    ),
    # Anthropic
    "claude-3-opus": ModelPricing(0.015, 0.075, "anthropic", "claude-3-opus"),
    "claude-3-sonnet": ModelPricing(
        0.003, 0.015, "anthropic", "claude-3-sonnet"
    ),
    "claude-3-haiku": ModelPricing(0.00025, 0.00125, "anthropic", "claude-3-haiku"),
    # Cohere
    "command": ModelPricing(0.0015, 0.002, "cohere", "command"),
    "command-light": ModelPricing(0.0003, 0.0006, "cohere", "command-light"),
    "embed-english-v3.0": ModelPricing(0.0001, 0.0, "cohere", "embed-english-v3.0"),
    "embed-multilingual-v3.0": ModelPricing(
        0.0001, 0.0, "cohere", "embed-multilingual-v3.0"
    ),
    "rerank-english-v3.0": ModelPricing(0.002, 0.0, "cohere", "rerank-english-v3.0"),
    "rerank-multilingual-v3.0": ModelPricing(
        0.002, 0.0, "cohere", "rerank-multilingual-v3.0"
    ),
}


class CostCalculator:
    """Calculator for API costs based on token usage.

    Tracks and calculates costs for various LLM and embedding operations.

    Example:
        ```python
        calculator = CostCalculator()

        # Calculate cost for LLM call
        cost = calculator.calculate_cost(
            model="gpt-4",
            input_tokens=100,
            output_tokens=50
        )
        print(f"Cost: ${cost:.6f}")

        # Calculate cost for embedding
        cost = calculator.calculate_embedding_cost(
            model="text-embedding-3-small",
            tokens=500
        )
        ```
    """

    def __init__(self, custom_pricing: Optional[Dict[str, ModelPricing]] = None):
        """Initialize cost calculator.

        Args:
            custom_pricing: Optional custom pricing to override defaults
        """
        self.pricing = DEFAULT_PRICING.copy()
        if custom_pricing:
            self.pricing.update(custom_pricing)

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
    ) -> float:
        """Calculate cost for a model operation.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in dollars

        Raises:
            ValueError: If model pricing is not found
        """
        if model not in self.pricing:
            raise ValueError(
                f"Pricing not found for model '{model}'. "
                f"Available models: {list(self.pricing.keys())}"
            )

        pricing = self.pricing[model]

        input_cost = (input_tokens / 1000) * pricing.input_cost_per_1k_tokens
        output_cost = (output_tokens / 1000) * pricing.output_cost_per_1k_tokens

        return input_cost + output_cost

    def calculate_embedding_cost(self, model: str, tokens: int) -> float:
        """Calculate cost for an embedding operation.

        Args:
            model: Embedding model identifier
            tokens: Number of tokens

        Returns:
            Total cost in dollars
        """
        return self.calculate_cost(model, tokens, 0)

    def calculate_rerank_cost(
        self, model: str, num_documents: int, query_tokens: int = 10
    ) -> float:
        """Calculate cost for a reranking operation.

        Cohere charges per search (query + document pairs).

        Args:
            model: Reranker model identifier
            num_documents: Number of documents to rerank
            query_tokens: Approximate tokens in query

        Returns:
            Total cost in dollars
        """
        if "rerank" not in model:
            raise ValueError(f"Model '{model}' is not a reranking model")

        # Cohere charges per search unit
        # Approximate: query tokens * num_documents
        total_tokens = query_tokens * num_documents
        return self.calculate_cost(model, total_tokens, 0)

    def add_custom_model(
        self,
        model_id: str,
        input_cost_per_1k: float,
        output_cost_per_1k: float,
        provider: str,
    ):
        """Add custom model pricing.

        Args:
            model_id: Model identifier
            input_cost_per_1k: Cost per 1000 input tokens
            output_cost_per_1k: Cost per 1000 output tokens
            provider: Provider name
        """
        self.pricing[model_id] = ModelPricing(
            input_cost_per_1k_tokens=input_cost_per_1k,
            output_cost_per_1k_tokens=output_cost_per_1k,
            provider=provider,
            model_name=model_id,
        )

    def get_model_pricing(self, model: str) -> Optional[ModelPricing]:
        """Get pricing information for a model.

        Args:
            model: Model identifier

        Returns:
            ModelPricing object or None if not found
        """
        return self.pricing.get(model)

    def list_available_models(self) -> Dict[str, ModelPricing]:
        """Get all available model pricing.

        Returns:
            Dictionary of model pricing
        """
        return self.pricing.copy()


# Global cost calculator instance
_global_calculator: Optional[CostCalculator] = None


def get_cost_calculator(
    custom_pricing: Optional[Dict[str, ModelPricing]] = None,
) -> CostCalculator:
    """Get or create the global cost calculator instance.

    Args:
        custom_pricing: Optional custom pricing

    Returns:
        CostCalculator instance
    """
    global _global_calculator
    if _global_calculator is None:
        _global_calculator = CostCalculator(custom_pricing)
    return _global_calculator
