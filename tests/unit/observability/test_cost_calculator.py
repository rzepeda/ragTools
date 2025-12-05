"""Unit tests for cost calculator."""

import pytest

from rag_factory.observability.metrics.cost import (
    CostCalculator,
    ModelPricing,
    get_cost_calculator,
    DEFAULT_PRICING,
)


@pytest.fixture
def calculator():
    """Create a CostCalculator instance for testing."""
    return CostCalculator()


@pytest.fixture
def reset_global_calculator():
    """Reset global calculator after each test."""
    yield
    import rag_factory.observability.metrics.cost as cost_module
    cost_module._global_calculator = None


class TestModelPricing:
    """Tests for ModelPricing dataclass."""

    def test_model_pricing_creation(self):
        """Test creating model pricing."""
        pricing = ModelPricing(
            input_cost_per_1k_tokens=0.01,
            output_cost_per_1k_tokens=0.03,
            provider="openai",
            model_name="gpt-4",
        )

        assert pricing.input_cost_per_1k_tokens == 0.01
        assert pricing.output_cost_per_1k_tokens == 0.03
        assert pricing.provider == "openai"
        assert pricing.model_name == "gpt-4"


class TestCostCalculator:
    """Tests for CostCalculator class."""

    def test_calculator_initialization(self, calculator):
        """Test calculator initializes with default pricing."""
        assert len(calculator.pricing) > 0
        assert "gpt-4" in calculator.pricing
        assert "claude-3-opus" in calculator.pricing

    def test_calculate_cost_llm(self, calculator):
        """Test cost calculation for LLM calls."""
        # GPT-4: $0.03/1K input, $0.06/1K output
        cost = calculator.calculate_cost(
            model="gpt-4",
            input_tokens=1000,
            output_tokens=500,
        )

        # Expected: (1000/1000) * 0.03 + (500/1000) * 0.06 = 0.03 + 0.03 = 0.06
        assert cost == pytest.approx(0.06, abs=0.001)

    def test_calculate_cost_embedding(self, calculator):
        """Test cost calculation for embeddings."""
        # text-embedding-3-small: $0.00002/1K tokens
        cost = calculator.calculate_embedding_cost(
            model="text-embedding-3-small",
            tokens=5000,
        )

        # Expected: (5000/1000) * 0.00002 = 0.0001
        assert cost == pytest.approx(0.0001, abs=0.000001)

    def test_calculate_cost_unknown_model(self, calculator):
        """Test cost calculation with unknown model."""
        with pytest.raises(ValueError, match="Pricing not found"):
            calculator.calculate_cost(
                model="unknown-model",
                input_tokens=1000,
                output_tokens=500,
            )

    def test_calculate_rerank_cost(self, calculator):
        """Test cost calculation for reranking."""
        cost = calculator.calculate_rerank_cost(
            model="rerank-english-v3.0",
            num_documents=10,
            query_tokens=10,
        )

        # Should calculate based on query_tokens * num_documents
        assert cost > 0

    def test_calculate_rerank_cost_invalid_model(self, calculator):
        """Test rerank cost with non-rerank model."""
        with pytest.raises(ValueError, match="not a reranking model"):
            calculator.calculate_rerank_cost(
                model="gpt-4",
                num_documents=10,
            )

    def test_add_custom_model(self, calculator):
        """Test adding custom model pricing."""
        calculator.add_custom_model(
            model_id="custom-model",
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.02,
            provider="custom-provider",
        )

        assert "custom-model" in calculator.pricing

        cost = calculator.calculate_cost(
            model="custom-model",
            input_tokens=1000,
            output_tokens=1000,
        )

        # Expected: 0.01 + 0.02 = 0.03
        assert cost == pytest.approx(0.03, abs=0.001)

    def test_get_model_pricing(self, calculator):
        """Test getting model pricing info."""
        pricing = calculator.get_model_pricing("gpt-4")

        assert pricing is not None
        assert pricing.model_name == "gpt-4"
        assert pricing.provider == "openai"

    def test_get_model_pricing_nonexistent(self, calculator):
        """Test getting pricing for non-existent model."""
        pricing = calculator.get_model_pricing("nonexistent")
        assert pricing is None

    def test_list_available_models(self, calculator):
        """Test listing all available models."""
        models = calculator.list_available_models()

        assert len(models) > 0
        assert "gpt-4" in models
        assert "claude-3-opus" in models

    def test_custom_pricing_override(self):
        """Test initializing with custom pricing."""
        custom_pricing = {
            "test-model": ModelPricing(
                input_cost_per_1k_tokens=0.001,
                output_cost_per_1k_tokens=0.002,
                provider="test",
                model_name="test-model",
            )
        }

        calculator = CostCalculator(custom_pricing=custom_pricing)

        assert "test-model" in calculator.pricing
        assert "gpt-4" in calculator.pricing  # Default models still available

    def test_zero_cost_models(self, calculator):
        """Test models with zero cost (e.g., local models)."""
        # Add a local model with zero cost
        calculator.add_custom_model(
            model_id="local-model",
            input_cost_per_1k=0.0,
            output_cost_per_1k=0.0,
            provider="local",
        )

        cost = calculator.calculate_cost(
            model="local-model",
            input_tokens=10000,
            output_tokens=5000,
        )

        assert cost == 0.0


class TestGetCostCalculator:
    """Tests for get_cost_calculator function."""

    def test_get_calculator_singleton(self, reset_global_calculator):
        """Test that get_cost_calculator returns singleton instance."""
        calc1 = get_cost_calculator()
        calc2 = get_cost_calculator()

        assert calc1 is calc2

    def test_get_calculator_with_custom_pricing(self, reset_global_calculator):
        """Test get_cost_calculator with custom pricing."""
        custom_pricing = {
            "test-model": ModelPricing(
                input_cost_per_1k_tokens=0.001,
                output_cost_per_1k_tokens=0.002,
                provider="test",
                model_name="test-model",
            )
        }

        calculator = get_cost_calculator(custom_pricing=custom_pricing)

        assert "test-model" in calculator.pricing


class TestDefaultPricing:
    """Tests for default pricing data."""

    def test_default_pricing_has_common_models(self):
        """Test that default pricing includes common models."""
        assert "gpt-4" in DEFAULT_PRICING
        assert "gpt-3.5-turbo" in DEFAULT_PRICING
        assert "claude-3-opus" in DEFAULT_PRICING
        assert "claude-3-sonnet" in DEFAULT_PRICING
        assert "text-embedding-ada-002" in DEFAULT_PRICING

    def test_default_pricing_structure(self):
        """Test that all default pricing entries are valid."""
        for model_id, pricing in DEFAULT_PRICING.items():
            assert isinstance(pricing, ModelPricing)
            assert pricing.input_cost_per_1k_tokens >= 0
            assert pricing.output_cost_per_1k_tokens >= 0
            assert pricing.provider
            assert pricing.model_name
