"""Unit tests for base query expansion classes."""

import pytest
from rag_factory.strategies.query_expansion.base import (
    ExpansionStrategy,
    ExpandedQuery,
    ExpansionResult,
    ExpansionConfig,
    IQueryExpander
)


class TestExpansionStrategy:
    """Tests for ExpansionStrategy enum."""

    def test_strategy_values(self):
        """Test all strategy values are defined."""
        assert ExpansionStrategy.KEYWORD.value == "keyword"
        assert ExpansionStrategy.REFORMULATION.value == "reformulation"
        assert ExpansionStrategy.QUESTION_GENERATION.value == "question_generation"
        assert ExpansionStrategy.MULTI_QUERY.value == "multi_query"
        assert ExpansionStrategy.HYDE.value == "hyde"


class TestExpandedQuery:
    """Tests for ExpandedQuery dataclass."""

    def test_expanded_query_creation(self):
        """Test creating an expanded query."""
        query = ExpandedQuery(
            original_query="test",
            expanded_query="test query",
            expansion_strategy=ExpansionStrategy.KEYWORD,
            added_terms=["query"]
        )

        assert query.original_query == "test"
        assert query.expanded_query == "test query"
        assert query.expansion_strategy == ExpansionStrategy.KEYWORD
        assert query.added_terms == ["query"]
        assert query.confidence == 1.0

    def test_expanded_query_with_reasoning(self):
        """Test expanded query with reasoning."""
        query = ExpandedQuery(
            original_query="test",
            expanded_query="test query",
            expansion_strategy=ExpansionStrategy.KEYWORD,
            added_terms=["query"],
            reasoning="Added relevant keyword"
        )

        assert query.reasoning == "Added relevant keyword"

    def test_expanded_query_with_metadata(self):
        """Test expanded query with metadata."""
        metadata = {"llm_model": "gpt-3.5-turbo", "tokens": 100}
        query = ExpandedQuery(
            original_query="test",
            expanded_query="test query",
            expansion_strategy=ExpansionStrategy.KEYWORD,
            added_terms=["query"],
            metadata=metadata
        )

        assert query.metadata == metadata


class TestExpansionConfig:
    """Tests for ExpansionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExpansionConfig()

        assert config.strategy == ExpansionStrategy.KEYWORD
        assert config.llm_model == "gpt-3.5-turbo"
        assert config.max_tokens == 150
        assert config.temperature == 0.3
        assert config.max_additional_terms == 5
        assert config.enable_cache is True
        assert config.enable_expansion is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExpansionConfig(
            strategy=ExpansionStrategy.HYDE,
            llm_model="gpt-4",
            max_tokens=200,
            temperature=0.7
        )

        assert config.strategy == ExpansionStrategy.HYDE
        assert config.llm_model == "gpt-4"
        assert config.max_tokens == 200
        assert config.temperature == 0.7


class MockQueryExpander(IQueryExpander):
    """Mock implementation of IQueryExpander for testing."""

    def expand(self, query: str) -> ExpandedQuery:
        """Mock expansion that just adds 'expanded' to query."""
        self.validate_query(query)
        expanded = f"{query} expanded"
        return ExpandedQuery(
            original_query=query,
            expanded_query=expanded,
            expansion_strategy=self.config.strategy,
            added_terms=self.extract_added_terms(query, expanded)
        )


class TestIQueryExpander:
    """Tests for IQueryExpander base class."""

    def test_validate_query_empty(self):
        """Test validation rejects empty query."""
        config = ExpansionConfig()
        expander = MockQueryExpander(config)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            expander.validate_query("")

    def test_validate_query_whitespace(self):
        """Test validation rejects whitespace-only query."""
        config = ExpansionConfig()
        expander = MockQueryExpander(config)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            expander.validate_query("   ")

    def test_validate_query_too_long(self):
        """Test validation rejects overly long query."""
        config = ExpansionConfig()
        expander = MockQueryExpander(config)

        long_query = "x" * 1001
        with pytest.raises(ValueError, match="Query too long"):
            expander.validate_query(long_query)

    def test_validate_query_valid(self):
        """Test validation accepts valid query."""
        config = ExpansionConfig()
        expander = MockQueryExpander(config)

        # Should not raise
        expander.validate_query("valid query")

    def test_extract_added_terms(self):
        """Test extraction of added terms."""
        config = ExpansionConfig()
        expander = MockQueryExpander(config)

        original = "machine learning"
        expanded = "machine learning algorithms neural networks"

        added = expander.extract_added_terms(original, expanded)

        assert "algorithms" in added
        assert "neural" in added
        assert "networks" in added
        assert "machine" not in added
        assert "learning" not in added

    def test_extract_added_terms_no_additions(self):
        """Test extraction when no terms added."""
        config = ExpansionConfig()
        expander = MockQueryExpander(config)

        original = "test query"
        expanded = "test query"

        added = expander.extract_added_terms(original, expanded)

        assert len(added) == 0

    def test_expand_mock(self):
        """Test mock expander expansion."""
        config = ExpansionConfig()
        expander = MockQueryExpander(config)

        result = expander.expand("test")

        assert result.original_query == "test"
        assert result.expanded_query == "test expanded"
        assert "expanded" in result.added_terms
