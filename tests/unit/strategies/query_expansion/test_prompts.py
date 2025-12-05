"""Unit tests for expansion prompts."""

import pytest
from rag_factory.strategies.query_expansion.prompts import ExpansionPrompts
from rag_factory.strategies.query_expansion.base import ExpansionStrategy, ExpansionConfig


class TestExpansionPrompts:
    """Tests for ExpansionPrompts class."""

    def test_initialization(self):
        """Test prompts initialization."""
        config = ExpansionConfig()
        prompts = ExpansionPrompts(config)

        assert prompts.config == config

    def test_get_system_prompt_keyword(self):
        """Test keyword expansion system prompt."""
        config = ExpansionConfig(strategy=ExpansionStrategy.KEYWORD)
        prompts = ExpansionPrompts(config)

        prompt = prompts.get_system_prompt(ExpansionStrategy.KEYWORD)

        assert "search query optimization" in prompt.lower()
        assert "keywords" in prompt.lower()
        assert "synonyms" in prompt.lower()

    def test_get_system_prompt_reformulation(self):
        """Test reformulation system prompt."""
        config = ExpansionConfig(strategy=ExpansionStrategy.REFORMULATION)
        prompts = ExpansionPrompts(config)

        prompt = prompts.get_system_prompt(ExpansionStrategy.REFORMULATION)

        assert "reformulate" in prompt.lower()
        assert "specific" in prompt.lower()

    def test_get_system_prompt_question_generation(self):
        """Test question generation system prompt."""
        config = ExpansionConfig(strategy=ExpansionStrategy.QUESTION_GENERATION)
        prompts = ExpansionPrompts(config)

        prompt = prompts.get_system_prompt(ExpansionStrategy.QUESTION_GENERATION)

        assert "question" in prompt.lower()

    def test_get_system_prompt_multi_query(self):
        """Test multi-query system prompt."""
        config = ExpansionConfig(strategy=ExpansionStrategy.MULTI_QUERY)
        prompts = ExpansionPrompts(config)

        prompt = prompts.get_system_prompt(ExpansionStrategy.MULTI_QUERY)

        assert "multiple" in prompt.lower() or "variations" in prompt.lower()

    def test_get_system_prompt_hyde(self):
        """Test HyDE system prompt."""
        config = ExpansionConfig(strategy=ExpansionStrategy.HYDE)
        prompts = ExpansionPrompts(config)

        prompt = prompts.get_system_prompt(ExpansionStrategy.HYDE)

        assert "hypothetical" in prompt.lower()
        assert "document" in prompt.lower()

    def test_custom_system_prompt(self):
        """Test custom system prompt override."""
        custom_prompt = "This is a custom system prompt"
        config = ExpansionConfig(system_prompt=custom_prompt)
        prompts = ExpansionPrompts(config)

        prompt = prompts.get_system_prompt(ExpansionStrategy.KEYWORD)

        assert prompt == custom_prompt

    def test_get_user_prompt_keyword(self):
        """Test keyword user prompt."""
        config = ExpansionConfig(
            strategy=ExpansionStrategy.KEYWORD,
            max_additional_terms=5
        )
        prompts = ExpansionPrompts(config)

        query = "machine learning"
        prompt = prompts.get_user_prompt(query, ExpansionStrategy.KEYWORD)

        assert query in prompt
        assert "5" in prompt
        assert "expand" in prompt.lower()

    def test_get_user_prompt_reformulation(self):
        """Test reformulation user prompt."""
        config = ExpansionConfig(strategy=ExpansionStrategy.REFORMULATION)
        prompts = ExpansionPrompts(config)

        query = "how does it work"
        prompt = prompts.get_user_prompt(query, ExpansionStrategy.REFORMULATION)

        assert query in prompt
        assert "reformulate" in prompt.lower()

    def test_get_user_prompt_question_generation(self):
        """Test question generation user prompt."""
        config = ExpansionConfig(strategy=ExpansionStrategy.QUESTION_GENERATION)
        prompts = ExpansionPrompts(config)

        query = "python tutorial"
        prompt = prompts.get_user_prompt(query, ExpansionStrategy.QUESTION_GENERATION)

        assert query in prompt
        assert "question" in prompt.lower()

    def test_get_user_prompt_multi_query(self):
        """Test multi-query user prompt."""
        config = ExpansionConfig(
            strategy=ExpansionStrategy.MULTI_QUERY,
            num_variants=3
        )
        prompts = ExpansionPrompts(config)

        query = "climate change"
        prompt = prompts.get_user_prompt(query, ExpansionStrategy.MULTI_QUERY)

        assert query in prompt
        assert "3" in prompt
        assert "variations" in prompt.lower() or "generate" in prompt.lower()

    def test_get_user_prompt_hyde(self):
        """Test HyDE user prompt."""
        config = ExpansionConfig(strategy=ExpansionStrategy.HYDE)
        prompts = ExpansionPrompts(config)

        query = "What is the capital of France?"
        prompt = prompts.get_user_prompt(query, ExpansionStrategy.HYDE)

        assert query in prompt
        assert "hypothetical" in prompt.lower() or "passage" in prompt.lower()
