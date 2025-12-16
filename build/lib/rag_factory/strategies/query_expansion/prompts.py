"""Prompt templates for different expansion strategies."""

from typing import Dict
from .base import ExpansionStrategy, ExpansionConfig


class ExpansionPrompts:
    """Prompt templates for different expansion strategies."""

    def __init__(self, config: ExpansionConfig):
        """Initialize prompts with configuration.

        Args:
            config: Expansion configuration
        """
        self.config = config

    def get_system_prompt(self, strategy: ExpansionStrategy) -> str:
        """Get system prompt for expansion strategy.

        Args:
            strategy: The expansion strategy to use

        Returns:
            System prompt for the strategy
        """
        # Allow custom system prompt override
        if self.config.system_prompt:
            return self.config.system_prompt

        prompts = {
            ExpansionStrategy.KEYWORD: """You are a search query optimization expert.
Your task is to expand user queries by adding relevant keywords and synonyms to improve search results.
Add specific terms that capture the query's intent without changing its meaning.
Keep expansions concise and focused.""",

            ExpansionStrategy.REFORMULATION: """You are a search query optimization expert.
Your task is to reformulate user queries to make them more specific and searchable.
Rephrase the query to be clearer and more precise while preserving the original intent.
Focus on making the query more actionable for retrieval.""",

            ExpansionStrategy.QUESTION_GENERATION: """You are a search query optimization expert.
Your task is to convert user queries into well-formed questions that capture their information need.
Generate clear, specific questions that would help find relevant information.""",

            ExpansionStrategy.MULTI_QUERY: """You are a search query optimization expert.
Your task is to generate multiple variations of the user's query to improve search coverage.
Create diverse queries that capture different aspects of the user's information need.""",

            ExpansionStrategy.HYDE: """You are a search query optimization expert.
Your task is to generate a hypothetical document or passage that would answer the user's query.
Create a realistic, detailed response that contains the information the user is looking for.
This will be used to search for similar real documents."""
        }

        return prompts.get(strategy, prompts[ExpansionStrategy.KEYWORD])

    def get_user_prompt(self, query: str, strategy: ExpansionStrategy) -> str:
        """Get user prompt with the query for expansion.

        Args:
            query: The user's original query
            strategy: The expansion strategy to use

        Returns:
            User prompt with the query
        """
        prompts = {
            ExpansionStrategy.KEYWORD: f"""Original query: "{query}"

Expand this query by adding {self.config.max_additional_terms} relevant keywords or synonyms.
Return only the expanded query, nothing else.

Expanded query:""",

            ExpansionStrategy.REFORMULATION: f"""Original query: "{query}"

Reformulate this query to be more specific and searchable.
Return only the reformulated query, nothing else.

Reformulated query:""",

            ExpansionStrategy.QUESTION_GENERATION: f"""Original query: "{query}"

Convert this into a clear, specific question.
Return only the question, nothing else.

Question:""",

            ExpansionStrategy.MULTI_QUERY: f"""Original query: "{query}"

Generate {self.config.num_variants} different variations of this query.
Each variation should capture a different aspect or perspective.
Return only the queries, one per line.

Variations:""",

            ExpansionStrategy.HYDE: f"""Original query: "{query}"

Generate a hypothetical document passage (2-3 sentences) that would perfectly answer this query.
Be specific and detailed.
Return only the passage, nothing else.

Passage:"""
        }

        return prompts.get(strategy, prompts[ExpansionStrategy.KEYWORD])
