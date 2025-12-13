"""LLM-based query expander implementation."""

from typing import Optional, List
from .base import IQueryExpander, ExpandedQuery, ExpansionConfig, ExpansionStrategy
from .prompts import ExpansionPrompts
from ...services.llm.service import LLMService
from ...services.llm.base import Message, MessageRole


class LLMQueryExpander(IQueryExpander):
    """
    Query expander using LLM to intelligently expand queries.
    Supports multiple expansion strategies via different prompts.
    """

    def __init__(self, config: ExpansionConfig, llm_service: LLMService):
        """Initialize LLM expander.

        Args:
            config: Expansion configuration
            llm_service: LLM service for generating expansions
        """
        super().__init__(config)
        self.llm_service = llm_service
        self.prompts = ExpansionPrompts(config)

    def expand(self, query: str) -> ExpandedQuery:
        """Expand query using LLM.

        Args:
            query: Original query to expand

        Returns:
            ExpandedQuery with expansion details

        Raises:
            ValueError: If query is invalid
        """
        self.validate_query(query)

        # Get appropriate prompt for strategy
        system_prompt = self.prompts.get_system_prompt(self.config.strategy)
        user_prompt = self.prompts.get_user_prompt(query, self.config.strategy)

        # Add domain context if provided
        if self.config.domain_context:
            system_prompt = f"{system_prompt}\n\nDomain context: {self.config.domain_context}"

        # Build messages for LLM
        messages = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=user_prompt)
        ]

        # Call LLM
        response = self.llm_service.complete(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        expanded_query = response.content.strip()
        
        # Strip surrounding quotes if present (some models wrap responses in quotes)
        if expanded_query.startswith('"') and expanded_query.endswith('"'):
            expanded_query = expanded_query[1:-1].strip()

        # Extract added terms
        added_terms = self.extract_added_terms(query, expanded_query)

        return ExpandedQuery(
            original_query=query,
            expanded_query=expanded_query,
            expansion_strategy=self.config.strategy,
            added_terms=added_terms,
            reasoning=self._extract_reasoning(response.content),
            confidence=1.0,
            metadata={
                "llm_model": self.config.llm_model,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
                "cost": response.cost
            }
        )

    def _extract_reasoning(self, response: str) -> Optional[str]:
        """Extract reasoning from LLM response if present.

        Args:
            response: LLM response text

        Returns:
            Extracted reasoning or None
        """
        # Some prompts ask LLM to explain reasoning
        # This could be parsed from structured output
        # For now, we don't extract reasoning from the response
        return None
