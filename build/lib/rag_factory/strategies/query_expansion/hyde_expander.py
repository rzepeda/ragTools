"""HyDE (Hypothetical Document Expansion) implementation."""

from .base import IQueryExpander, ExpandedQuery, ExpansionConfig, ExpansionStrategy
from .prompts import ExpansionPrompts
from ...services.llm.service import LLMService
from ...services.llm.base import Message, MessageRole


class HyDEExpander(IQueryExpander):
    """
    Hypothetical Document Expansion (HyDE).

    Instead of expanding the query with keywords, HyDE generates a hypothetical
    document/passage that would answer the query, then uses that for retrieval.
    This often works better than keyword expansion for semantic search.
    """

    def __init__(self, config: ExpansionConfig, llm_service: LLMService):
        """Initialize HyDE expander.

        Args:
            config: Expansion configuration
            llm_service: LLM service for generating hypothetical documents
        """
        super().__init__(config)
        self.llm_service = llm_service
        self.prompts = ExpansionPrompts(config)

    def expand(self, query: str) -> ExpandedQuery:
        """Generate hypothetical document for query.

        Args:
            query: Original query to expand

        Returns:
            ExpandedQuery with hypothetical document

        Raises:
            ValueError: If query is invalid
        """
        self.validate_query(query)

        system_prompt = self.prompts.get_system_prompt(ExpansionStrategy.HYDE)
        user_prompt = self.prompts.get_user_prompt(query, ExpansionStrategy.HYDE)

        # Add domain context if provided
        if self.config.domain_context:
            system_prompt = f"{system_prompt}\n\nDomain context: {self.config.domain_context}"

        # Build messages for LLM
        messages = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=user_prompt)
        ]

        # Generate hypothetical document with slightly higher temperature for creativity
        response = self.llm_service.complete(
            messages=messages,
            temperature=0.7,  # Higher than keyword expansion for more creative responses
            max_tokens=self.config.max_tokens
        )

        hypothetical_doc = response.content.strip()

        return ExpandedQuery(
            original_query=query,
            expanded_query=hypothetical_doc,
            expansion_strategy=ExpansionStrategy.HYDE,
            added_terms=[],  # Not applicable for HyDE
            reasoning="Generated hypothetical document for semantic search",
            confidence=1.0,
            metadata={
                "method": "HyDE",
                "llm_model": self.config.llm_model,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
                "cost": response.cost
            }
        )
