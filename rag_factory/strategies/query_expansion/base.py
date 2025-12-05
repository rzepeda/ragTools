"""Base classes and interfaces for query expansion strategies."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class ExpansionStrategy(Enum):
    """Enumeration of query expansion strategies."""
    KEYWORD = "keyword"
    REFORMULATION = "reformulation"
    QUESTION_GENERATION = "question_generation"
    MULTI_QUERY = "multi_query"
    HYDE = "hyde"  # Hypothetical Document Expansion


@dataclass
class ExpandedQuery:
    """Result from query expansion."""
    original_query: str
    expanded_query: str
    expansion_strategy: ExpansionStrategy
    added_terms: List[str]
    reasoning: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpansionResult:
    """Complete result from query expansion service."""
    original_query: str
    expanded_queries: List[ExpandedQuery]
    primary_expansion: ExpandedQuery  # Main expanded query to use
    execution_time_ms: float
    cache_hit: bool
    llm_used: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpansionConfig:
    """Configuration for query expansion."""
    strategy: ExpansionStrategy = ExpansionStrategy.KEYWORD

    # LLM settings
    llm_model: str = "gpt-3.5-turbo"
    max_tokens: int = 150
    temperature: float = 0.3  # Lower for more consistent expansions

    # Expansion settings
    max_additional_terms: int = 5
    generate_multiple_variants: bool = False
    num_variants: int = 3

    # Strategy-specific settings
    include_synonyms: bool = True
    include_related_terms: bool = True
    preserve_query_structure: bool = True

    # Prompt customization
    system_prompt: Optional[str] = None
    domain_context: Optional[str] = None

    # Performance settings
    enable_cache: bool = True
    cache_ttl: int = 3600
    timeout_seconds: float = 5.0

    # A/B testing
    enable_expansion: bool = True
    track_metrics: bool = True

    # Additional config
    extra_config: Dict[str, Any] = field(default_factory=dict)


class IQueryExpander(ABC):
    """Abstract base class for query expansion strategies."""

    def __init__(self, config: ExpansionConfig):
        """Initialize expander with configuration.

        Args:
            config: Configuration for query expansion
        """
        self.config = config

    @abstractmethod
    def expand(self, query: str) -> ExpandedQuery:
        """
        Expand a query to improve search.

        Args:
            query: The original user query

        Returns:
            ExpandedQuery with expansion details
        """
        pass

    def validate_query(self, query: str) -> None:
        """Validate input query.

        Args:
            query: Query to validate

        Raises:
            ValueError: If query is invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if len(query) > 1000:
            raise ValueError(f"Query too long: {len(query)} characters (max: 1000)")

    def extract_added_terms(self, original: str, expanded: str) -> List[str]:
        """Extract terms that were added during expansion.

        Args:
            original: Original query
            expanded: Expanded query

        Returns:
            List of terms added during expansion
        """
        original_terms = set(original.lower().split())
        expanded_terms = set(expanded.lower().split())
        added = expanded_terms - original_terms
        return list(added)
