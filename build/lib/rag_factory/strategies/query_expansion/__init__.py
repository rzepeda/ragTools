"""Query expansion strategies for improving search precision and recall."""

from .base import (
    ExpansionStrategy,
    ExpandedQuery,
    ExpansionResult,
    ExpansionConfig,
    IQueryExpander
)
from .expander_service import QueryExpanderService
from .llm_expander import LLMQueryExpander
from .hyde_expander import HyDEExpander
from .prompts import ExpansionPrompts
from .cache import ExpansionCache
from .metrics import (
    ExpansionMetrics,
    AggregatedMetrics,
    MetricsTracker
)

__all__ = [
    # Base classes and enums
    "ExpansionStrategy",
    "ExpandedQuery",
    "ExpansionResult",
    "ExpansionConfig",
    "IQueryExpander",

    # Main service
    "QueryExpanderService",

    # Expanders
    "LLMQueryExpander",
    "HyDEExpander",

    # Supporting components
    "ExpansionPrompts",
    "ExpansionCache",

    # Metrics
    "ExpansionMetrics",
    "AggregatedMetrics",
    "MetricsTracker"
]
