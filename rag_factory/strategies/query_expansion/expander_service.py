"""Main service for query expansion."""

from typing import List, Dict, Any, Optional
import time
import hashlib
import logging
from .base import IQueryExpander, ExpansionConfig, ExpansionResult, ExpandedQuery, ExpansionStrategy
from .llm_expander import LLMQueryExpander
from .hyde_expander import HyDEExpander
from .cache import ExpansionCache
from ...services.llm.service import LLMService

logger = logging.getLogger(__name__)


class QueryExpanderService:
    """
    Service for expanding user queries to improve search.

    Example:
        config = ExpansionConfig(
            strategy=ExpansionStrategy.KEYWORD,
            max_additional_terms=5
        )
        service = QueryExpanderService(config, llm_service)
        result = service.expand("machine learning")
    """

    def __init__(self, config: ExpansionConfig, llm_service: LLMService):
        """Initialize query expander service.

        Args:
            config: Expansion configuration
            llm_service: LLM service for generating expansions
        """
        self.config = config
        self.llm_service = llm_service
        self.expander = self._init_expander()
        self.cache = ExpansionCache(config) if config.enable_cache else None
        self._stats = {
            "total_expansions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_execution_time_ms": 0.0,
            "expansion_enabled_count": 0,
            "expansion_disabled_count": 0
        }

    def _init_expander(self) -> IQueryExpander:
        """Initialize the appropriate expander based on config.

        Returns:
            Initialized expander instance
        """
        if self.config.strategy == ExpansionStrategy.HYDE:
            return HyDEExpander(self.config, self.llm_service)
        else:
            return LLMQueryExpander(self.config, self.llm_service)

    def expand(
        self,
        query: str,
        enable_expansion: Optional[bool] = None
    ) -> ExpansionResult:
        """
        Expand a query to improve search.

        Args:
            query: The original user query
            enable_expansion: Override config to enable/disable expansion (for A/B testing)

        Returns:
            ExpansionResult with original and expanded queries
        """
        start_time = time.time()
        self._stats["total_expansions"] += 1

        # Check if expansion is enabled
        expansion_enabled = enable_expansion if enable_expansion is not None else self.config.enable_expansion

        if expansion_enabled:
            self._stats["expansion_enabled_count"] += 1
        else:
            self._stats["expansion_disabled_count"] += 1

        # If disabled, return original query
        if not expansion_enabled:
            return self._create_passthrough_result(query, start_time)

        # Check cache
        cache_hit = False
        if self.cache:
            cache_key = self._compute_cache_key(query)
            cached_result = self.cache.get(cache_key)

            if cached_result:
                self._stats["cache_hits"] += 1
                cached_result.cache_hit = True
                return cached_result

        self._stats["cache_misses"] += 1

        try:
            # Expand query
            if self.config.generate_multiple_variants:
                expanded_queries = self._expand_multiple(query)
            else:
                expanded_query = self.expander.expand(query)
                expanded_queries = [expanded_query]

            # Select primary expansion
            primary_expansion = expanded_queries[0]

            execution_time_ms = (time.time() - start_time) * 1000

            result = ExpansionResult(
                original_query=query,
                expanded_queries=expanded_queries,
                primary_expansion=primary_expansion,
                execution_time_ms=execution_time_ms,
                cache_hit=cache_hit,
                llm_used=self.config.llm_model
            )

            # Cache the result
            if self.cache:
                self.cache.set(cache_key, result)

            # Update stats
            self._update_avg_execution_time(execution_time_ms)

            # Log expansion
            if self.config.track_metrics:
                self._log_expansion(result)

            return result

        except Exception as e:
            logger.error(f"Query expansion failed: {e}", exc_info=True)
            # Fallback to original query
            return self._create_passthrough_result(query, start_time, error=str(e))

    def _expand_multiple(self, query: str) -> List[ExpandedQuery]:
        """Generate multiple query variations.

        Args:
            query: Original query

        Returns:
            List of expanded queries
        """
        expansions = []

        if self.config.strategy == ExpansionStrategy.MULTI_QUERY:
            # LLM will generate multiple queries
            expanded = self.expander.expand(query)

            # Parse multiple queries from response
            queries = expanded.expanded_query.split('\n')
            for q in queries[:self.config.num_variants]:
                if q.strip():
                    expansions.append(ExpandedQuery(
                        original_query=query,
                        expanded_query=q.strip(),
                        expansion_strategy=self.config.strategy,
                        added_terms=self.expander.extract_added_terms(query, q.strip()),
                        confidence=1.0
                    ))
        else:
            # Generate single expansion
            expansions.append(self.expander.expand(query))

        return expansions if expansions else [ExpandedQuery(
            original_query=query,
            expanded_query=query,
            expansion_strategy=self.config.strategy,
            added_terms=[],
            confidence=1.0
        )]

    def _create_passthrough_result(
        self,
        query: str,
        start_time: float,
        error: Optional[str] = None
    ) -> ExpansionResult:
        """Create result that passes through original query.

        Args:
            query: Original query
            start_time: Start time of expansion attempt
            error: Optional error message

        Returns:
            ExpansionResult with original query unchanged
        """
        execution_time_ms = (time.time() - start_time) * 1000

        passthrough = ExpandedQuery(
            original_query=query,
            expanded_query=query,
            expansion_strategy=self.config.strategy,
            added_terms=[],
            confidence=1.0,
            metadata={"passthrough": True, "error": error} if error else {"passthrough": True}
        )

        return ExpansionResult(
            original_query=query,
            expanded_queries=[passthrough],
            primary_expansion=passthrough,
            execution_time_ms=execution_time_ms,
            cache_hit=False,
            metadata={"expansion_disabled": True} if not error else {"error": error}
        )

    def _compute_cache_key(self, query: str) -> str:
        """Compute cache key for query.

        Args:
            query: Query to compute key for

        Returns:
            Cache key hash
        """
        content = f"{query}:{self.config.strategy.value}:{self.config.llm_model}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _update_avg_execution_time(self, new_time_ms: float):
        """Update average execution time.

        Args:
            new_time_ms: New execution time to include in average
        """
        total = self._stats["total_expansions"]
        current_avg = self._stats["avg_execution_time_ms"]

        new_avg = ((current_avg * (total - 1)) + new_time_ms) / total
        self._stats["avg_execution_time_ms"] = new_avg

    def _log_expansion(self, result: ExpansionResult):
        """Log expansion for analysis.

        Args:
            result: Expansion result to log
        """
        logger.info(
            f"Query expanded: '{result.original_query}' -> '{result.primary_expansion.expanded_query}' "
            f"(+{len(result.primary_expansion.added_terms)} terms, {result.execution_time_ms:.0f}ms)"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics.

        Returns:
            Dictionary with service statistics
        """
        cache_hit_rate = 0.0
        if self._stats["total_expansions"] > 0:
            cache_hit_rate = self._stats["cache_hits"] / self._stats["total_expansions"]

        expansion_rate = 0.0
        if self._stats["total_expansions"] > 0:
            expansion_rate = self._stats["expansion_enabled_count"] / self._stats["total_expansions"]

        return {
            **self._stats,
            "cache_hit_rate": cache_hit_rate,
            "expansion_rate": expansion_rate,
            "strategy": self.config.strategy.value,
            "llm_model": self.config.llm_model
        }

    def clear_cache(self):
        """Clear the expansion cache."""
        if self.cache:
            self.cache.clear()
