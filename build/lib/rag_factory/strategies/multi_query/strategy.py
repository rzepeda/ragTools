"""Multi-Query RAG Strategy Implementation."""

from typing import List, Dict, Any, Optional, Set
import logging
import asyncio

from .config import MultiQueryConfig
from .variant_generator import QueryVariantGenerator
from .parallel_executor import ParallelQueryExecutor
from .deduplicator import ResultDeduplicator
from .ranker import ResultRanker
from ...services.dependencies import StrategyDependencies, ServiceDependency
from ..base import IRAGStrategy
from ...factory import register_rag_strategy as register_strategy

logger = logging.getLogger(__name__)


@register_strategy("MultiQueryRAGStrategy")
class MultiQueryRAGStrategy(IRAGStrategy):
    """Multi-Query RAG: Generate multiple query variants and merge results.

    This strategy:
    1. Generates 3-5 query variants using LLM
    2. Executes all variants in parallel
    3. Deduplicates results across variants
    4. Ranks merged results using RRF or other strategies
    
    This improves retrieval coverage by capturing different perspectives
    and phrasings of the user's intent.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dependencies: StrategyDependencies
    ):
        """Initialize multi-query strategy.

        Args:
            config: Strategy configuration dictionary
            dependencies: Injected service dependencies
        """
        # Initialize base class (validates dependencies)
        super().__init__(config, dependencies)
        
        # Parse configuration
        self.strategy_config = config if isinstance(config, MultiQueryConfig) else MultiQueryConfig(**config)

        # Initialize components
        self.variant_generator = QueryVariantGenerator(self.deps.llm_service, self.strategy_config)
        # Use database_service as vector store
        self.parallel_executor = ParallelQueryExecutor(self.deps.database_service, self.strategy_config)
        self.deduplicator = ResultDeduplicator(self.strategy_config, self.deps.embedding_service)
        self.ranker = ResultRanker(self.strategy_config)
    
    def requires_services(self) -> Set[ServiceDependency]:
        """Declare required services."""
        return {ServiceDependency.LLM, ServiceDependency.EMBEDDING, ServiceDependency.DATABASE}

    async def aretrieve(
        self,
        query: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Async retrieve with multi-query strategy.

        Args:
            query: User query
            **kwargs: Additional parameters (currently unused)

        Returns:
            List of ranked results
        """
        logger.info(f"Multi-query retrieval for: {query}")

        try:
            # Step 1: Generate query variants
            variants = await self.variant_generator.generate_variants(query)

            if self.strategy_config.log_variants:
                logger.info(f"Query variants: {variants}")

            # Step 2: Execute variants in parallel
            query_results = await self.parallel_executor.execute_queries(variants)

            # Step 3: Deduplicate results
            deduplicated = self.deduplicator.deduplicate(query_results)

            # Step 4: Rank merged results
            ranked = self.ranker.rank(deduplicated)

            logger.info(f"Multi-query retrieval complete: {len(ranked)} results")

            return ranked

        except Exception as e:
            logger.error(f"Multi-query retrieval failed: {e}")

            # Fallback to single query
            if self.strategy_config.fallback_to_original:
                logger.info("Falling back to single-query retrieval")
                return await self._fallback_retrieve(query)
            else:
                raise
    
    def prepare_data(self, documents: List[Dict[str, Any]]):
        """Prepare and chunk documents for retrieval."""
        raise NotImplementedError("prepare_data not yet implemented for MultiQueryRAGStrategy")
    
    def process_query(self, query: str, context):
        """Process query with context."""
        raise NotImplementedError("process_query not yet implemented for MultiQueryRAGStrategy")

    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Synchronous retrieve (wrapper for async).

        Args:
            query: User query
            **kwargs: Additional parameters

        Returns:
            List of ranked results
        """
        # Run async retrieve in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create new task
                # Note: This will return a coroutine, not the actual results
                # Caller should await it
                logger.warning(
                    "retrieve() called from async context, "
                    "consider using aretrieve() instead"
                )
                return asyncio.create_task(self.aretrieve(query, **kwargs))
            else:
                # Run in event loop
                return loop.run_until_complete(self.aretrieve(query, **kwargs))
        except RuntimeError:
            # No event loop, create new one
            return asyncio.run(self.aretrieve(query, **kwargs))

    async def _fallback_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Fallback to single query retrieval.
        
        Args:
            query: User query
            
        Returns:
            List of search results
        """
        # Use database_service for vector search
        if self.deps.database_service:
            # Check if database service has async search
            if hasattr(self.deps.database_service, 'asearch_similar'):
                return await self.deps.database_service.asearch_similar(
                    query=query,
                    top_k=self.strategy_config.final_top_k
                )
            elif hasattr(self.deps.database_service, 'search_similar'):
                # Fallback to sync search
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self.deps.database_service.search_similar(
                        query=query,
                        top_k=self.strategy_config.final_top_k
                    )
                )
        
        # If no database service, return empty results
        logger.warning("No database service available for fallback retrieval")
        return []

    @property
    def name(self) -> str:
        """Strategy name."""
        return "multi_query"

    @property
    def description(self) -> str:
        """Strategy description."""
        return "Generate multiple query variants and merge results for broader coverage"
