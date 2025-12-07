"""Multi-Query RAG Strategy Implementation."""

from typing import List, Dict, Any, Optional
import logging
import asyncio

from .config import MultiQueryConfig
from .variant_generator import QueryVariantGenerator
from .parallel_executor import ParallelQueryExecutor
from .deduplicator import ResultDeduplicator
from .ranker import ResultRanker

logger = logging.getLogger(__name__)


class MultiQueryRAGStrategy:
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
        vector_store_service,
        llm_service,
        embedding_service=None,
        config: Optional[MultiQueryConfig] = None
    ):
        """Initialize multi-query strategy.

        Args:
            vector_store_service: Vector store for retrieval
            llm_service: LLM service for variant generation
            embedding_service: Optional embedding service for near-duplicate detection
            config: Multi-query configuration
        """
        self.vector_store = vector_store_service
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.config = config or MultiQueryConfig()

        # Initialize components
        self.variant_generator = QueryVariantGenerator(llm_service, self.config)
        self.parallel_executor = ParallelQueryExecutor(vector_store_service, self.config)
        self.deduplicator = ResultDeduplicator(self.config, embedding_service)
        self.ranker = ResultRanker(self.config)

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

            if self.config.log_variants:
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
            if self.config.fallback_to_original:
                logger.info("Falling back to single-query retrieval")
                return await self._fallback_retrieve(query)
            else:
                raise

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
        if hasattr(self.vector_store, 'asearch'):
            return await self.vector_store.asearch(
                query=query,
                top_k=self.config.final_top_k
            )
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.vector_store.search(
                    query=query,
                    top_k=self.config.final_top_k
                )
            )

    @property
    def name(self) -> str:
        """Strategy name."""
        return "multi_query"

    @property
    def description(self) -> str:
        """Strategy description."""
        return "Generate multiple query variants and merge results for broader coverage"
