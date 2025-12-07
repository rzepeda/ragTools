"""Parallel Query Executor for Multi-Query RAG Strategy."""

import asyncio
from typing import List, Dict, Any
import logging
import time

from .config import MultiQueryConfig

logger = logging.getLogger(__name__)


class ParallelQueryExecutor:
    """Executes multiple query variants in parallel.
    
    This class handles concurrent execution of query variants with timeout
    handling, error recovery, and performance tracking.
    """

    def __init__(self, vector_store_service, config: MultiQueryConfig):
        """Initialize parallel executor.

        Args:
            vector_store_service: Vector store for querying
            config: Multi-query configuration
        """
        self.vector_store = vector_store_service
        self.config = config

    async def execute_queries(self, query_variants: List[str]) -> List[Dict[str, Any]]:
        """Execute all query variants in parallel.

        Args:
            query_variants: List of query variants to execute

        Returns:
            List of dicts with query results and metadata
            
        Raises:
            ValueError: If fewer than min_successful_queries succeed
        """
        logger.info(f"Executing {len(query_variants)} queries in parallel")

        # Create tasks for all queries
        tasks = []
        for i, query in enumerate(query_variants):
            task = self._execute_single_query(query, variant_index=i)
            tasks.append(task)

        # Execute all queries concurrently with timeout
        start_time = time.time()

        results = await asyncio.gather(*tasks, return_exceptions=True)

        execution_time = time.time() - start_time
        logger.info(f"Parallel execution completed in {execution_time:.2f}s")

        # Process results (handle exceptions)
        processed_results = []
        successful_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Query variant {i} failed: {result}")
                # Add empty result
                processed_results.append({
                    "query": query_variants[i],
                    "variant_index": i,
                    "results": [],
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
                successful_count += 1

        # Check minimum successful queries
        if successful_count < self.config.min_successful_queries:
            raise ValueError(
                f"Only {successful_count} queries succeeded, "
                f"minimum required: {self.config.min_successful_queries}"
            )

        logger.info(f"{successful_count}/{len(query_variants)} queries succeeded")

        return processed_results

    async def _execute_single_query(
        self,
        query: str,
        variant_index: int
    ) -> Dict[str, Any]:
        """Execute a single query with timeout.

        Args:
            query: Query string
            variant_index: Index of this variant

        Returns:
            Dict with query results and metadata
            
        Raises:
            asyncio.TimeoutError: If query times out
        """
        logger.debug(f"Executing variant {variant_index}: {query}")

        start_time = time.time()

        try:
            # Execute query with timeout
            results = await asyncio.wait_for(
                self._query_vector_store(query),
                timeout=self.config.query_timeout
            )

            execution_time = time.time() - start_time

            return {
                "query": query,
                "variant_index": variant_index,
                "results": results,
                "success": True,
                "execution_time": execution_time,
                "num_results": len(results)
            }

        except asyncio.TimeoutError:
            logger.warning(
                f"Query variant {variant_index} timed out after "
                f"{self.config.query_timeout}s"
            )
            raise
        except Exception as e:
            logger.error(f"Error executing variant {variant_index}: {e}")
            raise

    async def _query_vector_store(self, query: str) -> List[Dict[str, Any]]:
        """Query vector store (async wrapper).
        
        Args:
            query: Query string
            
        Returns:
            List of search results
        """
        # If vector store has async support, use it directly
        if hasattr(self.vector_store, 'asearch'):
            return await self.vector_store.asearch(
                query=query,
                top_k=self.config.top_k_per_variant
            )
        elif hasattr(self.vector_store, 'search'):
            # Otherwise, run in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.vector_store.search(
                    query=query,
                    top_k=self.config.top_k_per_variant
                )
            )
        else:
            raise ValueError(
                "Vector store must have 'asearch' or 'search' method"
            )
