"""Self-Reflective RAG Strategy Implementation."""

from typing import List, Dict, Any, Optional
import logging
import time

from .grader import ResultGrader
from .refiner import QueryRefiner
from .models import RetrievalAttempt, QueryRefinement
from ...factory import register_rag_strategy as register_strategy

logger = logging.getLogger(__name__)


@register_strategy("SelfReflectiveRAGStrategy")
class SelfReflectiveRAGStrategy:
    """Self-Reflective RAG: Grades results and retries with refined queries.

    Evaluates retrieved results, and if quality is below threshold,
    refines the query and retries the search.
    
    This is a wrapper/decorator pattern that enhances any base retrieval
    strategy with self-reflection capabilities.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Any] = None
    ):
        """Initialize self-reflective RAG strategy.
        
        Args:
            config: Configuration dictionary
            dependencies: StrategyDependencies with required services
        """
        # Store dependencies
        self.deps = dependencies
        
        # Get services from dependencies
        self.llm_service = dependencies.llm_service if dependencies else None
        # Base strategy can be created from database service
        self.base_strategy = dependencies.database_service if dependencies else None

        # Initialize components
        config = config or {}
        self.grader = ResultGrader(self.llm_service, config) if self.llm_service else None
        self.refiner = QueryRefiner(self.llm_service, config) if self.llm_service else None

        # Configuration
        self.grade_threshold = config.get("grade_threshold", 4.0)
        self.max_retries = config.get("max_retries", 2)
        self.timeout_seconds = config.get("timeout_seconds", 10)

    async def retrieve(
        self,
        query: str,
        context: Any,  # RetrievalContext
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve with self-reflection and retry.

        Args:
            query: User query
            top_k: Number of results
            **kwargs: Additional parameters for base strategy

        Returns:
            List of retrieved chunks (best results across all attempts)
        """
        logger.info(f"Self-reflective retrieval for: {query}")

        start_time = time.time()
        attempts: List[RetrievalAttempt] = []
        refinements: List[QueryRefinement] = []

        current_query = query
        iteration = 0

        while iteration <= self.max_retries:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                logger.warning(f"Timeout reached after {elapsed:.1f}s")
                break

            iteration += 1
            logger.info(f"Attempt {iteration}: {current_query}")

            attempt_start = time.time()

            # Perform retrieval with base strategy
            # Use embedding service to embed query, then search
            if self.deps and self.deps.embedding_service:
                query_embedding = await self.deps.embedding_service.embed(current_query)
                results = await self.base_strategy.search_chunks(
                    query_embedding=query_embedding,
                    top_k=top_k
                )
            else:
                # Fallback: return empty results
                logger.warning("No embedding service available for retrieval")
                results = []

            # Convert results to dict format if needed
            results = self._normalize_results(results)

            # Grade results
            grades = self.grader.grade_results(current_query, results)

            # Calculate average grade
            avg_grade = sum(g.score for g in grades) / len(grades) if grades else 0.0

            attempt_latency = (time.time() - attempt_start) * 1000

            # Record attempt
            attempt = RetrievalAttempt(
                attempt_number=iteration,
                query=current_query,
                results=results,
                grades=grades,
                average_grade=avg_grade,
                timestamp=time.time(),
                latency_ms=attempt_latency
            )

            attempts.append(attempt)

            logger.info(
                f"Attempt {iteration} complete. "
                f"Average grade: {avg_grade:.2f}, "
                f"Latency: {attempt_latency:.0f}ms"
            )

            # Check if results are good enough
            if avg_grade >= self.grade_threshold:
                logger.info(
                    f"Results meet threshold ({avg_grade:.2f} >= {self.grade_threshold}). "
                    f"Stopping."
                )
                break

            # If not last iteration, refine query and retry
            if iteration < self.max_retries:
                logger.info(
                    f"Results below threshold ({avg_grade:.2f} < {self.grade_threshold}). "
                    f"Refining query..."
                )

                refinement = self.refiner.refine_query(
                    original_query=query,
                    grades=grades,
                    iteration=iteration,
                    previous_refinements=refinements
                )

                refinements.append(refinement)
                attempt.refinement = refinement

                # Update current query
                current_query = refinement.refined_query

                # Prevent infinite loops (same query)
                if current_query == query:
                    logger.warning("Refined query same as original. Stopping.")
                    break
            else:
                logger.info("Max retries reached.")

        # Aggregate results across all attempts
        final_results = self._aggregate_results(attempts, top_k)

        total_latency = (time.time() - start_time) * 1000

        logger.info(
            f"Self-reflective retrieval complete. "
            f"{len(attempts)} attempts, "
            f"{total_latency:.0f}ms total"
        )

        # Add metadata
        for result in final_results:
            result["strategy"] = "self_reflective"
            result["total_attempts"] = len(attempts)
            result["refinements"] = [r.dict() for r in refinements]

        return final_results

    def _normalize_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Normalize results to dict format.
        
        Args:
            results: Results from base strategy (may be Chunk objects or dicts)
            
        Returns:
            List of result dictionaries
        """
        normalized = []
        
        for i, result in enumerate(results):
            if isinstance(result, dict):
                normalized.append(result)
            else:
                # Assume it's a Chunk object
                normalized.append({
                    "chunk_id": getattr(result, "chunk_id", f"chunk_{i}"),
                    "text": getattr(result, "text", ""),
                    "score": getattr(result, "score", 0.0),
                    "metadata": getattr(result, "metadata", {})
                })
        
        return normalized

    def _aggregate_results(
        self,
        attempts: List[RetrievalAttempt],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Aggregate results from multiple attempts.

        Strategy:
        1. Collect all results across attempts
        2. Deduplicate by chunk_id
        3. For each chunk, use highest grade
        4. Sort by grade * similarity_score
        5. Return top_k

        Args:
            attempts: All retrieval attempts
            top_k: Number of results to return
            
        Returns:
            Aggregated and deduplicated results
        """
        all_results = {}  # chunk_id -> (result, grade, attempt_number)

        for attempt in attempts:
            for result, grade in zip(attempt.results, attempt.grades):
                chunk_id = result.get("chunk_id")

                if chunk_id:
                    # Keep result with highest grade
                    if chunk_id not in all_results or grade.score > all_results[chunk_id][1].score:
                        all_results[chunk_id] = (result, grade, attempt.attempt_number)

        # Build final result list
        final_results = []
        for chunk_id, (result, grade, attempt_num) in all_results.items():
            # Combine similarity score and grade
            original_score = result.get("score", 0.5)
            combined_score = (original_score + grade.score / 5.0) / 2.0

            final_results.append({
                **result,
                "grade": grade.score,
                "grade_level": grade.level.name,
                "grade_reasoning": grade.reasoning,
                "combined_score": combined_score,
                "retrieval_attempt": attempt_num
            })

        # Sort by combined score
        final_results.sort(key=lambda x: x["combined_score"], reverse=True)

        return final_results[:top_k]

    @property
    def name(self) -> str:
        """Strategy name."""
        return "self_reflective"

    @property
    def description(self) -> str:
        """Strategy description."""
        return "Self-correcting retrieval with result grading and query refinement"
