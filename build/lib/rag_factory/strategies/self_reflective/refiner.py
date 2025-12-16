"""Query refinement logic for self-reflective RAG strategy."""

from typing import List, Dict, Any
import logging
import re

from .models import QueryRefinement, RefinementStrategy, Grade
from rag_factory.services.llm import LLMService
from rag_factory.services.llm.base import Message, MessageRole

logger = logging.getLogger(__name__)


class QueryRefiner:
    """Refines queries based on grading feedback.
    
    Analyzes poor grades to identify gaps and generates refined
    queries that address those gaps using various strategies.
    """

    def __init__(self, llm_service: LLMService, config: Dict[str, Any]):
        """Initialize the query refiner.
        
        Args:
            llm_service: LLM service for refinement
            config: Configuration dictionary
        """
        self.llm_service = llm_service
        self.config = config
        self.default_strategy = config.get(
            "refinement_strategy",
            RefinementStrategy.REFORMULATION
        )

    def refine_query(
        self,
        original_query: str,
        grades: List[Grade],
        iteration: int,
        previous_refinements: List[QueryRefinement] = None
    ) -> QueryRefinement:
        """Generate refined query based on grading feedback.

        Args:
            original_query: Original user query
            grades: Grades from previous retrieval
            iteration: Current iteration number
            previous_refinements: Previous refinement attempts

        Returns:
            QueryRefinement with new query
        """
        logger.info(f"Refining query (iteration {iteration}): {original_query}")

        # Analyze grades to identify gaps
        gaps = self._identify_gaps(grades)

        # Generate refined query
        prompt = self._build_refinement_prompt(
            original_query,
            gaps,
            previous_refinements or []
        )

        try:
            messages = [Message(role=MessageRole.USER, content=prompt)]
            response = self.llm_service.complete(
                messages,
                temperature=0.3,  # Some creativity for refinement
                max_tokens=200
            )

            # Parse refined query
            refined_query, strategy, reasoning = self._parse_refinement(
                response.content
            )

            refinement = QueryRefinement(
                original_query=original_query,
                refined_query=refined_query,
                strategy=strategy,
                reasoning=reasoning,
                iteration=iteration
            )

            logger.info(
                f"Refined query: {refined_query} "
                f"(strategy: {strategy.value})"
            )

            return refinement
            
        except Exception as e:
            logger.error(f"Error in query refinement: {e}")
            # Fallback: return original query with slight modification
            return QueryRefinement(
                original_query=original_query,
                refined_query=f"{original_query} (refined)",
                strategy=self.default_strategy,
                reasoning=f"Refinement failed: {str(e)}",
                iteration=iteration
            )

    def _identify_gaps(self, grades: List[Grade]) -> List[str]:
        """Identify what's missing or weak in results.
        
        Args:
            grades: List of grades to analyze
            
        Returns:
            List of identified gaps
        """
        if not grades:
            return ["No results to analyze"]
            
        gaps = []

        # Analyze grade reasoning
        for grade in grades:
            if grade.score < 4.0:
                # Extract what's missing from reasoning
                reasoning_lower = grade.reasoning.lower()

                if "incomplete" in reasoning_lower or "missing" in reasoning_lower:
                    gaps.append("Results are incomplete")
                if "not relevant" in reasoning_lower or "irrelevant" in reasoning_lower:
                    gaps.append("Results not relevant enough")
                if "vague" in reasoning_lower or "unclear" in reasoning_lower:
                    gaps.append("Query may be too vague")
                if "specific" in reasoning_lower:
                    gaps.append("Query may need more specificity")

        # Check completeness scores
        avg_completeness = sum(g.completeness for g in grades) / len(grades)
        if avg_completeness < 0.7:
            gaps.append("Results lack completeness")

        # Check relevance scores
        avg_relevance = sum(g.relevance for g in grades) / len(grades)
        if avg_relevance < 0.7:
            gaps.append("Results lack relevance")

        return list(set(gaps))  # Deduplicate

    def _build_refinement_prompt(
        self,
        original_query: str,
        gaps: List[str],
        previous_refinements: List[QueryRefinement]
    ) -> str:
        """Build prompt for query refinement.
        
        Args:
            original_query: Original query
            gaps: Identified gaps
            previous_refinements: Previous attempts
            
        Returns:
            Refinement prompt string
        """
        previous_attempts = ""
        if previous_refinements:
            previous_attempts = "\n\nPrevious refinement attempts:\n"
            for ref in previous_refinements:
                previous_attempts += f"- {ref.refined_query} (didn't work)\n"

        gaps_text = "\n".join(f"- {gap}" for gap in gaps) if gaps else "- No specific gaps identified"

        prompt = f"""The search results for the following query were not satisfactory.

Original Query: "{original_query}"

Issues with results:
{gaps_text}{previous_attempts}

Generate a refined search query that addresses these issues. The refined query should:
1. Be different from the original and any previous attempts
2. Address the identified gaps
3. Maintain the original intent
4. Be clear and specific

Provide:
1. Refined query
2. Strategy used (expansion/reformulation/decomposition/specificity/context_addition)
3. Brief reasoning

Format:
Refined Query: [your refined query]
Strategy: [strategy name]
Reasoning: [why this refinement should work better]
"""

        return prompt

    def _parse_refinement(
        self,
        response: str
    ) -> tuple[str, RefinementStrategy, str]:
        """Parse refinement from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Tuple of (refined_query, strategy, reasoning)
        """
        # Extract refined query
        query_match = re.search(r'Refined Query:\s*(.+?)(?=\n|$)', response)
        refined_query = query_match.group(1).strip() if query_match else response.strip()

        # Extract strategy
        strategy_match = re.search(
            r'Strategy:\s*(expansion|reformulation|decomposition|specificity|context_addition)',
            response,
            re.IGNORECASE
        )
        strategy_str = strategy_match.group(1).lower() if strategy_match else "reformulation"

        try:
            strategy = RefinementStrategy(strategy_str)
        except ValueError:
            strategy = RefinementStrategy.REFORMULATION

        # Extract reasoning
        reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=\n\n|$)', response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"

        return refined_query, strategy, reasoning
