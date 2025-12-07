"""Result grading logic for self-reflective RAG strategy."""

from typing import List, Dict, Any
import logging
import time
import re

from .models import Grade, GradeLevel
from rag_factory.services.llm import LLMService
from rag_factory.services.llm.base import Message, MessageRole

logger = logging.getLogger(__name__)


class ResultGrader:
    """Grades retrieved results for quality and relevance.
    
    Uses an LLM to evaluate retrieved chunks on a 1-5 scale,
    assessing relevance, completeness, and overall quality.
    """

    def __init__(self, llm_service: LLMService, config: Dict[str, Any]):
        """Initialize the result grader.
        
        Args:
            llm_service: LLM service for grading
            config: Configuration dictionary
        """
        self.llm_service = llm_service
        self.config = config
        self.grading_prompt_template = config.get(
            "grading_prompt",
            self._default_grading_prompt()
        )
        self.batch_size = config.get("batch_grading_size", 5)

    def grade_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Grade]:
        """Grade retrieved results for relevance and quality.

        Args:
            query: Original user query
            results: Retrieved chunks to grade

        Returns:
            List of Grade objects
        """
        if not results:
            logger.warning("No results to grade")
            return []
            
        logger.info(f"Grading {len(results)} results for query: {query}")

        # Batch grading for efficiency
        grades = []
        for i in range(0, len(results), self.batch_size):
            batch = results[i:i + self.batch_size]
            batch_grades = self._grade_batch(query, batch)
            grades.extend(batch_grades)

        avg_grade = sum(g.score for g in grades) / len(grades) if grades else 0
        logger.info(f"Grading complete. Average grade: {avg_grade:.2f}")

        return grades

    def _grade_batch(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Grade]:
        """Grade a batch of results together.
        
        Args:
            query: User query
            results: Batch of results to grade
            
        Returns:
            List of Grade objects
        """
        start_time = time.time()

        # Build grading prompt
        prompt = self._build_grading_prompt(query, results)

        # Call LLM for grading
        messages = [Message(role=MessageRole.USER, content=prompt)]
        
        try:
            response = self.llm_service.complete(
                messages,
                temperature=0.1,  # Low temperature for consistent grading
                max_tokens=1000
            )

            # Parse grades from response
            grades = self._parse_grades(response.content, results)

            latency = (time.time() - start_time) * 1000
            logger.info(f"Batch grading completed in {latency:.0f}ms")

            return grades
            
        except Exception as e:
            logger.error(f"Error in batch grading: {e}")
            # Return fallback grades
            return [
                Grade(
                    chunk_id=result.get("chunk_id", f"chunk_{i}"),
                    score=3.0,
                    level=GradeLevel.FAIR,
                    relevance=0.5,
                    completeness=0.5,
                    reasoning=f"Grading failed: {str(e)}"
                )
                for i, result in enumerate(results)
            ]

    def _build_grading_prompt(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for grading results.
        
        Args:
            query: User query
            results: Results to grade
            
        Returns:
            Grading prompt string
        """
        results_text = ""
        for i, result in enumerate(results, 1):
            text = result.get("text", "")[:500]  # Limit text length
            results_text += f"\n{i}. {text}\n"

        prompt = f"""You are evaluating search results for relevance and quality.

Query: "{query}"

Retrieved Results:
{results_text}

For each result, grade it on a scale of 1-5:
- 5: Perfect match - highly relevant, complete answer
- 4: Good match - relevant, mostly complete
- 3: Partial match - some relevance, incomplete
- 2: Poor match - low relevance
- 1: No match - irrelevant

For each result, provide:
1. Grade (1-5)
2. Relevance score (0-1)
3. Completeness score (0-1)
4. Brief reasoning

Format your response as:
Result 1:
Grade: 4
Relevance: 0.8
Completeness: 0.7
Reasoning: [your reasoning]

Result 2:
...
"""

        return prompt

    def _parse_grades(
        self,
        response: str,
        results: List[Dict[str, Any]]
    ) -> List[Grade]:
        """Parse grades from LLM response.
        
        Args:
            response: LLM response text
            results: Original results
            
        Returns:
            List of Grade objects
        """
        grades = []

        # Split into result sections
        sections = re.split(r'Result \d+:', response)
        sections = [s.strip() for s in sections if s.strip()]

        for i, section in enumerate(sections[:len(results)]):
            try:
                # Extract grade
                grade_match = re.search(r'Grade:\s*(\d+(?:\.\d+)?)', section)
                grade_value = float(grade_match.group(1)) if grade_match else 3.0

                # Extract relevance
                rel_match = re.search(r'Relevance:\s*(\d+(?:\.\d+)?)', section)
                relevance = float(rel_match.group(1)) if rel_match else 0.5

                # Extract completeness
                comp_match = re.search(r'Completeness:\s*(\d+(?:\.\d+)?)', section)
                completeness = float(comp_match.group(1)) if comp_match else 0.5

                # Extract reasoning
                reas_match = re.search(r'Reasoning:\s*(.+?)(?=\n\n|$)', section, re.DOTALL)
                reasoning = reas_match.group(1).strip() if reas_match else "No reasoning provided"

                # Determine grade level
                if grade_value >= 4.5:
                    level = GradeLevel.EXCELLENT
                elif grade_value >= 3.5:
                    level = GradeLevel.GOOD
                elif grade_value >= 2.5:
                    level = GradeLevel.FAIR
                elif grade_value >= 1.5:
                    level = GradeLevel.POOR
                else:
                    level = GradeLevel.IRRELEVANT

                grade = Grade(
                    chunk_id=results[i].get("chunk_id", f"chunk_{i}"),
                    score=min(5.0, max(1.0, grade_value)),
                    level=level,
                    relevance=min(1.0, max(0.0, relevance)),
                    completeness=min(1.0, max(0.0, completeness)),
                    reasoning=reasoning
                )

                grades.append(grade)

            except Exception as e:
                logger.warning(f"Error parsing grade for result {i}: {e}")
                # Fallback grade
                grades.append(Grade(
                    chunk_id=results[i].get("chunk_id", f"chunk_{i}"),
                    score=3.0,
                    level=GradeLevel.FAIR,
                    relevance=0.5,
                    completeness=0.5,
                    reasoning="Failed to parse grade"
                ))

        # Fill remaining results with default grades
        while len(grades) < len(results):
            i = len(grades)
            grades.append(Grade(
                chunk_id=results[i].get("chunk_id", f"chunk_{i}"),
                score=3.0,
                level=GradeLevel.FAIR,
                relevance=0.5,
                completeness=0.5,
                reasoning="No grade provided"
            ))

        return grades

    def _default_grading_prompt(self) -> str:
        """Default grading prompt template.
        
        Returns:
            Default prompt string
        """
        return "Grade the following search results..."
