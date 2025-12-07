"""Query Variant Generator for Multi-Query RAG Strategy."""

from typing import List
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import MultiQueryConfig
from .prompts import get_variant_prompt

logger = logging.getLogger(__name__)


class QueryVariantGenerator:
    """Generates query variants using LLM.
    
    This class uses an LLM to generate multiple variants of a user query,
    capturing different phrasings, perspectives, and specificity levels.
    """

    def __init__(self, llm_service, config: MultiQueryConfig):
        """Initialize variant generator.

        Args:
            llm_service: LLM service for generation (must have agenerate or complete method)
            config: Multi-query configuration
        """
        self.llm_service = llm_service
        self.config = config

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def generate_variants(self, query: str) -> List[str]:
        """Generate query variants from original query.

        Args:
            query: Original user query

        Returns:
            List of query variants (including original if configured)
            
        Raises:
            Exception: If variant generation fails and fallback is disabled
        """
        logger.info(f"Generating {self.config.num_variants} variants for query: {query}")

        try:
            # Build prompt for variant generation
            prompt = self._build_variant_prompt(query)

            # Generate variants using LLM
            response = await self._call_llm(prompt)

            # Parse variants from response
            variants = self._parse_variants(response, query)

            # Validate variants
            variants = self._validate_variants(variants, query)

            # Include original query if configured
            if self.config.include_original and query not in variants:
                variants.insert(0, query)

            if self.config.log_variants:
                logger.info(f"Generated {len(variants)} variants: {variants}")

            return variants

        except Exception as e:
            logger.error(f"Error generating variants: {e}")

            # Fallback to original query
            if self.config.fallback_to_original:
                logger.info("Falling back to original query")
                return [query]
            else:
                raise

    def _build_variant_prompt(self, query: str) -> str:
        """Build prompt for variant generation based on configured types.
        
        Args:
            query: Original user query
            
        Returns:
            Formatted prompt string
        """
        return get_variant_prompt(
            query=query,
            num_variants=self.config.num_variants,
            variant_types=self.config.variant_types,
            use_type_specific=False  # Use default prompt for mixed types
        )

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM service with timeout.
        
        Args:
            prompt: Prompt to send to LLM
            
        Returns:
            LLM response text
        """
        # Check if LLM service has async method
        if hasattr(self.llm_service, 'agenerate'):
            response = await self.llm_service.agenerate(
                prompt=prompt,
                temperature=self.config.llm_temperature,
                max_tokens=500,
                timeout=self.config.variant_generation_timeout
            )
            return response.text if hasattr(response, 'text') else str(response)
        elif hasattr(self.llm_service, 'complete'):
            # Fallback to sync method (will be wrapped in executor)
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.llm_service.complete(
                    prompt=prompt,
                    temperature=self.config.llm_temperature,
                    max_tokens=500
                )
            )
            return response.content if hasattr(response, 'content') else str(response)
        else:
            raise ValueError("LLM service must have 'agenerate' or 'complete' method")

    def _parse_variants(self, response: str, original_query: str) -> List[str]:
        """Parse variants from LLM response.
        
        Args:
            response: LLM response text
            original_query: Original query for reference
            
        Returns:
            List of parsed variant strings
        """
        lines = response.strip().split("\n")

        variants = []
        for line in lines:
            # Clean up line (remove numbering, bullets, etc.)
            cleaned = line.strip()
            # Remove common prefixes: "1.", "1)", "-", "*", "•"
            cleaned = cleaned.lstrip("0123456789.-)*• ")

            if cleaned and len(cleaned) > 5:  # Skip very short lines
                variants.append(cleaned)

        # Limit to requested number
        return variants[:self.config.num_variants]

    def _validate_variants(self, variants: List[str], original_query: str) -> List[str]:
        """Validate generated variants.
        
        Filters out:
        - Empty or very short variants
        - Exact duplicates
        - Case-insensitive duplicates
        
        Args:
            variants: List of variant strings to validate
            original_query: Original query for reference
            
        Returns:
            List of validated variant strings
        """
        validated = []
        seen_lower = set()

        for variant in variants:
            # Check not empty
            if not variant or len(variant.strip()) < 5:
                continue

            # Check not duplicate (case-insensitive)
            variant_lower = variant.lower()
            if variant_lower in seen_lower:
                continue

            validated.append(variant)
            seen_lower.add(variant_lower)

        return validated
