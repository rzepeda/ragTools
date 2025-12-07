"""
Context generation using LLM.

This module provides the ContextGenerator class that uses an LLM to generate
contextual descriptions for document chunks.
"""

from typing import Dict, Any, Optional
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import ContextualRetrievalConfig, ContextSource
from .prompts import get_prompt_template, build_contextual_info

logger = logging.getLogger(__name__)


class ContextGenerator:
    """
    Generates contextual descriptions for chunks using LLM.
    
    Uses an LLM service to create concise contextual descriptions that
    help understand what each chunk is about and where it fits in the
    larger document.
    """

    def __init__(self, llm_service: Any, config: ContextualRetrievalConfig):
        """
        Initialize context generator.
        
        Args:
            llm_service: LLM service for context generation
            config: Contextual retrieval configuration
        """
        self.llm_service = llm_service
        self.config = config

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def generate_context(
        self,
        chunk: Dict[str, Any],
        document_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Generate contextual description for a chunk.
        
        Args:
            chunk: Chunk dict with 'text' and 'metadata' keys
            document_context: Optional document-level context information
                (e.g., title, preceding_text, document_summary)
                
        Returns:
            Generated context string, or None if skipped/failed
        """
        if not self.config.enable_contextualization:
            return None
        
        # Check if chunk should be contextualized
        if not self._should_contextualize(chunk):
            logger.debug(f"Skipping contextualization for chunk {chunk.get('chunk_id')}")
            return None
        
        try:
            # Build context generation prompt
            prompt = self._build_context_prompt(chunk, document_context)
            
            # Generate context using LLM
            response = await self.llm_service.agenerate(
                prompt=prompt,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )
            
            context = response.text.strip()
            
            # Validate context length
            context_tokens = self._count_tokens(context)
            if context_tokens < self.config.context_length_min:
                logger.warning(f"Generated context too short: {context_tokens} tokens")
                return None
            elif context_tokens > self.config.context_length_max:
                # Truncate if too long
                context = self._truncate_context(context, self.config.context_length_max)
            
            logger.debug(f"Generated context ({context_tokens} tokens): {context[:100]}...")
            
            return context
            
        except Exception as e:
            logger.error(f"Error generating context for chunk {chunk.get('chunk_id')}: {e}")
            
            if self.config.fallback_to_no_context:
                return None
            else:
                raise

    def _should_contextualize(self, chunk: Dict[str, Any]) -> bool:
        """
        Determine if chunk should be contextualized.
        
        Args:
            chunk: Chunk to evaluate
            
        Returns:
            True if chunk should be contextualized, False otherwise
        """
        # Check contextualize_all setting
        if not self.config.contextualize_all:
            return False
        
        # Check minimum chunk size
        chunk_text = chunk.get("text", "")
        token_count = self._count_tokens(chunk_text)
        
        if token_count < self.config.min_chunk_size_for_context:
            logger.debug(f"Chunk too small ({token_count} tokens), skipping")
            return False
        
        # Check if code block (simple heuristic)
        if self.config.skip_code_blocks:
            if chunk_text.strip().startswith("```") or "def " in chunk_text or "class " in chunk_text:
                logger.debug("Detected code block, skipping")
                return False
        
        # Check if table
        if self.config.skip_tables:
            if "|" in chunk_text and chunk_text.count("|") > 5:
                logger.debug("Detected table, skipping")
                return False
        
        return True

    def _build_context_prompt(
        self,
        chunk: Dict[str, Any],
        document_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build prompt for context generation.
        
        Args:
            chunk: Chunk to generate context for
            document_context: Optional document-level context
            
        Returns:
            Formatted prompt string
        """
        chunk_text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        
        # Gather contextual information based on sources
        document_title = None
        section_hierarchy = None
        preceding_text = None
        document_id = None
        
        if ContextSource.DOCUMENT_METADATA in self.config.context_sources:
            if document_context and "title" in document_context:
                document_title = document_context["title"]
            elif "document_id" in metadata:
                document_id = metadata["document_id"]
        
        if ContextSource.SECTION_HIERARCHY in self.config.context_sources:
            section_hierarchy = metadata.get("section_hierarchy")
        
        if ContextSource.SURROUNDING_CHUNKS in self.config.context_sources:
            if document_context and "preceding_text" in document_context:
                preceding_text = document_context["preceding_text"]
        
        # Build contextual info string
        contextual_info = build_contextual_info(
            document_title=document_title,
            section_hierarchy=section_hierarchy,
            preceding_text=preceding_text,
            document_id=document_id
        )
        
        # Get appropriate prompt template
        chunk_type = metadata.get("chunk_type", "default")
        template = get_prompt_template(chunk_type)
        
        # Format prompt
        prompt = template.format(
            contextual_info=contextual_info,
            chunk_text=chunk_text,
            min_tokens=self.config.context_length_min,
            max_tokens=self.config.context_length_max
        )
        
        return prompt

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Uses a simplified approximation: 1 token ≈ 4 characters.
        In production, this should use tiktoken or similar.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate token count
        """
        return len(text) // 4

    def _truncate_context(self, context: str, max_tokens: int) -> str:
        """
        Truncate context to max tokens.
        
        Attempts to truncate at sentence boundary if possible.
        
        Args:
            context: Context to truncate
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated context
        """
        max_chars = max_tokens * 4
        if len(context) <= max_chars:
            return context
        
        # Truncate at sentence boundary if possible
        truncated = context[:max_chars]
        last_period = truncated.rfind(".")
        if last_period > max_chars * 0.7:  # At least 70% of desired length
            return truncated[:last_period + 1]
        
        return truncated + "..."
