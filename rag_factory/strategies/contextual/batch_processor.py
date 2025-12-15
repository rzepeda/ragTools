"""
Batch processing for efficient context generation.

This module provides the BatchProcessor class that processes chunks in batches
for efficient LLM API usage and parallel processing.
"""

import asyncio
from typing import List, Dict, Any, Optional
import logging

from .config import ContextualRetrievalConfig
from .context_generator import ContextGenerator
from .cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Processes chunks in batches for efficient context generation.
    
    Supports both parallel and sequential batch processing with
    configurable batch sizes and concurrency limits.
    """

    def __init__(
        self,
        context_generator: ContextGenerator,
        cost_tracker: CostTracker,
        config: ContextualRetrievalConfig
    ):
        """
        Initialize batch processor.
        
        Args:
            context_generator: Context generator instance
            cost_tracker: Cost tracker instance
            config: Contextual retrieval configuration
        """
        self.context_generator = context_generator
        self.cost_tracker = cost_tracker
        self.config = config

    async def process_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process chunks in batches to generate contexts.
        
        Args:
            chunks: List of chunks to process
            document_context: Optional document-level context
            
        Returns:
            List of chunks with generated contexts
        """
        logger.info(f"Processing {len(chunks)} chunks in batches of {self.config.batch_size}")
        
        # Split into batches
        batches = self._create_batches(chunks)
        
        # Process batches
        if self.config.enable_parallel_batches:
            processed_chunks = await self._process_batches_parallel(batches, document_context)
        else:
            processed_chunks = await self._process_batches_sequential(batches, document_context)
        
        logger.info(f"Processed {len(processed_chunks)} chunks")
        
        return processed_chunks

    def _create_batches(self, chunks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Split chunks into batches.
        
        Args:
            chunks: List of chunks to batch
            
        Returns:
            List of batches
        """
        batches = []
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i:i + self.config.batch_size]
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} batches")
        return batches

    async def _process_batches_parallel(
        self,
        batches: List[List[Dict[str, Any]]],
        document_context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process batches in parallel.
        
        Args:
            batches: List of batches to process
            document_context: Optional document-level context
            
        Returns:
            List of processed chunks
        """
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
        
        async def process_batch_with_semaphore(batch):
            async with semaphore:
                return await self._process_batch(batch, document_context)
        
        # Process all batches concurrently
        results = await asyncio.gather(
            *[process_batch_with_semaphore(batch) for batch in batches],
            return_exceptions=True
        )
        
        # Flatten results
        processed_chunks = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed: {result}")
                continue
            processed_chunks.extend(result)
        
        return processed_chunks

    async def _process_batches_sequential(
        self,
        batches: List[List[Dict[str, Any]]],
        document_context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process batches sequentially.
        
        Args:
            batches: List of batches to process
            document_context: Optional document-level context
            
        Returns:
            List of processed chunks
        """
        processed_chunks = []
        
        for i, batch in enumerate(batches, 1):
            logger.info(f"Processing batch {i}/{len(batches)}")
            
            try:
                batch_results = await self._process_batch(batch, document_context)
                processed_chunks.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch {i} failed: {e}")
                # Continue with next batch if fallback enabled
                if self.config.fallback_to_no_context:
                    processed_chunks.extend(batch)  # Add chunks without context
                else:
                    raise
        
        return processed_chunks

    async def _process_batch(
        self,
        batch: List[Dict[str, Any]],
        document_context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process a single batch of chunks.
        
        Args:
            batch: Batch of chunks to process
            document_context: Optional document-level context
            
        Returns:
            List of processed chunks
        """
        processed_chunks = []
        
        # Generate contexts for all chunks in batch concurrently
        tasks = [
            self._process_single_chunk(chunk, document_context)
            for chunk in batch
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for chunk, result in zip(batch, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process chunk {chunk.get('chunk_id')}: {result}")
                
                if self.config.fallback_to_no_context:
                    processed_chunks.append(chunk)  # Add without context
                else:
                    raise result
            else:
                processed_chunks.append(result)
        
        logger.info(f"Batch processed {len(processed_chunks)} chunks out of {len(batch)}")
        return processed_chunks

    async def _process_single_chunk(
        self,
        chunk: Dict[str, Any],
        document_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a single chunk.
        
        Args:
            chunk: Chunk to process
            document_context: Optional document-level context
            
        Returns:
            Processed chunk with context
        """
        chunk_id = chunk.get("chunk_id")
        
        try:
            # Generate context
            context = await self.context_generator.generate_context(chunk, document_context)
            
            if context:
                # Create contextualized text
                original_text = chunk.get("text", "")
                contextualized_text = f"{self.config.context_prefix} {context}\n\n{original_text}"
                
                # Track cost
                input_tokens = self.context_generator._count_tokens(original_text) + 100  # Approx prompt overhead
                output_tokens = self.context_generator._count_tokens(context)
                
                cost = self.cost_tracker.calculate_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )
                
                self.cost_tracker.record_chunk_cost(
                    chunk_id=chunk_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost
                )
                
                # Update chunk with context
                chunk["context_description"] = context
                chunk["contextualized_text"] = contextualized_text
                chunk["context_token_count"] = output_tokens
                chunk["context_cost"] = cost
                chunk["context_generation_method"] = self.config.context_method
                
                logger.debug(f"Chunk {chunk_id} contextualized (cost: ${cost:.6f})")
            else:
                logger.warning(f"Chunk {chunk_id} returned without context (context was None)")
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {e}")
            
            if self.config.fallback_to_no_context:
                # Return chunk without context
                logger.warning(f"Returning chunk {chunk_id} without context due to error")
                return chunk
            else:
                raise
