"""
Contextual Retrieval Strategy.

This module implements the main ContextualRetrievalStrategy class that
enriches chunks with document context before embedding for improved retrieval.
"""

from typing import List, Dict, Any, Optional
import logging
import asyncio

from .config import ContextualRetrievalConfig
from .context_generator import ContextGenerator
from .batch_processor import BatchProcessor
from .cost_tracker import CostTracker
from .storage import ContextualStorageManager

logger = logging.getLogger(__name__)


class ContextualRetrievalStrategy:
    """
    Contextual Retrieval: Enrich chunks with document context before embedding.
    
    This strategy:
    1. Generates contextual descriptions for each chunk using LLM
    2. Prepends context to chunk text before embedding
    3. Stores both original and contextualized versions
    4. Returns original text in retrieval results (configurable)
    
    This improves retrieval relevance by ensuring embeddings capture more
    contextual information about what each chunk is about and where it fits
    in the larger document.
    """

    def __init__(
        self,
        vector_store_service: Any,
        database_service: Any,
        llm_service: Any,
        embedding_service: Any,
        config: Optional[ContextualRetrievalConfig] = None
    ):
        """
        Initialize contextual retrieval strategy.
        
        Args:
            vector_store_service: Vector store for retrieval
            database_service: Database for chunk storage
            llm_service: LLM service for context generation
            embedding_service: Embedding service for vectorization
            config: Contextual retrieval configuration
        """
        self.vector_store = vector_store_service
        self.database = database_service
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.config = config or ContextualRetrievalConfig()
        
        # Initialize components
        self.context_generator = ContextGenerator(llm_service, self.config)
        self.cost_tracker = CostTracker(self.config)
        self.batch_processor = BatchProcessor(
            self.context_generator,
            self.cost_tracker,
            self.config
        )
        self.storage_manager = ContextualStorageManager(database_service, self.config)

    async def aindex_document(
        self,
        document: str,
        document_id: str,
        chunks: List[Dict[str, Any]],
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Index document with contextual enrichment.
        
        Args:
            document: Original document text
            document_id: Document identifier
            chunks: Pre-chunked document chunks
            document_metadata: Optional document metadata
            
        Returns:
            Indexing statistics including cost information
        """
        logger.info(f"Indexing document with contextualization: {document_id}")
        
        # Reset cost tracker
        self.cost_tracker.reset()
        
        # Prepare document context
        document_context = {
            "document_id": document_id,
            "title": document_metadata.get("title") if document_metadata else document_id
        }
        
        # Process chunks in batches to generate contexts
        contextualized_chunks = await self.batch_processor.process_chunks(
            chunks,
            document_context
        )
        
        # Generate embeddings for contextualized text
        for chunk in contextualized_chunks:
            text_to_embed = chunk.get("contextualized_text") or chunk.get("text")
            
            embedding_result = self.embedding_service.embed([text_to_embed])
            chunk["embedding"] = embedding_result.embeddings[0]
        
        # Store chunks (dual storage)
        self.storage_manager.store_chunks(contextualized_chunks)
        
        # Index in vector store (using contextualized embeddings)
        for chunk in contextualized_chunks:
            self.vector_store.index_chunk(
                chunk_id=chunk["chunk_id"],
                embedding=chunk["embedding"],
                metadata={
                    "document_id": document_id,
                    "has_context": "context_description" in chunk
                }
            )
        
        # Get cost summary
        cost_summary = self.cost_tracker.get_summary()
        
        logger.info(
            f"Indexed {len(contextualized_chunks)} chunks. "
            f"Cost: ${cost_summary['total_cost']:.4f}"
        )
        
        return {
            "document_id": document_id,
            "total_chunks": len(contextualized_chunks),
            "contextualized_chunks": sum(1 for c in contextualized_chunks if "context_description" in c),
            **cost_summary
        }

    def index_document(
        self,
        document: str,
        document_id: str,
        chunks: List[Dict[str, Any]],
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous index document wrapper.
        
        Args:
            document: Original document text
            document_id: Document identifier
            chunks: Pre-chunked document chunks
            document_metadata: Optional document metadata
            
        Returns:
            Indexing statistics
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, create a task
            return asyncio.create_task(
                self.aindex_document(document, document_id, chunks, document_metadata)
            )
        else:
            # If no loop is running, run until complete
            return loop.run_until_complete(
                self.aindex_document(document, document_id, chunks, document_metadata)
            )

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using contextualized embeddings.
        
        Args:
            query: User query
            top_k: Number of results to return
            **kwargs: Additional parameters
            
        Returns:
            List of results (with original text by default)
        """
        logger.info(f"Contextual retrieval for: {query}")
        
        # Search using contextualized embeddings
        results = self.vector_store.search(query=query, top_k=top_k)
        
        # Get chunk IDs
        chunk_ids = [r.get("chunk_id") or r.get("id") for r in results]
        
        # Retrieve chunks with desired format
        return_format = "original" if self.config.return_original_text else "contextualized"
        
        if self.config.return_context:
            return_format = "both"
        
        formatted_chunks = self.storage_manager.retrieve_chunks(
            chunk_ids,
            return_format=return_format
        )
        
        # Merge with search results (scores, etc.)
        for result, chunk in zip(results, formatted_chunks):
            result.update(chunk)
        
        logger.info(f"Retrieved {len(results)} results")
        
        return results

    @property
    def name(self) -> str:
        """Strategy name."""
        return "contextual"

    @property
    def description(self) -> str:
        """Strategy description."""
        return "Enrich chunks with document context for improved retrieval"

    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get cost tracking summary.
        
        Returns:
            Dictionary with cost statistics
        """
        return self.cost_tracker.get_summary()
