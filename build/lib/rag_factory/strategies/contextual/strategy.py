"""
Contextual Retrieval Strategy.

This module implements the main ContextualRetrievalStrategy class that
enriches chunks with document context before embedding for improved retrieval.
"""

from typing import List, Dict, Any, Optional, Set
import logging
import asyncio

from .config import ContextualRetrievalConfig
from .context_generator import ContextGenerator
from .batch_processor import BatchProcessor
from .cost_tracker import CostTracker
from .storage import ContextualStorageManager
from ...services.dependencies import StrategyDependencies, ServiceDependency
from ..base import IRAGStrategy
from ...factory import register_rag_strategy as register_strategy
from ...core.capabilities import IndexCapability

logger = logging.getLogger(__name__)


@register_strategy("ContextualRetrievalStrategy")
class ContextualRetrievalStrategy(IRAGStrategy):
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
        config: Dict[str, Any],
        dependencies: StrategyDependencies
    ):
        """
        Initialize contextual retrieval strategy.
        
        Args:
            config: Strategy configuration dictionary
            dependencies: Injected service dependencies
        """
        # Initialize base class (validates dependencies)
        super().__init__(config, dependencies)
        
        # Parse configuration
        self.strategy_config = config if isinstance(config, ContextualRetrievalConfig) else ContextualRetrievalConfig(**config)
        
        # Initialize components
        self.context_generator = ContextGenerator(self.deps.llm_service, self.strategy_config)
        self.cost_tracker = CostTracker(self.strategy_config)
        self.batch_processor = BatchProcessor(
            self.context_generator,
            self.cost_tracker,
            self.strategy_config
        )
        self.storage_manager = ContextualStorageManager(self.deps.database_service, self.strategy_config)
    
    def requires_services(self) -> Set[ServiceDependency]:
        """Declare required services.
        
        Returns:
            Set of required service dependencies
        """
        return {ServiceDependency.LLM, ServiceDependency.EMBEDDING, ServiceDependency.DATABASE}

    def produces(self) -> Set[IndexCapability]:
        """Declare produced capabilities.
        
        Returns:
            Set containing VECTORS and DATABASE capabilities.
        """
        return {
            IndexCapability.VECTORS,
            IndexCapability.DATABASE
        }

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
            
            embedding_result = self.deps.embedding_service.embed([text_to_embed])
            chunk["embedding"] = embedding_result.embeddings[0]
        
        # Store chunks (dual storage)
        self.storage_manager.store_chunks(contextualized_chunks)
        
        # Index in vector store (using contextualized embeddings)
        # Check if vector_store is available (optional dependency for testing)
        vector_store = getattr(self.deps, 'vector_store', None)
        if vector_store:
            for chunk in contextualized_chunks:
                vector_store.index_chunk(
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
            **cost_summary,
            "total_chunks": len(contextualized_chunks),
            "contextualized_chunks": sum(1 for c in contextualized_chunks if "context_description" in c)
        }

    def prepare_data(self, documents: List[Dict[str, Any]]):
        """Prepare and chunk documents for retrieval.
        
        Args:
            documents: List of documents to prepare
            
        Returns:
            PreparedData container
        """
        # TODO: Implement document preparation
        raise NotImplementedError("prepare_data not yet implemented for ContextualRetrievalStrategy")
    
    def process_query(self, query: str, context):
        """Process query with context to generate answer.
        
        Args:
            query: User query
            context: Retrieved context chunks
            
        Returns:
            Generated answer
        """
        # TODO: Implement query processing
        raise NotImplementedError("process_query not yet implemented for ContextualRetrievalStrategy")
    
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
        vector_store = getattr(self.deps, 'vector_store', None)
        if vector_store:
            results = vector_store.search(query=query, top_k=top_k)
        else:
            # Fallback: return empty results if no vector store
            logger.warning("No vector_store available for retrieval")
            results = []
        
        # Get chunk IDs
        chunk_ids = [r.get("chunk_id") or r.get("id") for r in results]
        
        # Retrieve chunks with desired format
        return_format = "original" if self.strategy_config.return_original_text else "contextualized"
        
        if self.strategy_config.return_context:
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

    async def aretrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using contextualized embeddings (async version).
        
        Args:
            query: User query
            top_k: Number of results to return
            **kwargs: Additional parameters
            
        Returns:
            List of results (with original text by default)
        """
        # For now, just call the sync version
        # TODO: Implement true async retrieval when vector_store supports it
        return self.retrieve(query, top_k, **kwargs)


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
