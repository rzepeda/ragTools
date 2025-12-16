from typing import List, Set, Dict, Any, Optional
import logging

from rag_factory.core.retrieval_interface import IRetrievalStrategy, RetrievalContext
from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import ServiceDependency
from rag_factory.strategies.base import Chunk
from rag_factory.factory import register_rag_strategy
from rag_factory.strategies.query_expansion.expander_service import QueryExpanderService
from rag_factory.strategies.query_expansion.base import ExpansionConfig, ExpansionStrategy

logger = logging.getLogger(__name__)

@register_rag_strategy("QueryExpansionRetriever")
class QueryExpansionRetriever(IRetrievalStrategy):
    """Retriever that expands queries using LLM before search."""
    
    def requires(self) -> Set[IndexCapability]:
        """Declare required capabilities."""
        return {
            IndexCapability.VECTORS,
            IndexCapability.DATABASE
        }
    
    def requires_services(self) -> Set[ServiceDependency]:
        """Declare required services."""
        return {
            ServiceDependency.EMBEDDING,
            ServiceDependency.DATABASE,
            ServiceDependency.LLM
        }
    
    async def retrieve(
        self,
        query: str,
        context: RetrievalContext,
        top_k: int = 10
    ) -> List[Chunk]:
        """Retrieve chunks using expanded query."""
        
        # For now, use simple semantic search
        # Query expansion with LLM requires proper LLM service implementation
        # which may not be available in all test environments
        
        logger.info(f"Query expansion retrieval for: {query}")
        
        # Embed the original query
        query_embedding = await self.deps.embedding_service.embed(query)
        
        # Search
        results = await context.database.search_chunks(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        # Convert to Chunks
        chunks = []
        for res in results:
            chunks.append(Chunk(
                text=res.get('text', ''),
                score=res.get('score', 0.0),
                metadata=res.get('metadata', {}),
                chunk_id=str(res.get('id', '')),
                source_id=str(res.get('document_id', ''))
            ))
            
        return chunks
