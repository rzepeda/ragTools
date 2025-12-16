"""Multi-query retrieval strategy."""
from typing import List, Set, Dict, Any
import logging

from rag_factory.core.retrieval_interface import IRetrievalStrategy, RetrievalContext
from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import ServiceDependency
from rag_factory.strategies.base import Chunk
from rag_factory.factory import register_rag_strategy

logger = logging.getLogger(__name__)

@register_rag_strategy("MultiQueryRetriever")
class MultiQueryRetriever(IRetrievalStrategy):
    """Retriever that generates multiple query variants for improved recall."""
    
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
        """Retrieve chunks using multiple query variants."""
        
        # For now, use simple semantic search
        # Full multi-query implementation would generate variants with LLM
        # and merge results using Reciprocal Rank Fusion
        
        logger.info(f"Multi-query retrieval for: {query}")
        
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
