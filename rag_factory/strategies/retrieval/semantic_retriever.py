"""Semantic retrieval strategy."""
from typing import List, Set, Dict, Any

from rag_factory.core.retrieval_interface import IRetrievalStrategy, RetrievalContext
from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import ServiceDependency
from rag_factory.strategies.base import Chunk
from rag_factory.factory import register_rag_strategy

@register_rag_strategy("SemanticRetriever")
class SemanticRetriever(IRetrievalStrategy):
    """
    Retrieval strategy that uses semantic vector search.
    
    This strategy embeds the query using an embedding service and performs
    vector similarity search in the database.
    """
    
    def requires(self) -> Set[IndexCapability]:
        """
        Declares that this strategy requires vector embeddings and database access.
        """
        return {
            IndexCapability.VECTORS,
            IndexCapability.DATABASE
        }
    
    def requires_services(self) -> Set[ServiceDependency]:
        """
        Declares dependencies on embedding and database services.
        """
        return {
            ServiceDependency.EMBEDDING,
            ServiceDependency.DATABASE
        }
    
    async def retrieve(
        self,
        query: str,
        context: RetrievalContext,
        top_k: int = 10
    ) -> List[Chunk]:
        """
        Retrieve relevant chunks for query.
        
        Args:
            query: User query string
            context: Retrieval context containing database service
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks
        """
        # Get configuration from context or usage default
        # Note: self.config is available but context.config might override?
        # Typically strategy config is fixed at creation, context allows override?
        # IRetrievalStrategy docstring says context.config is "Configuration dictionary for retrieval parameters"
        # I will merge them or prefer context.
        
        # Embed the query
        query_embedding = await self.deps.embedding_service.embed(query)
        
        # Search the database
        results = await context.database.search_chunks(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        # Convert dict results to Chunk objects
        chunks = []
        for res in results:
            chunk = Chunk(
                text=res.get('text', ''),
                score=res.get('score', 0.0),
                metadata=res.get('metadata', {}),
                chunk_id=str(res.get('id', '')),
                source_id=str(res.get('document_id', ''))
            )
            chunks.append(chunk)
            
        return chunks
