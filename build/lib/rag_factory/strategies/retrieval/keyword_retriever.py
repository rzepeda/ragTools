"""
Keyword-based retrieval strategy using BM25.

This module implements keyword search without embeddings,
using BM25 algorithm for ranking.
"""

from typing import List, Dict, Any, Set
import logging

from rag_factory.core.retrieval_interface import (
    IRetrievalStrategy,
    RetrievalContext
)
from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import ServiceDependency
from rag_factory.strategies.base import Chunk
from rag_factory.factory import register_rag_strategy

logger = logging.getLogger(__name__)


@register_rag_strategy("KeywordRetriever")
class KeywordRetriever(IRetrievalStrategy):
    """Retrieves chunks using BM25 keyword search.
    
    This strategy performs keyword-based search without requiring
    embeddings, using BM25 algorithm for ranking.
    
    Requires:
        - KEYWORD: Inverted index for keyword search
        - DATABASE: Data stored in database
    """

    def requires(self) -> Set[IndexCapability]:
        """Declare required capabilities.

        Returns:
            Set containing KEYWORD and DATABASE capabilities
        """
        return {
            IndexCapability.KEYWORDS,
            IndexCapability.DATABASE
        }

    def requires_services(self) -> Set[ServiceDependency]:
        """Declare required services.

        Returns:
            Set containing DATABASE service dependency
        """
        return {
            ServiceDependency.DATABASE
        }

    async def retrieve(
        self,
        query: str,
        context: RetrievalContext
    ) -> List[Chunk]:
        """Retrieve chunks using BM25 keyword search.

        Args:
            query: Search query
            context: Retrieval context with database service

        Returns:
            List of relevant chunks ranked by BM25 score
        """
        # Get configuration
        top_k = self.config.get('top_k', 5)
        algorithm = self.config.get('algorithm', 'bm25')
        k1 = self.config.get('k1', 1.5)
        b = self.config.get('b', 0.75)

        logger.info(f"Keyword retrieval for: {query} (algorithm={algorithm})")

        # Tokenize query
        query_terms = self._tokenize(query)

        if not query_terms:
            logger.warning("No valid query terms after tokenization")
            return []

        # Search using keyword index
        # This would typically use the inverted index stored during indexing
        try:
            results = await context.database.search_keyword(
                query_terms=query_terms,
                top_k=top_k,
                algorithm=algorithm,
                k1=k1,
                b=b
            )
        except AttributeError:
            # Fallback if database doesn't have search_keyword method
            logger.warning("Database service doesn't support keyword search, using fallback")
            results = await self._fallback_search(query_terms, context, top_k)

        # Convert to Chunk objects
        chunks = []
        for result in results:
            chunk = Chunk(
                text=result.get('text', ''),
                metadata=result.get('metadata', {}),
                score=result.get('score', 0.0),
                source_id=result.get('document_id', ''),
                chunk_id=result.get('chunk_id', result.get('id', ''))
            )
            chunks.append(chunk)

        logger.info(f"Retrieved {len(chunks)} chunks using keyword search")
        return chunks

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms.

        Args:
            text: Text to tokenize

        Returns:
            List of terms
        """
        # Simple tokenization - split on whitespace and lowercase
        # In production, this should use proper tokenization with stemming/lemmatization
        terms = text.lower().split()
        
        # Remove common stopwords if configured
        if self.config.get('remove_stopwords', True):
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
            terms = [t for t in terms if t not in stopwords]
        
        return terms

    async def _fallback_search(
        self,
        query_terms: List[str],
        context: RetrievalContext,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Fallback search using simple text matching.

        Args:
            query_terms: Query terms
            context: Retrieval context
            top_k: Number of results

        Returns:
            List of matching chunks
        """
        # Simple fallback: search for chunks containing any query term
        # This is not true BM25 but provides basic functionality
        try:
            # Try to get all chunks and filter
            all_chunks = await context.database.get_all_chunks()
            
            scored_chunks = []
            for chunk in all_chunks[:1000]:  # Limit to first 1000 for performance
                text = chunk.get('text', '').lower()
                score = sum(1 for term in query_terms if term in text)
                
                if score > 0:
                    chunk['score'] = score / len(query_terms)
                    scored_chunks.append(chunk)
            
            # Sort by score and return top_k
            scored_chunks.sort(key=lambda x: x['score'], reverse=True)
            return scored_chunks[:top_k]
            
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
