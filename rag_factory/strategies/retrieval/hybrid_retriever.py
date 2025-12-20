"""Hybrid search retriever combining semantic and keyword search.

This module implements a hybrid retrieval strategy that combines:
1. Semantic/Vector search (cosine similarity)
2. Keyword/BM25 search (lexical matching)
3. Reciprocal Rank Fusion (RRF) for score combination
"""

from typing import List, Set, Dict, Any
import logging

from rag_factory.core.retrieval_interface import IRetrievalStrategy, RetrievalContext
from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import ServiceDependency
from rag_factory.strategies.base import Chunk
from rag_factory.factory import register_rag_strategy

logger = logging.getLogger(__name__)


@register_rag_strategy("HybridSearchRetriever")
class HybridSearchRetriever(IRetrievalStrategy):
    """
    Hybrid retrieval combining semantic vector search and BM25 keyword search.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from both methods,
    providing better recall than either method alone.
    """
    
    def requires(self) -> Set[IndexCapability]:
        """Declares that this strategy requires vectors (keywords optional)."""
        return {
            IndexCapability.VECTORS,
            IndexCapability.DATABASE
        }
    
    def requires_services(self) -> Set[ServiceDependency]:
        """Declares dependencies on embedding and database services."""
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
        """Retrieve relevant chunks using hybrid search."""
        logger.info(f"Hybrid search for: {query}")
        
        # Get configuration
        effective_top_k = getattr(self.config, 'top_k', top_k)
        semantic_weight = self.config.get('semantic_weight', 0.5)
        keyword_weight = self.config.get('keyword_weight', 0.5)
        rrf_k = self.config.get('rrf_k', 60)
        
        # Retrieve more candidates from each method
        candidate_k = effective_top_k * 2
        
        # 1. Semantic search
        query_embedding = await self.deps.embedding_service.embed(query)
        semantic_results = await context.database.search_chunks(
            query_embedding=query_embedding,
            top_k=candidate_k
        )
        
        # 2. Keyword search (BM25) - fallback to semantic if not available
        try:
            keyword_results = await context.database.search_chunks_by_keywords(
                query=query,
                top_k=candidate_k
            )
        except (AttributeError, NotImplementedError):
            logger.warning("Keyword search not available, using semantic only")
            keyword_results = []
        
        # 3. Combine using RRF
        fused_results = self._reciprocal_rank_fusion(
            semantic_results, keyword_results,
            semantic_weight, keyword_weight, rrf_k
        )
        
        # 4. Convert to Chunk objects
        chunks = []
        for result in fused_results[:effective_top_k]:
            chunk = Chunk(
                text=result.get('text', ''),
                score=result.get('hybrid_score', 0.0),
                metadata={
                    **result.get('metadata', {}),
                    'semantic_score': result.get('semantic_score', 0.0),
                    'keyword_score': result.get('keyword_score', 0.0),
                    'search_method': 'hybrid'
                },
                chunk_id=str(result.get('id', result.get('chunk_id', ''))),
                source_id=str(result.get('document_id', ''))
            )
            chunks.append(chunk)
        
        return chunks
    
    def _reciprocal_rank_fusion(
        self, semantic_results, keyword_results,
        semantic_weight, keyword_weight, k=60
    ):
        """Combine results using Reciprocal Rank Fusion."""
        # Build rank maps
        semantic_ranks = {
            (r.get('id') or r.get('chunk_id')): {'rank': i+1, 'score': r.get('score', 0), 'result': r}
            for i, r in enumerate(semantic_results) if r.get('id') or r.get('chunk_id')
        }
        
        keyword_ranks = {
            (r.get('id') or r.get('chunk_id')): {'rank': i+1, 'score': r.get('score', 0), 'result': r}
            for i, r in enumerate(keyword_results) if r.get('id') or r.get('chunk_id')
        }
        
        # Combine all unique chunk IDs
        all_ids = set(semantic_ranks.keys()) | set(keyword_ranks.keys())
        
        # Calculate RRF scores
        fused = []
        for chunk_id in all_ids:
            sem_rank = semantic_ranks.get(chunk_id, {}).get('rank', 0)
            kw_rank = keyword_ranks.get(chunk_id, {}).get('rank', 0)
            
            rrf_score = 0.0
            if sem_rank > 0:
                rrf_score += semantic_weight / (k + sem_rank)
            if kw_rank > 0:
                rrf_score += keyword_weight / (k + kw_rank)
            
            result = (semantic_ranks.get(chunk_id) or keyword_ranks.get(chunk_id))['result']
            fused.append({
                **result,
                'hybrid_score': rrf_score,
                'semantic_score': semantic_ranks.get(chunk_id, {}).get('score', 0.0),
                'keyword_score': keyword_ranks.get(chunk_id, {}).get('score', 0.0),
            })
        
        fused.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return fused
