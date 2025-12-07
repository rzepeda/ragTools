"""
Hybrid retrieval combining vector search with graph traversal.

This module implements the hybrid retriever that combines vector similarity
search with graph-based entity relationships for enhanced retrieval.
"""

from typing import List, Dict, Any, Optional
import logging

from .models import HybridSearchResult, Entity
from .graph_store import GraphStore
from .config import KnowledgeGraphConfig

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines vector search with graph traversal."""

    def __init__(
        self,
        vector_store: Any,
        graph_store: GraphStore,
        config: KnowledgeGraphConfig
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store for similarity search
            graph_store: Graph store for entity relationships
            config: Knowledge graph configuration
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.config = config
        
        # Weights for combining scores
        self.vector_weight = config.vector_weight
        self.graph_weight = config.graph_weight
        self.max_hops = config.max_graph_hops

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[HybridSearchResult]:
        """
        Hybrid retrieval combining vector search and graph traversal.
        
        Args:
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional parameters
            
        Returns:
            List of hybrid search results
        """
        logger.info(f"Hybrid retrieval for query: {query}")
        
        # Step 1: Vector search
        vector_results = self.vector_store.search(query, top_k=top_k * 2)
        logger.info(f"Vector search returned {len(vector_results)} results")
        
        if not vector_results:
            return []
        
        # Step 2: Extract entities from retrieved chunks
        chunk_ids = [r.get("chunk_id") or r.get("id") for r in vector_results]
        entities_in_chunks = self._get_entities_in_chunks(chunk_ids)
        logger.info(f"Found {len(entities_in_chunks)} entities in retrieved chunks")
        
        # Step 3: Graph expansion
        graph_result = None
        if entities_in_chunks:
            entity_ids = [e.id for e in entities_in_chunks]
            graph_result = self.graph_store.traverse(
                start_entity_ids=entity_ids,
                max_hops=self.max_hops
            )
            logger.info(
                f"Graph traversal found {len(graph_result.entities)} entities, "
                f"{len(graph_result.relationships)} relationships"
            )
        
        # Step 4: Combine scores
        hybrid_results = self._combine_results(
            vector_results,
            entities_in_chunks,
            graph_result
        )
        
        # Step 5: Re-rank and return top_k
        hybrid_results.sort(key=lambda x: x.combined_score, reverse=True)
        return hybrid_results[:top_k]

    def _get_entities_in_chunks(self, chunk_ids: List[str]) -> List[Entity]:
        """Find all entities that appear in given chunks."""
        entities = []
        
        for entity in self.graph_store.entities.values():
            # Check if entity appears in any of the chunks
            if any(chunk_id in entity.source_chunks for chunk_id in chunk_ids):
                entities.append(entity)
        
        return entities

    def _combine_results(
        self,
        vector_results: List[Dict[str, Any]],
        entities_in_chunks: List[Entity],
        graph_result: Optional[Any]
    ) -> List[HybridSearchResult]:
        """Combine vector and graph results."""
        hybrid_results = []
        
        for vec_result in vector_results:
            chunk_id = vec_result.get("chunk_id") or vec_result.get("id")
            vector_score = vec_result.get("score", 0.0)
            
            # Find entities in this chunk
            related_entities = [
                e for e in entities_in_chunks
                if chunk_id in e.source_chunks
            ]
            
            # Calculate graph score
            graph_score = 0.0
            relationship_paths = []
            
            if graph_result and related_entities:
                # Graph score based on entity importance and connectivity
                for entity in related_entities:
                    entity_score = graph_result.scores.get(entity.id, 0.0)
                    graph_score = max(graph_score, entity_score)
                
                # Extract relationship paths involving these entities
                for path in graph_result.paths:
                    if any(e.id in path for e in related_entities):
                        relationship_paths.append(path)
            
            # Combined score
            combined_score = (
                self.vector_weight * vector_score +
                self.graph_weight * graph_score
            )
            
            # Create hybrid result
            result = HybridSearchResult(
                chunk_id=chunk_id,
                text=vec_result.get("text", ""),
                vector_score=vector_score,
                graph_score=graph_score,
                combined_score=combined_score,
                related_entities=related_entities,
                relationship_paths=relationship_paths[:5],  # Limit paths
                metadata=vec_result.get("metadata", {})
            )
            
            hybrid_results.append(result)
        
        return hybrid_results
