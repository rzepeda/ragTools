"""
Knowledge Graph RAG Strategy.

This module implements the main strategy class that combines vector search
with graph-based entity relationships for enhanced retrieval.
"""

from typing import List, Dict, Any, Optional, Set
import logging

from .entity_extractor import EntityExtractor
from .relationship_extractor import RelationshipExtractor
from .graph_store import GraphStore
from .memory_graph_store import MemoryGraphStore
from .hybrid_retriever import HybridRetriever
from .config import KnowledgeGraphConfig
from ...services.dependencies import StrategyDependencies, ServiceDependency
from ..base import IRAGStrategy

logger = logging.getLogger(__name__)


class KnowledgeGraphRAGStrategy(IRAGStrategy):
    """
    Knowledge Graph RAG: Combine vector search with graph relationships.
    
    Extracts entities and relationships from documents, stores them in a
    graph database, and uses graph traversal to enhance retrieval.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dependencies: StrategyDependencies
    ):
        """
        Initialize knowledge graph RAG strategy.
        
        Args:
            config: Strategy configuration dictionary
            dependencies: Injected service dependencies
        """
        # Initialize base class (validates dependencies)
        super().__init__(config, dependencies)
        
        # Parse configuration
        self.strategy_config = config if isinstance(config, KnowledgeGraphConfig) else KnowledgeGraphConfig(**config)
        
        # Initialize components
        self.entity_extractor = EntityExtractor(self.deps.llm_service, self.strategy_config)
        self.relationship_extractor = RelationshipExtractor(self.deps.llm_service, self.strategy_config)
        
        # Initialize graph store (default to in-memory)
        # TODO: Graph service should be used from dependencies
        graph_backend = self.strategy_config.graph_backend
        if graph_backend == "memory":
            self.graph_store = MemoryGraphStore()
        else:
            raise ValueError(f"Unsupported graph backend: {graph_backend}")
        
        # Initialize hybrid retriever
        # TODO: vector_store needs to be from dependencies
        self.hybrid_retriever = HybridRetriever(
            None,  # vector_store_service placeholder
            self.graph_store,
            self.strategy_config
        )
    
    def requires_services(self) -> Set[ServiceDependency]:
        """Declare required services.
        
        Returns:
            Set of required service dependencies
        """
        return {ServiceDependency.LLM, ServiceDependency.EMBEDDING, ServiceDependency.GRAPH}

    def index_document(
        self,
        document: str,
        document_id: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Index document with entity and relationship extraction.
        
        Args:
            document: Document text
            document_id: Unique document ID
            document_metadata: Optional document metadata
            
        Returns:
            Indexing statistics
        """
        logger.info(f"Indexing document with knowledge graph: {document_id}")
        
        # Chunk document (simple paragraph-based chunking)
        chunks = self._chunk_document(document, document_id)
        
        # Index chunks in vector store
        for chunk in chunks:
            self.vector_store.index_chunk(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                metadata={"document_id": document_id}
            )
        
        # Extract entities from each chunk
        all_entities = []
        for chunk in chunks:
            entities = self.entity_extractor.extract_entities(
                chunk["text"],
                chunk["chunk_id"]
            )
            all_entities.extend(entities)
            
            # Add entities to graph
            for entity in entities:
                self.graph_store.add_entity(entity)
        
        # Deduplicate entities
        unique_entities = self.entity_extractor.deduplicate_entities(all_entities)
        
        # Extract relationships
        all_relationships = []
        for chunk in chunks:
            # Find entities in this chunk
            chunk_entities = [
                e for e in unique_entities
                if chunk["chunk_id"] in e.source_chunks
            ]
            
            if len(chunk_entities) >= 2:
                relationships = self.relationship_extractor.extract_relationships(
                    chunk["text"],
                    chunk_entities,
                    chunk["chunk_id"]
                )
                all_relationships.extend(relationships)
                
                # Add relationships to graph
                for rel in relationships:
                    self.graph_store.add_relationship(rel)
        
        logger.info(
            f"Indexed document {document_id}: "
            f"{len(chunks)} chunks, {len(unique_entities)} entities, "
            f"{len(all_relationships)} relationships"
        )
        
        # Log graph stats
        stats = self.graph_store.get_stats()
        logger.info(f"Graph stats: {stats}")
        
        return {
            "document_id": document_id,
            "total_chunks": len(chunks),
            "total_entities": len(unique_entities),
            "total_relationships": len(all_relationships),
            "graph_stats": stats
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval using vector search + graph traversal.
        
        Args:
            query: Search query
            top_k: Number of results
            **kwargs: Additional parameters
            
        Returns:
            List of hybrid search results
        """
        logger.info(f"Knowledge graph retrieval for: {query}")
        
        # Hybrid retrieval
        results = self.hybrid_retriever.retrieve(query, top_k=top_k, **kwargs)
        
        # Convert to dict format
        output = []
        for result in results:
            output.append({
                "chunk_id": result.chunk_id,
                "text": result.text,
                "score": result.combined_score,
                "vector_score": result.vector_score,
                "graph_score": result.graph_score,
                "related_entities": [
                    {"name": e.name, "type": e.type.value}
                    for e in result.related_entities
                ],
                "relationship_paths": result.relationship_paths,
                "metadata": result.metadata
            })
        
        return output

    def _chunk_document(
        self,
        document: str,
        document_id: str
    ) -> List[Dict[str, str]]:
        """Simple document chunking (paragraph-based)."""
        # Split by paragraphs
        paragraphs = [p.strip() for p in document.split("\n\n") if p.strip()]
        
        chunks = []
        for i, para in enumerate(paragraphs):
            chunks.append({
                "chunk_id": f"{document_id}_chunk_{i}",
                "text": para,
                "document_id": document_id
            })
        
        return chunks
    
    def prepare_data(self, documents: List[Dict[str, Any]]):
        """Prepare and chunk documents for retrieval."""
        raise NotImplementedError("prepare_data not yet implemented for KnowledgeGraphRAGStrategy")
    
    async def aretrieve(self, query: str, top_k: int):
        """Async retrieve."""
        return self.retrieve(query, top_k)
    
    def process_query(self, query: str, context):
        """Process query with context."""
        raise NotImplementedError("process_query not yet implemented for KnowledgeGraphRAGStrategy")

    @property
    def name(self) -> str:
        """Strategy name."""
        return "knowledge_graph"

    @property
    def description(self) -> str:
        """Strategy description."""
        return "Combine vector search with graph-based entity relationships"
