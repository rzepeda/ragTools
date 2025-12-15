"""
Hierarchical RAG Strategy Implementation.

This module implements a hierarchical RAG strategy that indexes documents
with parent-child chunk relationships and retrieves small chunks while
expanding them with parent context.
"""

from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4

from ..base import IRAGStrategy, Chunk, StrategyConfig, PreparedData
from .models import (
    HierarchicalConfig,
    ExpansionStrategy,
    HierarchicalChunk,
    ExpandedChunk
)
from .hierarchy_builder import HierarchyBuilder
from .parent_retriever import ParentRetriever


class HierarchicalRAGStrategy(IRAGStrategy):
    """Hierarchical RAG strategy with parent-child chunk relationships.
    
    This strategy:
    1. Builds a hierarchy from documents (document → section → paragraph)
    2. Indexes small chunks for precise vector search
    3. Expands retrieved chunks with parent context for better LLM comprehension
    """
    
    def __init__(
        self,
        vector_store_service,
        database_service,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the hierarchical RAG strategy.
        
        Args:
            vector_store_service: Service for vector embeddings and search
            database_service: Service for database operations
            config: Configuration dictionary
        """
        self.vector_store = vector_store_service
        self.database = database_service
        
        # Parse hierarchical config
        config = config or {}
        expansion_strategy = config.get("expansion_strategy", "immediate_parent")
        if isinstance(expansion_strategy, str):
            expansion_strategy = ExpansionStrategy(expansion_strategy)
        
        self.hierarchical_config = HierarchicalConfig(
            expansion_strategy=expansion_strategy,
            small_chunk_size=config.get("small_chunk_size", 256),
            large_chunk_size=config.get("large_chunk_size", 1024),
            search_small_chunks=config.get("search_small_chunks", True),
            window_size=config.get("window_size", 2),
            max_hierarchy_depth=config.get("max_hierarchy_depth", 4),
            min_chunk_size=config.get("min_chunk_size", 50)
        )
        
        self.hierarchy_builder = HierarchyBuilder(self.hierarchical_config)
        self.parent_retriever = ParentRetriever(
            self.database.chunk_repository,
            self.hierarchical_config
        )
        
        self.strategy_config = StrategyConfig(
            chunk_size=self.hierarchical_config.small_chunk_size,
            chunk_overlap=0,  # No overlap in hierarchical chunking
            top_k=5,
            strategy_name="hierarchical"
        )
    
    def requires_services(self):
        """Declare required services.
        
        Returns:
            Set of required service dependencies
        """
        from ...services.dependencies import ServiceDependency
        return {ServiceDependency.DATABASE}
    
    def initialize(self, config: StrategyConfig) -> None:
        """Initialize the strategy with configuration.
        
        Args:
            config: Strategy configuration
        """
        self.strategy_config = config
    
    def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
        """Prepare and chunk documents with hierarchy.
        
        Args:
            documents: List of documents with 'text' and 'id' fields
            
        Returns:
            PreparedData with hierarchical chunks
        """
        all_chunks = []
        all_embeddings = []
        
        for doc in documents:
            text = doc.get("text", "")
            doc_id = doc.get("id", str(uuid4()))
            
            # Build hierarchy
            hierarchy = self.hierarchy_builder.build(text, doc_id)
            
            # Convert to base Chunk objects and get embeddings
            for chunk_id, h_chunk in hierarchy.all_chunks.items():
                # Create embedding
                embedding = self.vector_store.embed_text(h_chunk.text)
                
                # Convert to base Chunk
                chunk = Chunk(
                    text=h_chunk.text,
                    metadata={
                        "document_id": h_chunk.document_id,
                        "hierarchy_level": h_chunk.hierarchy_level.value,
                        "parent_chunk_id": h_chunk.parent_chunk_id,
                        **h_chunk.metadata
                    },
                    score=0.0,
                    source_id=h_chunk.document_id,
                    chunk_id=h_chunk.chunk_id
                )
                
                all_chunks.append(chunk)
                all_embeddings.append(embedding)
        
        return PreparedData(
            chunks=all_chunks,
            embeddings=all_embeddings,
            index_metadata={
                "strategy": "hierarchical",
                "expansion_strategy": self.hierarchical_config.expansion_strategy.value,
                "total_chunks": len(all_chunks)
            }
        )
    
    def index_document(self, text: str, document_id: str) -> None:
        """Index a document with hierarchical structure.
        
        Args:
            text: The document text
            document_id: Unique identifier for the document
        """
        # Build hierarchy
        hierarchy = self.hierarchy_builder.build(text, document_id)
        
        # Store chunks in database with hierarchy
        for chunk_id, h_chunk in hierarchy.all_chunks.items():
            # Generate embedding
            embedding = self.vector_store.embed_text(h_chunk.text)
            
            # Convert document_id to UUID for this chunk
            try:
                chunk_doc_uuid = UUID(h_chunk.document_id)
            except (ValueError, AttributeError):
                # If document_id is not a valid UUID, generate a deterministic one
                import uuid
                chunk_doc_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, h_chunk.document_id)
            
            # Store in database
            self.database.chunk_repository.create(
                document_id=chunk_doc_uuid,
                chunk_index=h_chunk.hierarchy_metadata.position_in_parent,
                text=h_chunk.text,
                embedding=embedding,
                metadata={
                    "parent_chunk_id": h_chunk.parent_chunk_id,
                    "hierarchy_level": h_chunk.hierarchy_level.value,
                    "hierarchy_metadata": {
                        "position_in_parent": h_chunk.hierarchy_metadata.position_in_parent,
                        "total_siblings": h_chunk.hierarchy_metadata.total_siblings,
                        "depth_from_root": h_chunk.hierarchy_metadata.depth_from_root
                    },
                    **h_chunk.metadata
                }
            )
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Retrieve relevant chunks with parent context expansion.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of chunks with expanded context
        """
        # Generate query embedding
        query_embedding = self.vector_store.embed_text(query)
        
        # Search for similar chunks
        results = self.database.chunk_repository.search_similar(
            embedding=query_embedding,
            top_k=top_k
        )
        
        # Convert to hierarchical chunks
        hierarchical_chunks = []
        for db_chunk, score in results:
            h_chunk = self._db_chunk_to_hierarchical(db_chunk, score)
            hierarchical_chunks.append(h_chunk)
        
        # Expand with parent context
        expanded_chunks = self.parent_retriever.expand_chunks(hierarchical_chunks)
        
        # Convert back to base Chunk objects
        result_chunks = []
        for expanded in expanded_chunks:
            chunk = Chunk(
                text=expanded.expanded_text,
                metadata={
                    "original_text": expanded.original_chunk.text,
                    "expansion_strategy": expanded.expansion_strategy.value,
                    "parent_chunks": [p.chunk_id for p in expanded.parent_chunks],
                    "total_tokens": expanded.total_tokens
                },
                score=expanded.original_chunk.metadata.get("score", 0.0),
                source_id=expanded.original_chunk.document_id,
                chunk_id=expanded.original_chunk.chunk_id
            )
            result_chunks.append(chunk)
        
        return result_chunks
    
    async def aretrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Async retrieve (delegates to sync version for now).
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of chunks with expanded context
        """
        # For now, delegate to sync version
        # TODO: Implement true async version
        return self.retrieve(query, top_k)
    
    def process_query(self, query: str, context: List[Chunk]) -> str:
        """Process query with retrieved context.
        
        Args:
            query: The user's query
            context: List of relevant chunks
            
        Returns:
            Generated answer (placeholder for now)
        """
        # This would typically call an LLM
        # For now, return a simple concatenation
        context_text = "\n\n---\n\n".join([chunk.text for chunk in context])
        return f"Query: {query}\n\nContext:\n{context_text}"
    
    def _db_chunk_to_hierarchical(
        self,
        db_chunk,
        score: float = 0.0
    ) -> HierarchicalChunk:
        """Convert database chunk to hierarchical chunk.
        
        Args:
            db_chunk: Database Chunk model instance
            score: Relevance score
            
        Returns:
            HierarchicalChunk instance
        """
        from .models import HierarchyMetadata, HierarchyLevel
        
        # Extract hierarchy metadata
        h_meta_dict = db_chunk.metadata_.get("hierarchy_metadata", {})
        hierarchy_metadata = HierarchyMetadata(
            position_in_parent=h_meta_dict.get("position_in_parent", 0),
            total_siblings=h_meta_dict.get("total_siblings", 0),
            depth_from_root=h_meta_dict.get("depth_from_root", 0)
        )
        
        hierarchy_level = HierarchyLevel(
            db_chunk.metadata_.get("hierarchy_level", 0)
        )
        
        parent_chunk_id = db_chunk.metadata_.get("parent_chunk_id")
        
        return HierarchicalChunk(
            chunk_id=str(db_chunk.chunk_id),
            document_id=str(db_chunk.document_id),
            text=db_chunk.text,
            hierarchy_level=hierarchy_level,
            hierarchy_metadata=hierarchy_metadata,
            parent_chunk_id=parent_chunk_id,
            token_count=len(db_chunk.text.split()),
            metadata={**db_chunk.metadata_, "score": score},
            embedding=None
        )
