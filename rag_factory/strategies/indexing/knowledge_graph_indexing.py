"""Knowledge graph indexing strategy."""
from typing import List, Set, Dict, Any
import logging

from rag_factory.core.indexing_interface import IIndexingStrategy, IndexingContext, IndexingResult
from rag_factory.core.capabilities import IndexCapability
from rag_factory.services.dependencies import ServiceDependency
from rag_factory.factory import register_rag_strategy

logger = logging.getLogger(__name__)

@register_rag_strategy("KnowledgeGraphIndexer")
class KnowledgeGraphIndexing(IIndexingStrategy):
    """Indexing strategy that creates vector embeddings and extracts knowledge graph entities."""
    
    def produces(self) -> Set[IndexCapability]:
        """Declare produced capabilities."""
        return {
            IndexCapability.VECTORS,
            IndexCapability.CHUNKS,
            IndexCapability.GRAPH,
            IndexCapability.DATABASE
        }
    
    def requires_services(self) -> Set[ServiceDependency]:
        """Declare required services."""
        return {
            ServiceDependency.EMBEDDING,
            ServiceDependency.DATABASE,
            ServiceDependency.LLM,
            ServiceDependency.GRAPH
        }
    
    async def process(
        self,
        documents: List[Dict[str, Any]],
        context: IndexingContext
    ) -> IndexingResult:
        """Process documents by creating embeddings and extracting entities.
        
        Args:
            documents: List of documents to index
            context: Indexing context with database service
            
        Returns:
            IndexingResult with capabilities and metrics
        """
        logger.info(f"Starting KnowledgeGraphIndexing.process with {len(documents)} documents")
        
        # Configuration
        chunk_size = self.config.get('chunk_size', 512)
        overlap = self.config.get('overlap', 50)
        
        # Chunk documents
        all_chunks = []
        for doc in documents:
            text = doc.get('text', '')
            if not text:
                continue
                
            doc_id = doc.get('id', 'unknown')
            
            # Simple chunking
            chunks = self._chunk_document(text, doc_id, chunk_size, overlap)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return IndexingResult(
                capabilities=self.produces(),
                metadata={},
                document_count=len(documents),
                chunk_count=0
            )
        
        # Generate embeddings
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        embeddings = await self.deps.embedding_service.embed_batch(chunk_texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk['embedding'] = embedding
        
        # Store chunks
        await context.database.store_chunks(all_chunks)
        
        # TODO: Extract entities and relationships using LLM service
        # For now, we just do basic vector indexing
        
        return IndexingResult(
            capabilities=self.produces(),
            metadata={
                'chunk_size': chunk_size,
                'overlap': overlap,
                'embedding_dimension': self.deps.embedding_service.get_dimension()
            },
            document_count=len(documents),
            chunk_count=len(all_chunks)
        )
    
    def _chunk_document(
        self,
        text: str,
        doc_id: str,
        chunk_size: int,
        overlap: int
    ) -> List[Dict[str, Any]]:
        """Simple character-based chunking."""
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunks.append({
                'chunk_id': f"{doc_id}_{chunk_idx}",
                'text': chunk_text,
                'document_id': doc_id,
                'metadata': {'chunk_index': chunk_idx}
            })
            
            if end == len(text):
                break
                
            start += chunk_size - overlap
            chunk_idx += 1
        
        return chunks
