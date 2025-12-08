"""Vector embedding indexing strategy."""
from typing import List, Set, Dict, Any

from rag_factory.core.indexing_interface import IIndexingStrategy, IndexingContext
from rag_factory.core.capabilities import IndexCapability, IndexingResult
from rag_factory.services.dependencies import ServiceDependency

class VectorEmbeddingIndexing(IIndexingStrategy):
    """
    Indexing strategy that creates vector embeddings for text chunks.
    
    This strategy retrieves existing chunks from the database, generates
    embeddings using the configured embedding service, and stores them
    in the vector database.
    """
    
    def produces(self) -> Set[IndexCapability]:
        """
        Declares that this strategy produces vector embeddings and uses the database.
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
    
    async def process(
        self,
        documents: List[Dict[str, Any]],
        context: IndexingContext
    ) -> IndexingResult:
        """
        Create and store vector embeddings for the given documents.
        
        Args:
            documents: List of documents to process
            context: Indexing context containing services and configuration
            
        Returns:
            IndexingResult with metadata about the operation
            
        Raises:
            ValueError: If no chunks are found for the documents
        """
        batch_size = self.config.get('batch_size', 32)
        
        if not self.deps.embedding_service:
            raise ValueError("Embedding service is required")
        
        # Retrieve chunks for these documents
        # We assume a previous strategy (like ContextAwareChunkingIndexing) has already
        # created chunks and stored them in the database.
        document_ids = [str(doc.get('id')) for doc in documents if doc.get('id')]
        chunks = await context.database.get_chunks_for_documents(document_ids)
        
        if not chunks:
            raise ValueError(
                "No chunks found for the provided documents. "
                "Ensure a chunking strategy has been executed first."
            )
        
        # Embed in batches
        all_embeddings: List[List[float]] = []
        
        # Process chunks in batches to avoid overwhelming the embedding service
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c['text'] for c in batch]
            
            # Generate embeddings for the batch
            embeddings = await self.deps.embedding_service.embed_batch(texts)
            all_embeddings.extend(embeddings)
        
        # Store embeddings in the database
        # We map embeddings back to their chunk IDs
        chunks_to_update = []
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk['embedding'] = embedding
            chunks_to_update.append(chunk)
            
        await context.database.store_chunks(chunks_to_update)
        
        return IndexingResult(
            capabilities=self.produces(),
            metadata={
                'embedding_model': self.config.get('model', 'default'),
                'embedding_dimension': self.deps.embedding_service.get_dimension(),
                'batch_size': batch_size,
                'total_embeddings': len(all_embeddings)
            },
            document_count=len(documents),
            chunk_count=len(chunks)
        )
