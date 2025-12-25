"""Vector embedding indexing strategy."""
import logging
import uuid
from typing import List, Set, Dict, Any

from rag_factory.core.indexing_interface import IIndexingStrategy, IndexingContext
from rag_factory.core.capabilities import IndexCapability, IndexingResult
from rag_factory.services.dependencies import ServiceDependency
from rag_factory.factory import register_rag_strategy

logger = logging.getLogger(__name__)

@register_rag_strategy("VectorEmbeddingIndexer")
class VectorEmbeddingIndexing(IIndexingStrategy):
    """
    Indexing strategy that creates vector embeddings for text chunks.
    
    This strategy chunks documents, generates embeddings using the 
    configured embedding service, and stores them in the vector database.
    """
    
    def produces(self) -> Set[IndexCapability]:
        """
        Declares that this strategy produces vector embeddings, chunks, and uses the database.
        """
        return {
            IndexCapability.VECTORS,
            IndexCapability.CHUNKS,
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
        """
        logger.info(f"Starting VectorEmbeddingIndexing.process with {len(documents)} documents")

        batch_size = self.config.get('batch_size', 32)
        chunk_size = self.config.get('chunk_size', 512)
        overlap = self.config.get('overlap', 50)

        logger.debug(f"Configuration: batch_size={batch_size}, chunk_size={chunk_size}, overlap={overlap}")

        if not self.deps.embedding_service:
            logger.error("Embedding service is not available in dependencies")
            raise ValueError("Embedding service is required")

        logger.info(f"Using embedding service: {type(self.deps.embedding_service).__name__}")

        # Store documents first to satisfy foreign key constraints
        logger.info(f"Storing {len(documents)} documents...")
        for doc in documents:
            doc_id = doc.get("id")
            if not doc_id:
                logger.warning("Document is missing an 'id', skipping.")
                continue
            
            doc_data = {
                "document_id": doc_id,
                "filename": doc.get("metadata", {}).get("filename", str(doc_id)),
                "source_path": doc.get("metadata", {}).get("source_path", str(doc_id)),
                "content_hash": str(uuid.uuid4()),  # Placeholder
                "total_chunks": 0,  # Will be updated after chunking
                "metadata": doc.get("metadata", {}),
                "status": "indexing",
            }
            try:
                context.database.insert("documents", doc_data)
            except Exception as e:
                logger.error(f"Failed to insert document {doc_id}: {e}")
                # Depending on desired behavior, we might want to skip this doc
                # or raise the exception. For now, we log and continue.
        logger.info("Document storage complete.")


        # 1. Chunk documents
        logger.info("Starting document chunking...")
        chunks = self._chunk_documents(documents, chunk_size, overlap)
        logger.info(f"Document chunking complete: created {len(chunks)} chunks from {len(documents)} documents")
        
        if not chunks:
            # Handle empty documents case gracefully
            logger.warning("No chunks created from documents (all empty?), returning empty result")
            return IndexingResult(
                capabilities=self.produces(),
                metadata={},
                document_count=len(documents),
                chunk_count=0
            )

        # 2. Embed in batches
        logger.info(f"Starting embedding generation for {len(chunks)} chunks in batches of {batch_size}...")
        all_embeddings: List[List[float]] = []

        num_batches = (len(chunks) + batch_size - 1) // batch_size
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c['text'] for c in batch]

            batch_num = (i // batch_size) + 1
            logger.debug(f"Processing batch {batch_num}/{num_batches} with {len(texts)} texts")

            # Generate embeddings for the batch
            embeddings = await self.deps.embedding_service.embed_batch(texts)
            all_embeddings.extend(embeddings)

            logger.debug(f"Batch {batch_num}/{num_batches} complete: generated {len(embeddings)} embeddings")

        logger.info(f"Embedding generation complete: created {len(all_embeddings)} embeddings")

        # 3. Add embeddings to chunks and store
        logger.info("Adding embeddings to chunks...")
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk['embedding'] = embedding

        chunks_table = self.config.get('db_config', {}).get('tables', {}).get('chunks')
        logger.info(f"Storing {len(chunks)} chunks to table: {chunks_table or 'default'}...")
        await context.database.store_chunks(chunks)
        logger.info("Database storage complete")

        embedding_dim = self.deps.embedding_service.get_dimension()
        logger.info(f"Indexing complete - Documents: {len(documents)}, Chunks: {len(chunks)}, Embedding dimension: {embedding_dim}")

        return IndexingResult(
            capabilities=self.produces(),
            metadata={
                'embedding_model': self.config.get('model', 'default'),
                'embedding_dimension': embedding_dim,
                'batch_size': batch_size,
                'chunk_size': chunk_size,
                'overlap': overlap,
                'total_embeddings': len(all_embeddings)
            },
            document_count=len(documents),
            chunk_count=len(chunks)
        )
        
    def _chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int,
        overlap: int
    ) -> List[Dict[str, Any]]:
        """Split documents into chunks."""
        logger.debug(f"_chunk_documents called with {len(documents)} documents, chunk_size={chunk_size}, overlap={overlap}")
        chunks = []

        for doc_idx, doc in enumerate(documents):
            text = doc.get('text', '')
            doc_id = str(doc.get('id', ''))
            metadata = doc.get('metadata', {}).copy()
            metadata['document_id'] = doc_id

            logger.debug(f"Processing document {doc_idx + 1}/{len(documents)}: doc_id='{doc_id}', text_length={len(text)}")

            if not text:
                logger.debug(f"Skipping empty document: {doc_id}")
                continue

            # Simple character-based chunking
            # Ideally use a better chunker, but this suffices for MVP
            start = 0
            chunk_idx = 0

            if len(text) <= chunk_size:
                logger.debug(f"Document {doc_id} fits in single chunk (length={len(text)} <= {chunk_size})")
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "text": text,
                    "metadata": metadata,
                    "document_id": doc_id,
                    "chunk_index": chunk_idx
                })
                continue

            logger.debug(f"Document {doc_id} requires chunking (length={len(text)} > {chunk_size})")
            iteration_count = 0
            max_iterations = len(text) // (chunk_size - overlap) + 10  # Safety limit

            while start < len(text):
                iteration_count += 1
                if iteration_count > max_iterations:
                    logger.error(f"INFINITE LOOP DETECTED in document {doc_id}: iteration {iteration_count}, start={start}, text_length={len(text)}")
                    break

                # Ensure we don't go past end
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end]
                original_end = end

                # If we are not at end, try to find a space to break
                if end < len(text) and ' ' in chunk_text:
                    last_space = chunk_text.rfind(' ')
                    # Only cut back if space is reasonably far
                    if last_space > chunk_size * 0.5:
                        end = start + last_space
                        chunk_text = text[start:end]
                        logger.debug(f"Adjusted chunk boundary at word break: {original_end} -> {end}")

                logger.debug(f"Creating chunk {chunk_idx} for doc {doc_id}: start={start}, end={end}, chunk_length={len(chunk_text)}")

                if end == len(text):
                    chunks.append({
                        "chunk_id": str(uuid.uuid4()),
                        "text": chunk_text,
                        "document_id": doc_id, # Use doc_id directly
                        "chunk_index": chunk_idx,
                        "metadata": metadata # Use metadata directly
                    })
                    break 

                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "document_id": doc_id, # Use doc_id directly
                    "chunk_index": chunk_idx,
                    "metadata": metadata # Use metadata directly
                })
                
                step = len(chunk_text) - overlap
                if step <= 0:
                     step = 1 # Force advance
                     
                start += step
                chunk_idx += 1

            logger.debug(f"Document {doc_id} chunking complete: created {chunk_idx} chunks")

        logger.debug(f"_chunk_documents complete: total {len(chunks)} chunks created")
        return chunks
