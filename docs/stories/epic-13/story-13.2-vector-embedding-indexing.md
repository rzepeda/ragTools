# Story 13.2: Implement Vector Embedding Indexing

**Story ID:** 13.2
**Epic:** Epic 13 - Core Indexing Strategies Implementation
**Story Points:** 8
**Priority:** High
**Dependencies:** Epic 13.1 (Chunking), Epic 10 (ONNX Embedding Service)

---

## User Story

**As a** system
**I want** to create vector embeddings for chunks
**So that** semantic search is possible

---

## Detailed Requirements

### Functional Requirements

1.  **Vector Embedding Strategy**
    *   Implement `VectorEmbeddingIndexing` class implementing `IIndexingStrategy`.
    *   Load chunks from the database (assumes chunking strategy has run).
    *   Generate embeddings for chunk texts using the ONNX embedding service.
    *   Store embeddings in the vector database (pgvector).
    *   Produce `VECTORS` and `DATABASE` capabilities.

2.  **Batch Processing**
    *   Retrieve chunks in batches to manage memory usage.
    *   Send batched requests to the embedding service for efficiency.
    *   Store embeddings in batches.

3.  **Error Handling**
    *   Handle cases where no chunks exist for the document (raise informative error).
    *   Handle embedding service failures gracefully.

### Non-Functional Requirements

1.  **Performance**
    *   Target: <100ms per document (batched).
    *   Optimize batch sizes for throughput.

2.  **Scalability**
    *   Support large numbers of chunks without OOM errors.

---

## Acceptance Criteria

### AC1: Strategy Implementation
- [ ] `VectorEmbeddingIndexing` class exists and implements `IIndexingStrategy`.
- [ ] `produces()` returns `{IndexCapability.VECTORS, IndexCapability.DATABASE}`.
- [ ] `requires_services()` returns `{ServiceDependency.EMBEDDING, ServiceDependency.DATABASE}`.

### AC2: Embedding Generation
- [ ] Chunks are correctly retrieved from the database.
- [ ] Embeddings are generated using the configured model.
- [ ] Batching is implemented and configurable.

### AC3: Storage
- [ ] Embeddings are stored in the vector database associated with their chunk IDs.
- [ ] Vector dimensions match the model output.

### AC4: Testing
- [ ] Unit tests with mocked embedding service and database.
- [ ] Integration tests with pgvector.
- [ ] Performance benchmarks meet targets.

---

## Technical Specifications

### Implementation

```python
from rag_factory.core.indexing import IIndexingStrategy, IndexingContext, IndexingResult
from rag_factory.core.capabilities import IndexCapability
from rag_factory.core.dependencies import ServiceDependency

class VectorEmbeddingIndexing(IIndexingStrategy):
    """Creates vector embeddings for text chunks"""
    
    def produces(self) -> set[IndexCapability]:
        return {
            IndexCapability.VECTORS,
            IndexCapability.DATABASE
        }
    
    def requires_services(self) -> set[ServiceDependency]:
        return {
            ServiceDependency.EMBEDDING,
            ServiceDependency.DATABASE
        }
    
    async def process(
        self,
        documents: list['Document'],
        context: IndexingContext
    ) -> IndexingResult:
        """
        Create and store vector embeddings.
        
        Strategy:
        1. Retrieve chunks from database (assumes chunking already done)
        2. Batch embed all chunk texts
        3. Store embeddings in vector database
        """
        batch_size = self.config.get('batch_size', 32)
        
        # Retrieve chunks for these documents
        chunks = await context.database.get_chunks_for_documents(
            [doc.id for doc in documents]
        )
        
        if not chunks:
            raise ValueError(
                "No chunks found. Run a chunking strategy first "
                "(e.g., ContextAwareChunkingIndexing)"
            )
        
        # Embed in batches
        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c['text'] for c in batch]
            
            embeddings = await self.deps.embedding_service.embed_batch(texts)
            all_embeddings.extend(embeddings)
        
        # Store embeddings
        await context.database.store_embeddings(
            chunk_ids=[c['id'] for c in chunks],
            embeddings=all_embeddings
        )
        
        return IndexingResult(
            capabilities=self.produces(),
            metadata={
                'embedding_model': self.config.get('model', 'default'),
                'embedding_dimension': self.deps.embedding_service.get_dimension(),
                'batch_size': batch_size
            },
            document_count=len(documents),
            chunk_count=len(chunks)
        )
```

### Technical Dependencies
- ONNX embedding service
- PostgreSQL with pgvector extension
