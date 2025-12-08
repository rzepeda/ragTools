# Story 13.1: Implement Context-Aware Chunking (Indexing)

**Story ID:** 13.1
**Epic:** Epic 13 - Core Indexing Strategies Implementation
**Story Points:** 13
**Priority:** High
**Dependencies:** Epic 12 (Pipeline Separation), Epic 10 (ONNX Embedding Service)

---

## User Story

**As a** system
**I want** document chunking at natural semantic boundaries
**So that** chunk quality is optimized for retrieval

---

## Detailed Requirements

### Functional Requirements

1.  **Context-Aware Chunking Strategy**
    *   Implement `ContextAwareChunkingIndexing` class implementing `IIndexingStrategy`.
    *   Use embedding service to find semantic boundaries in text.
    *   Respect document structure (paragraphs, sections) during chunking.
    *   Support configurable chunk size ranges (min, max, target).
    *   Produce `CHUNKS` and `DATABASE` capabilities.

2.  **Semantic Boundary Detection**
    *   Split document into candidate chunks (sentences/paragraphs).
    *   Embed candidates using the ONNX embedding service.
    *   Calculate cosine similarity between consecutive candidate embeddings.
    *   Identify boundaries where similarity drops below a configurable threshold.

3.  **Chunk Creation & Merging**
    *   Merge candidates into final chunks based on identified boundaries.
    *   Enforce minimum and maximum chunk size constraints.
    *   Aim for a target chunk size for optimal retrieval performance.

4.  **Metadata & Storage**
    *   Store chunks with rich metadata:
        *   `document_id`: Reference to the parent document.
        *   `chunk_index`: Order of the chunk in the document.
        *   `total_chunks`: Total number of chunks for the document.
        *   `strategy`: 'context_aware'.
    *   Persist chunks to the database using the `IndexingContext`.

### Non-Functional Requirements

1.  **Performance**
    *   Processing speed: <1s per 10k words.
    *   Efficient batch embedding of candidates.

2.  **Reliability**
    *   Handle empty documents or documents smaller than min chunk size gracefully.
    *   Ensure all content is preserved (no text loss during chunking).

3.  **Maintainability**
    *   Clean separation of boundary detection logic.
    *   Configurable parameters for tuning (thresholds, sizes).

---

## Acceptance Criteria

### AC1: Strategy Implementation
- [ ] `ContextAwareChunkingIndexing` class exists and implements `IIndexingStrategy`.
- [ ] `produces()` returns `{IndexCapability.CHUNKS, IndexCapability.DATABASE}`.
- [ ] `requires_services()` returns `{ServiceDependency.EMBEDDING, ServiceDependency.DATABASE}`.

### AC2: Boundary Detection
- [ ] System correctly identifies semantic shifts in text (e.g., topic changes).
- [ ] Boundary threshold is configurable via strategy config.
- [ ] Embedding service is used efficiently (batched requests).

### AC3: Chunk Generation
- [ ] Chunks respect `chunk_size_min`, `chunk_size_max`, and `chunk_size_target`.
- [ ] Chunks do not break sentences (unless a single sentence exceeds max size).
- [ ] All document text is accounted for in the generated chunks.

### AC4: Storage & Metadata
- [ ] Chunks are stored in the database via `context.database`.
- [ ] Metadata fields (`document_id`, `chunk_index`, etc.) are correctly populated.

### AC5: Testing
- [ ] Unit tests for `_find_boundaries` logic.
- [ ] Unit tests for `_create_chunks` merging logic.
- [ ] Integration tests with the actual embedding service and database.
- [ ] Performance benchmarks meet targets.

---

## Technical Specifications

### Implementation

```python
from rag_factory.core.indexing import IIndexingStrategy, IndexingContext, IndexingResult
from rag_factory.core.capabilities import IndexCapability
from rag_factory.core.dependencies import ServiceDependency

class ContextAwareChunkingIndexing(IIndexingStrategy):
    """Chunks documents at semantic boundaries using embeddings"""
    
    def produces(self) -> set[IndexCapability]:
        return {
            IndexCapability.CHUNKS,
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
        Chunk documents at semantic boundaries.
        
        Strategy:
        1. Split document into candidate chunks (paragraphs/sentences)
        2. Embed all candidates
        3. Find boundaries where embedding similarity drops
        4. Merge candidates into final chunks within size constraints
        5. Store chunks with metadata
        """
        chunk_size_min = self.config.get('chunk_size_min', 256)
        chunk_size_max = self.config.get('chunk_size_max', 1024)
        chunk_size_target = self.config.get('chunk_size_target', 512)
        
        all_chunks = []
        
        for doc in documents:
            # Step 1: Split into candidate chunks (sentences/paragraphs)
            candidates = self._split_into_candidates(doc.content)
            
            # Step 2: Embed candidates
            embeddings = await self.deps.embedding_service.embed_batch(
                [c.text for c in candidates]
            )
            
            # Step 3: Find semantic boundaries
            boundaries = self._find_boundaries(
                embeddings,
                threshold=self.config.get('boundary_threshold', 0.5)
            )
            
            # Step 4: Create chunks respecting boundaries and size constraints
            chunks = self._create_chunks(
                candidates,
                boundaries,
                chunk_size_min,
                chunk_size_max,
                chunk_size_target
            )
            
            # Add metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata = {
                    'document_id': doc.id,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'strategy': 'context_aware'
                }
            
            all_chunks.extend(chunks)
        
        # Step 5: Store chunks
        await context.database.store_chunks([c.to_dict() for c in all_chunks])
        
        return IndexingResult(
            capabilities=self.produces(),
            metadata={
                'avg_chunk_size': sum(len(c.text) for c in all_chunks) / len(all_chunks),
                'chunk_size_config': {
                    'min': chunk_size_min,
                    'max': chunk_size_max,
                    'target': chunk_size_target
                }
            },
            document_count=len(documents),
            chunk_count=len(all_chunks)
        )
    
    def _split_into_candidates(self, text: str) -> list['ChunkCandidate']:
        """Split text into candidate chunks (paragraphs, sentences)"""
        # Implementation: split on paragraph breaks, then sentences
        pass
    
    def _find_boundaries(
        self,
        embeddings: list[list[float]],
        threshold: float
    ) -> list[int]:
        """Find indices where semantic similarity drops below threshold"""
        import numpy as np
        
        boundaries = [0]
        
        for i in range(1, len(embeddings)):
            # Calculate cosine similarity between consecutive embeddings
            similarity = np.dot(embeddings[i-1], embeddings[i]) / (
                np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )
            
            if similarity < threshold:
                boundaries.append(i)
        
        boundaries.append(len(embeddings))
        return boundaries
    
    def _create_chunks(
        self,
        candidates: list,
        boundaries: list[int],
        min_size: int,
        max_size: int,
        target_size: int
    ) -> list['Chunk']:
        """Merge candidates into chunks respecting boundaries and size"""
        # Implementation: merge candidates between boundaries
        pass
```

### Technical Dependencies
- ONNX embedding service (from Epic 10)
- Database service with chunk storage
