# Story 13.5: Implement In-Memory Indexing (Testing)

**Story ID:** 13.5
**Epic:** Epic 13 - Core Indexing Strategies Implementation
**Story Points:** 5
**Priority:** Low
**Dependencies:** None

---

## User Story

**As a** developer
**I want** in-memory only indexing for unit tests
**So that** tests don't require database setup

---

## Detailed Requirements

### Functional Requirements

1.  **In-Memory Indexing Strategy**
    *   Implement `InMemoryIndexing` class implementing `IIndexingStrategy`.
    *   Store chunks in a class-level dictionary (shared state for testing).
    *   Produce `CHUNKS` and `IN_MEMORY` capabilities.
    *   Require NO external services (no database, no embedding).

2.  **Simple Chunking**
    *   Implement basic chunking by character count or whitespace.
    *   Support configurable chunk size.

3.  **Test Utilities**
    *   Provide methods to clear storage (`clear_storage`).
    *   Provide methods to inspect stored chunks (`get_chunk`).

### Non-Functional Requirements

1.  **Speed**
    *   Must be extremely fast for unit tests.

2.  **Isolation**
    *   Must not persist data to disk.
    *   Data should be volatile (lost on process exit).

---

## Acceptance Criteria

### AC1: Strategy Implementation
- [ ] `InMemoryIndexing` class exists and implements `IIndexingStrategy`.
- [ ] `produces()` returns `{IndexCapability.CHUNKS, IndexCapability.IN_MEMORY}`.
- [ ] `requires_services()` returns empty set.

### AC2: In-Memory Storage
- [ ] Chunks are stored in memory.
- [ ] Storage can be accessed and verified in tests.
- [ ] Storage can be cleared between tests.

### AC3: Testing Usage
- [ ] Strategy can be used in a pipeline without a database connection.
- [ ] Documentation provided on how to use for testing.

---

## Technical Specifications

### Implementation

```python
from rag_factory.core.indexing import IIndexingStrategy, IndexingContext, IndexingResult
from rag_factory.core.capabilities import IndexCapability
from rag_factory.core.dependencies import ServiceDependency

class InMemoryIndexing(IIndexingStrategy):
    """Stores chunks in memory for testing"""
    
    # Class-level storage (shared across instances for testing)
    _storage = {}
    
    def produces(self) -> set[IndexCapability]:
        return {
            IndexCapability.CHUNKS,
            IndexCapability.IN_MEMORY
        }
    
    def requires_services(self) -> set[ServiceDependency]:
        return set()  # No services required!
    
    async def process(
        self,
        documents: list['Document'],
        context: IndexingContext
    ) -> IndexingResult:
        """
        Store chunks in memory for fast testing.
        """
        chunk_size = self.config.get('chunk_size', 512)
        all_chunks = []
        
        for doc in documents:
            # Simple chunking by character count
            chunks = self._chunk_by_size(doc.content, chunk_size)
            
            for i, chunk_text in enumerate(chunks):
                chunk = {
                    'id': f"{doc.id}_chunk_{i}",
                    'document_id': doc.id,
                    'text': chunk_text,
                    'index': i
                }
                all_chunks.append(chunk)
        
        # Store in class-level dict
        for chunk in all_chunks:
            self._storage[chunk['id']] = chunk
        
        return IndexingResult(
            capabilities=self.produces(),
            metadata={
                'storage_type': 'in_memory',
                'chunk_size': chunk_size
            },
            document_count=len(documents),
            chunk_count=len(all_chunks)
        )
    
    @classmethod
    def clear_storage(cls):
        """Clear in-memory storage (for test cleanup)"""
        cls._storage.clear()
    
    @classmethod
    def get_chunk(cls, chunk_id: str):
        """Get chunk from memory"""
        return cls._storage.get(chunk_id)
    
    def _chunk_by_size(self, text: str, size: int) -> list[str]:
        """Simple chunking by character count"""
        chunks = []
        for i in range(0, len(text), size):
            chunks.append(text[i:i + size])
        return chunks
```

### Usage in Tests

```python
def test_retrieval_pipeline():
    # Use in-memory indexing (no database needed)
    indexing = InMemoryIndexing(config={}, dependencies=StrategyDependencies())
    
    # Index test documents
    result = await indexing.process(test_documents, context)
    
    # Test retrieval
    # ...
    
    # Cleanup
    InMemoryIndexing.clear_storage()
```
