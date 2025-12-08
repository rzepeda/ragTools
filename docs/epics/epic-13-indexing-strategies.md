# Epic 13: Core Indexing Strategies Implementation

**Epic Goal:** Implement essential indexing strategies using the new `IIndexingStrategy` interface and capability system, enabling diverse indexing approaches beyond traditional vector embeddings.

**Epic Story Points Total:** 47

**Dependencies:** Epic 12 (Pipeline Separation - must be complete first)

**Status:** Ready for implementation after Epic 12

---

## Background

With the pipeline separation from Epic 12, we can now implement indexing strategies as first-class, independent components. This epic implements core indexing strategies that produce different capabilities:

- **Vector-based:** Traditional embedding strategies
- **Structural:** Chunking and hierarchy strategies
- **Alternative:** Keyword and graph-based strategies
- **Experimental:** No-storage and in-memory strategies

---

## Story 13.1: Implement Context-Aware Chunking (Indexing)

**As a** system  
**I want** document chunking at natural semantic boundaries  
**So that** chunk quality is optimized for retrieval

**Acceptance Criteria:**
- Implement `ContextAwareChunkingIndexing` as `IIndexingStrategy`
- Use embedding service to find semantic boundaries
- Respect document structure (paragraphs, sections)
- Configurable chunk size ranges (min, max, target)
- Produce `CHUNKS` and `DATABASE` capabilities
- Store chunks with metadata (position, parent document, etc.)
- Integration tests with various document types
- Performance benchmarks

**Implementation:**

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

**Technical Dependencies:**
- ONNX embedding service (from Epic 10)
- Database service with chunk storage

**Story Points:** 13

---

## Story 13.2: Implement Vector Embedding Indexing

**As a** system  
**I want** to create vector embeddings for chunks  
**So that** semantic search is possible

**Acceptance Criteria:**
- Implement `VectorEmbeddingIndexing` as `IIndexingStrategy`
- Load chunks from database (created by chunking strategy)
- Generate embeddings using embedding service
- Store embeddings in vector database
- Produce `VECTORS` and `DATABASE` capabilities
- Support batching for efficiency
- Integration tests with pgvector
- Performance benchmarks

**Implementation:**

```python
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

**Technical Dependencies:**
- ONNX embedding service
- PostgreSQL with pgvector extension

**Story Points:** 8

---

## Story 13.3: Implement Keyword Extraction Indexing

**As a** system  
**I want** keyword-based indexing without embeddings  
**So that** I can support non-ML retrieval approaches

**Acceptance Criteria:**
- Implement `KeywordIndexing` as `IIndexingStrategy`
- Extract keywords using TF-IDF or similar
- Build inverted index
- Produce `KEYWORDS` and `DATABASE` capabilities
- No embedding service required
- Support BM25 ranking
- Integration tests
- Performance benchmarks

**Implementation:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

class KeywordIndexing(IIndexingStrategy):
    """Creates keyword index for BM25/keyword retrieval"""
    
    def produces(self) -> set[IndexCapability]:
        return {
            IndexCapability.KEYWORDS,
            IndexCapability.DATABASE
        }
    
    def requires_services(self) -> set[ServiceDependency]:
        return {
            ServiceDependency.DATABASE  # No embedding or LLM needed!
        }
    
    async def process(
        self,
        documents: list['Document'],
        context: IndexingContext
    ) -> IndexingResult:
        """
        Extract keywords and build inverted index.
        
        Strategy:
        1. Get chunks from database
        2. Extract keywords using TF-IDF
        3. Build inverted index
        4. Store index in database
        """
        # Get chunks
        chunks = await context.database.get_chunks_for_documents(
            [doc.id for doc in documents]
        )
        
        if not chunks:
            raise ValueError("No chunks found. Run chunking strategy first.")
        
        # Extract keywords using TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=self.config.get('max_keywords', 1000),
            stop_words='english',
            ngram_range=(1, 2)  # unigrams and bigrams
        )
        
        texts = [c['text'] for c in chunks]
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Build inverted index
        feature_names = vectorizer.get_feature_names_out()
        inverted_index = {}
        
        for chunk_idx, chunk in enumerate(chunks):
            # Get keywords for this chunk
            chunk_vector = tfidf_matrix[chunk_idx]
            keywords = [
                (feature_names[i], chunk_vector[0, i])
                for i in chunk_vector.nonzero()[1]
            ]
            
            # Add to inverted index
            for keyword, score in keywords:
                if keyword not in inverted_index:
                    inverted_index[keyword] = []
                
                inverted_index[keyword].append({
                    'chunk_id': chunk['id'],
                    'score': float(score)
                })
        
        # Store inverted index
        await context.database.store_keyword_index(inverted_index)
        
        return IndexingResult(
            capabilities=self.produces(),
            metadata={
                'total_keywords': len(inverted_index),
                'avg_keywords_per_chunk': len(feature_names) / len(chunks),
                'method': 'tfidf'
            },
            document_count=len(documents),
            chunk_count=len(chunks)
        )
```

**Technical Dependencies:**
- scikit-learn (for TF-IDF)
- Database service

**Story Points:** 8

---

## Story 13.4: Implement Hierarchical Indexing

**As a** system  
**I want** parent-child chunk relationships  
**So that** retrieval can expand context

**Acceptance Criteria:**
- Implement `HierarchicalIndexing` as `IIndexingStrategy`
- Create multi-level chunk hierarchy (document → section → paragraph)
- Store parent-child relationships as metadata
- Produce `CHUNKS`, `HIERARCHY`, and `DATABASE` capabilities
- Support configurable hierarchy depth
- Integration tests
- Performance benchmarks

**Implementation:**

```python
class HierarchicalIndexing(IIndexingStrategy):
    """Creates hierarchical chunk relationships"""
    
    def produces(self) -> set[IndexCapability]:
        return {
            IndexCapability.CHUNKS,
            IndexCapability.HIERARCHY,
            IndexCapability.DATABASE
        }
    
    def requires_services(self) -> set[ServiceDependency]:
        return {
            ServiceDependency.DATABASE
        }
    
    async def process(
        self,
        documents: list['Document'],
        context: IndexingContext
    ) -> IndexingResult:
        """
        Create hierarchical chunks with parent-child relationships.
        
        Hierarchy levels:
        - Level 0: Full document
        - Level 1: Sections (e.g., headings)
        - Level 2: Paragraphs
        - Level 3: Sentences (optional)
        """
        max_depth = self.config.get('max_depth', 2)
        all_chunks = []
        
        for doc in documents:
            # Create hierarchy
            hierarchy = self._build_hierarchy(doc.content, max_depth)
            
            # Flatten and assign IDs with relationships
            chunks = self._flatten_hierarchy(hierarchy, doc.id)
            all_chunks.extend(chunks)
        
        # Store chunks with hierarchy metadata
        await context.database.store_chunks_with_hierarchy(all_chunks)
        
        return IndexingResult(
            capabilities=self.produces(),
            metadata={
                'max_depth': max_depth,
                'avg_chunks_per_doc': len(all_chunks) / len(documents)
            },
            document_count=len(documents),
            chunk_count=len(all_chunks)
        )
    
    def _build_hierarchy(self, text: str, max_depth: int):
        """Build hierarchical structure from text"""
        # Level 0: Full document
        hierarchy = {
            'level': 0,
            'text': text,
            'children': []
        }
        
        if max_depth >= 1:
            # Level 1: Split by sections (headings)
            sections = self._split_by_headings(text)
            for section in sections:
                section_node = {
                    'level': 1,
                    'text': section,
                    'children': []
                }
                
                if max_depth >= 2:
                    # Level 2: Split by paragraphs
                    paragraphs = self._split_by_paragraphs(section)
                    for para in paragraphs:
                        para_node = {
                            'level': 2,
                            'text': para,
                            'children': []
                        }
                        section_node['children'].append(para_node)
                
                hierarchy['children'].append(section_node)
        
        return hierarchy
    
    def _flatten_hierarchy(self, hierarchy, doc_id):
        """Flatten hierarchy into chunks with parent references"""
        chunks = []
        
        def traverse(node, parent_id=None, path=[]):
            chunk_id = f"{doc_id}_{'_'.join(map(str, path))}"
            
            chunk = {
                'id': chunk_id,
                'document_id': doc_id,
                'text': node['text'],
                'level': node['level'],
                'parent_id': parent_id,
                'path': path.copy()
            }
            chunks.append(chunk)
            
            # Traverse children
            for i, child in enumerate(node['children']):
                traverse(child, chunk_id, path + [i])
        
        traverse(hierarchy)
        return chunks
    
    def _split_by_headings(self, text: str) -> list[str]:
        """Split text by markdown/HTML headings"""
        # Implementation
        pass
    
    def _split_by_paragraphs(self, text: str) -> list[str]:
        """Split text by paragraph breaks"""
        return [p.strip() for p in text.split('\n\n') if p.strip()]
```

**Technical Dependencies:**
- Database service with hierarchy support

**Story Points:** 13

---

## Story 13.5: Implement In-Memory Indexing (Testing)

**As a** developer  
**I want** in-memory only indexing for unit tests  
**So that** tests don't require database setup

**Acceptance Criteria:**
- Implement `InMemoryIndexing` as `IIndexingStrategy`
- Store chunks in memory only (Python dict)
- Produce `CHUNKS` and `IN_MEMORY` capabilities
- No database service required
- Data lost on process restart (expected behavior)
- Support for test fixtures
- Documentation for testing usage

**Implementation:**

```python
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

**Usage in Tests:**

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

**Story Points:** 5

---

## Sprint Planning

**Sprint 16:** Stories 13.1, 13.2 (21 points)  
**Sprint 17:** Stories 13.3, 13.4, 13.5 (26 points)

---

## Indexing Strategy Compatibility Matrix

### Strategies and Their Outputs

| Strategy | VECTORS | KEYWORDS | GRAPH | CHUNKS | HIERARCHY | IN_MEMORY | DATABASE |
|----------|---------|----------|-------|--------|-----------|-----------|----------|
| Context-Aware Chunking | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ |
| Vector Embedding | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Keyword Extraction | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Hierarchical Indexing | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| In-Memory | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |

### Common Pipeline Combinations

**Standard Vector RAG:**
```python
indexing = factory.create_indexing_pipeline(
    ["context_aware_chunking", "vector_embedding"],
    [{}, {}]
)
# Produces: {CHUNKS, VECTORS, DATABASE}
```

**Hybrid Search:**
```python
indexing = factory.create_indexing_pipeline(
    ["context_aware_chunking", "vector_embedding", "keyword_extraction"],
    [{}, {}, {}]
)
# Produces: {CHUNKS, VECTORS, KEYWORDS, DATABASE}
```

**Hierarchical Vector RAG:**
```python
indexing = factory.create_indexing_pipeline(
    ["hierarchical_indexing", "vector_embedding"],
    [{}, {}]
)
# Produces: {CHUNKS, HIERARCHY, VECTORS, DATABASE}
```

**Keyword-Only (No ML):**
```python
indexing = factory.create_indexing_pipeline(
    ["context_aware_chunking", "keyword_extraction"],
    [{}, {}]
)
# Produces: {CHUNKS, KEYWORDS, DATABASE}
# No embedding service required!
```

**Testing (In-Memory):**
```python
indexing = factory.create_indexing_pipeline(
    ["in_memory_indexing"],
    [{}]
)
# Produces: {CHUNKS, IN_MEMORY}
# No database required!
```

---

## Testing Strategy

### Unit Tests
- Each strategy tested in isolation
- Mock dependencies
- Capability validation
- Service requirement validation

### Integration Tests
- Multi-strategy pipelines
- Real database interactions
- Performance benchmarks
- Capability aggregation

### Performance Targets
- Context-aware chunking: <1s per 10k words
- Vector embedding: <100ms per document (batch)
- Keyword extraction: <500ms per 10k words
- Hierarchical indexing: <2s per 10k words

---

## Documentation Updates

- [ ] Indexing strategy guide
- [ ] Capability reference
- [ ] Pipeline composition guide
- [ ] Performance tuning guide
- [ ] Testing guide (using in-memory strategy)
- [ ] Migration guide (from old to new strategies)

---

## Success Criteria

- [ ] All 5 indexing strategies implemented
- [ ] All strategies implement `IIndexingStrategy`
- [ ] All strategies declare capabilities correctly
- [ ] All strategies declare service requirements correctly
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Performance targets met
- [ ] Documentation complete
- [ ] Can create pipelines with multiple strategies
- [ ] Capability validation working

---

## Future Indexing Strategies (Epic 14+)

**Late Chunking** (Epic 14)
- Produces: VECTORS, LATE_CHUNKS, DATABASE
- Requires: EMBEDDING, DATABASE

**Contextual Enrichment** (Epic 14)
- Produces: CHUNKS, CONTEXTUAL, DATABASE
- Requires: LLM, DATABASE

**Knowledge Graph** (Epic 14)
- Produces: GRAPH, DATABASE
- Requires: LLM, GRAPH, DATABASE

**Full Document Storage** (Epic 14)
- Produces: FULL_DOCUMENT, DATABASE
- Requires: DATABASE

---

## Benefits Achieved

**Modularity:**
- ✅ Each indexing approach is independent
- ✅ Strategies can be combined freely
- ✅ Easy to add new strategies

**Flexibility:**
- ✅ Support vector, keyword, and hybrid approaches
- ✅ Support hierarchical structures
- ✅ Support testing without infrastructure

**Clarity:**
- ✅ Clear capability declarations
- ✅ Explicit service requirements
- ✅ Self-documenting strategies

**Quality:**
- ✅ Semantic boundary detection
- ✅ Configurable chunk sizes
- ✅ Metadata preservation
