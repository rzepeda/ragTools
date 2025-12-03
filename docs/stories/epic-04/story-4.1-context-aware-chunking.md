# Story 4.1: Implement Context-Aware Chunking Strategy

**Story ID:** 4.1
**Epic:** Epic 4 - Priority RAG Strategies
**Story Points:** 13
**Priority:** Critical (High-Impact Trio)
**Dependencies:** Epic 3 (Embedding Service)

---

## User Story

**As a** system
**I want** to split documents at natural semantic boundaries
**So that** document structure and context are preserved in chunks

---

## Detailed Requirements

### Functional Requirements

1. **Semantic Boundary Detection**
   - Use embedding similarity to detect topic shifts
   - Identify natural break points in text
   - Calculate semantic coherence scores
   - Support configurable similarity thresholds

2. **Document Structure Preservation**
   - Respect document hierarchy (headers, sections, paragraphs)
   - Maintain metadata about document structure
   - Support multiple document formats (markdown, HTML, PDF, plain text)
   - Preserve code blocks, lists, and tables as single units

3. **Hybrid Chunking Approach**
   - Integration with dockling or similar libraries
   - Combine structure-based and semantic chunking
   - Fallback to fixed-size chunking when needed
   - Support custom chunking rules per document type

4. **Configurable Chunk Sizing**
   - Min/max chunk size constraints
   - Target chunk size (soft limit)
   - Overlap between chunks (configurable)
   - Token counting for accurate sizing

5. **Metadata Management**
   - Store chunk position in document
   - Track parent-child relationships
   - Include document structure context
   - Add semantic coherence scores

6. **Strategy Implementation**
   - Implement IRAGStrategy interface
   - Support prepare_data and retrieve methods
   - Integrate with EmbeddingService
   - Store chunks in database via ChunkRepository

### Non-Functional Requirements

1. **Performance**
   - Process 100 pages/minute
   - Embedding generation optimized (batch)
   - Chunk storage optimized (bulk insert)
   - Memory efficient for large documents

2. **Quality**
   - Semantic coherence >0.7 within chunks
   - Structure preservation >90%
   - No information loss at boundaries
   - Reproducible chunking

3. **Maintainability**
   - Clear separation of chunking logic
   - Pluggable chunking algorithms
   - Comprehensive logging
   - Configuration-driven behavior

4. **Scalability**
   - Handle documents up to 1000 pages
   - Support concurrent document processing
   - Efficient memory usage
   - Stream processing for large documents

---

## Acceptance Criteria

### AC1: Semantic Boundary Detection
- [ ] Calculate embedding similarity between consecutive text segments
- [ ] Detect topic shifts based on similarity threshold
- [ ] Identify natural break points (paragraph endings, section breaks)
- [ ] Generate semantic coherence scores for chunks

### AC2: Structure Preservation
- [ ] Parse document structure (headers, sections, paragraphs)
- [ ] Respect structure boundaries in chunking
- [ ] Maintain parent-child relationships
- [ ] Preserve special elements (code, tables, lists)

### AC3: Hybrid Chunking
- [ ] Integrate structure-based chunking
- [ ] Integrate semantic chunking
- [ ] Combine both approaches intelligently
- [ ] Fallback mechanism when structure unclear

### AC4: Chunk Size Management
- [ ] Enforce min/max chunk size constraints
- [ ] Target optimal chunk size
- [ ] Add overlap between adjacent chunks
- [ ] Count tokens accurately

### AC5: Metadata Tracking
- [ ] Store chunk position and order
- [ ] Track document structure context
- [ ] Include parent document reference
- [ ] Add semantic coherence scores

### AC6: Strategy Integration
- [ ] Implement IRAGStrategy interface
- [ ] Integrate with EmbeddingService
- [ ] Store chunks via ChunkRepository
- [ ] Support full RAG workflow

### AC7: Testing
- [ ] Unit tests for chunking logic
- [ ] Integration tests with real documents
- [ ] Performance benchmarks
- [ ] Quality metrics validation

---

## Technical Specifications

### File Structure
```
rag_factory/
├── strategies/
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── base.py                    # Base chunker interface
│   │   ├── semantic.py                # Semantic chunker
│   │   ├── structural.py              # Structure-based chunker
│   │   ├── hybrid.py                  # Hybrid chunker
│   │   └── config.py                  # Chunking config
│   │
│   └── context_aware_rag.py           # Context-aware RAG strategy
│
tests/
├── unit/
│   └── strategies/
│       └── chunking/
│           ├── test_semantic_chunker.py
│           ├── test_structural_chunker.py
│           └── test_hybrid_chunker.py
│
├── integration/
│   └── strategies/
│       └── test_context_aware_rag.py
```

### Example Implementation Outline

```python
# rag_factory/strategies/chunking/semantic.py
class SemanticChunker:
    """Chunks text based on semantic similarity."""
    
    def __init__(self, embedding_service, min_chunk_size=100, 
                 max_chunk_size=1000, similarity_threshold=0.7):
        self.embedding_service = embedding_service
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
    
    def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk text based on semantic boundaries.
        
        1. Split text into sentences
        2. Generate embeddings for sentences
        3. Calculate similarity between consecutive sentences
        4. Identify break points where similarity drops
        5. Group sentences into chunks
        """
        # Implementation details in story
        pass

# rag_factory/strategies/context_aware_rag.py
class ContextAwareRAGStrategy(IRAGStrategy):
    """RAG strategy with context-aware chunking."""
    
    def prepare_data(self, documents: List[Dict]) -> PreparedData:
        """Prepare documents using context-aware chunking."""
        # Use hybrid chunker
        # Generate embeddings
        # Store in database
        pass
    
    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        """Retrieve relevant chunks."""
        # Standard vector search
        pass
```

---

## Key Implementation Details

### 1. Semantic Similarity Calculation

```python
def calculate_semantic_similarity(sentences, embeddings):
    """Calculate cosine similarity between consecutive sentences."""
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i+1])
        similarities.append(sim)
    return similarities

def find_breakpoints(similarities, threshold=0.7):
    """Find indices where similarity drops below threshold."""
    breakpoints = []
    for i, sim in enumerate(similarities):
        if sim < threshold:
            breakpoints.append(i + 1)
    return breakpoints
```

### 2. Structure Parsing

```python
def parse_document_structure(text: str, format: str):
    """Parse document structure based on format."""
    if format == "markdown":
        return parse_markdown(text)
    elif format == "html":
        return parse_html(text)
    else:
        return parse_plain_text(text)

def parse_markdown(text: str):
    """Extract markdown structure."""
    # Parse headers (# ## ###)
    # Identify sections
    # Find paragraphs
    # Detect code blocks
    pass
```

### 3. Hybrid Chunking Strategy

```python
def hybrid_chunk(text, structure, semantic_boundaries):
    """Combine structure and semantic boundaries."""
    # Start with structure-based chunks
    # Refine using semantic boundaries
    # Respect size constraints
    # Add overlap
    pass
```

---

## Unit Tests

### Test Cases

```python
import pytest
from rag_factory.strategies.chunking import SemanticChunker, StructuralChunker

def test_semantic_chunker_basic():
    """Test semantic chunker with simple text."""
    chunker = SemanticChunker(mock_embedding_service)
    text = "Sentence 1. Sentence 2. Different topic. Sentence 4."
    
    chunks = chunker.chunk(text)
    
    # Should create 2 chunks based on topic shift
    assert len(chunks) >= 2

def test_respect_min_max_size():
    """Test chunk size constraints."""
    chunker = SemanticChunker(
        mock_embedding_service,
        min_chunk_size=50,
        max_chunk_size=200
    )
    
    chunks = chunker.chunk(long_text)
    
    for chunk in chunks:
        assert len(chunk.text) >= 50
        assert len(chunk.text) <= 200

def test_structural_chunker_markdown():
    """Test structure-based chunking for markdown."""
    chunker = StructuralChunker()
    markdown_text = """
# Section 1
Paragraph 1.

## Subsection 1.1
Paragraph 2.

# Section 2
Paragraph 3.
"""
    
    chunks = chunker.chunk(markdown_text, format="markdown")
    
    # Should respect section boundaries
    assert len(chunks) >= 2
    assert "Section 1" in chunks[0].text
    assert "Section 2" in chunks[-1].text

def test_preserve_code_blocks():
    """Test that code blocks are not split."""
    chunker = StructuralChunker()
    text = """
Some text before.

```python
def function():
    return True
```

Some text after.
"""
    
    chunks = chunker.chunk(text, format="markdown")
    
    # Code block should be in single chunk
    code_chunks = [c for c in chunks if "def function" in c.text]
    assert len(code_chunks) == 1
    assert "return True" in code_chunks[0].text

def test_chunk_metadata():
    """Test chunk metadata is populated."""
    chunker = SemanticChunker(mock_embedding_service)
    chunks = chunker.chunk("Test text")
    
    assert chunks[0].metadata["chunk_index"] == 0
    assert "coherence_score" in chunks[0].metadata
    assert chunks[0].metadata["chunking_method"] == "semantic"
```

---

## Integration Tests

```python
@pytest.mark.integration
def test_context_aware_rag_full_workflow():
    """Test full workflow with context-aware chunking."""
    from rag_factory.strategies import ContextAwareRAGStrategy
    
    strategy = ContextAwareRAGStrategy(config)
    
    # Prepare documents
    documents = [
        {"text": sample_document, "metadata": {"source": "test.md"}}
    ]
    
    prepared = strategy.prepare_data(documents)
    
    # Verify chunks are semantically coherent
    assert len(prepared.chunks) > 0
    for chunk in prepared.chunks:
        assert chunk.metadata.get("coherence_score", 0) > 0.7
    
    # Test retrieval
    results = strategy.retrieve("test query", top_k=5)
    assert len(results) <= 5

@pytest.mark.integration
def test_large_document_processing():
    """Test processing large documents."""
    strategy = ContextAwareRAGStrategy(config)
    
    # 100-page document
    large_doc = {"text": "..." * 100000, "metadata": {}}
    
    import time
    start = time.time()
    prepared = strategy.prepare_data([large_doc])
    duration = time.time() - start
    
    # Should process within reasonable time
    assert duration < 60  # 1 minute for 100 pages
    assert len(prepared.chunks) > 0
```

---

## Performance Benchmarks

```python
@pytest.mark.benchmark
def test_chunking_performance():
    """Test chunking speed."""
    chunker = SemanticChunker(embedding_service)
    
    # 10-page document
    text = load_sample_text(pages=10)
    
    start = time.time()
    chunks = chunker.chunk(text)
    duration = time.time() - start
    
    pages_per_minute = (10 / duration) * 60
    
    # Should process >100 pages/minute
    assert pages_per_minute > 100

@pytest.mark.benchmark
def test_semantic_coherence():
    """Test semantic coherence of chunks."""
    strategy = ContextAwareRAGStrategy(config)
    documents = load_test_documents()
    
    prepared = strategy.prepare_data(documents)
    
    coherence_scores = [
        chunk.metadata.get("coherence_score", 0)
        for chunk in prepared.chunks
    ]
    
    avg_coherence = sum(coherence_scores) / len(coherence_scores)
    
    # Average coherence should be >0.7
    assert avg_coherence > 0.7
```

---

## Definition of Done

- [ ] Semantic chunker implemented
- [ ] Structural chunker implemented
- [ ] Hybrid chunker implemented
- [ ] Context-aware RAG strategy implements IRAGStrategy
- [ ] Integration with EmbeddingService working
- [ ] Integration with ChunkRepository working
- [ ] All unit tests pass (>90% coverage)
- [ ] All integration tests pass
- [ ] Performance benchmarks meet requirements
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] No linting errors

---

## Notes for Developers

1. **Start with structural chunking** - It's simpler and provides a baseline

2. **Semantic chunking is expensive** - Batch embedding generation to optimize

3. **Test with real documents** - Use various formats (markdown, PDF, plain text)

4. **Tune the similarity threshold** - 0.7 is a starting point, may need adjustment

5. **Consider dockling** - It provides hybrid chunking out of the box

6. **Metadata is crucial** - Store enough context to understand chunk provenance

7. **Handle edge cases** - Very short documents, code-heavy documents, etc.

8. **Overlap strategy** - Adding 10-20% overlap helps maintain context

9. **Token counting** - Use tiktoken or similar for accurate token counts

10. **Incremental approach** - Start simple, add sophistication iteratively

