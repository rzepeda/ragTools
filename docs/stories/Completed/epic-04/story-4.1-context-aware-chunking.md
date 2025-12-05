# Story 4.1: Implement Context-Aware Chunking Strategy

**Story ID:** 4.1
**Epic:** Epic 4 - Priority RAG Strategies
**Story Points:** 13
**Priority:** Critical
**Dependencies:** Epic 3 (Embedding Service, LLM Service)

---

## User Story

**As a** system
**I want** to split documents at natural boundaries
**So that** document structure is preserved and chunks are semantically coherent

---

## Detailed Requirements

### Functional Requirements

1. **Semantic Boundary Detection**
   - Use embedding model to identify semantic shifts in content
   - Calculate similarity scores between consecutive text segments
   - Identify natural break points based on semantic discontinuity
   - Support configurable similarity threshold for boundary detection

2. **Document Structure Preservation**
   - Respect markdown headers (h1, h2, h3, etc.)
   - Preserve section boundaries
   - Maintain paragraph integrity
   - Recognize list structures (ordered and unordered)
   - Detect code blocks and keep them intact
   - Handle tables as atomic units

3. **Hybrid Chunking with Dockling**
   - Integrate dockling library for advanced document parsing
   - Support PDF document layout analysis
   - Extract document structure (headers, paragraphs, tables, figures)
   - Use dockling's hybrid chunking capabilities
   - Fallback to basic chunking if dockling fails

4. **Configurable Chunk Size Ranges**
   - Minimum chunk size (e.g., 128 tokens)
   - Maximum chunk size (e.g., 1024 tokens)
   - Target chunk size (e.g., 512 tokens)
   - Allow chunks to exceed max size for atomic content (code blocks, tables)
   - Configurable token counting method (tiktoken, simple word count)

5. **Metadata Preservation**
   - Track original document structure (section hierarchy)
   - Store chunk position in original document
   - Maintain source document references
   - Record chunk creation method (semantic, structural, hybrid)
   - Include parent-child relationships for hierarchical content

6. **Multiple Chunking Strategies**
   - **Semantic Chunking**: Split based on embedding similarity
   - **Structural Chunking**: Split based on document structure
   - **Hybrid Chunking**: Combine semantic and structural approaches
   - **Fixed-Size Chunking**: Traditional fixed-size chunks (baseline)
   - Strategy selection via configuration

### Non-Functional Requirements

1. **Performance**
   - Process documents at >100 chunks/second
   - Embedding similarity calculation <50ms per comparison
   - Total chunking time <500ms for typical document (10 pages)
   - Support batch processing of multiple documents

2. **Quality**
   - Chunks should maintain semantic coherence (measurable via intra-chunk similarity)
   - Minimize orphaned sentences or incomplete thoughts
   - Preserve context by including surrounding information in metadata
   - Support chunk overlap for context preservation

3. **Flexibility**
   - Support multiple document formats (markdown, PDF, plain text, HTML)
   - Pluggable chunking strategies via strategy pattern
   - Easy to add new chunking approaches
   - Configuration-driven behavior

4. **Robustness**
   - Handle malformed documents gracefully
   - Fallback to simpler chunking if advanced methods fail
   - Comprehensive error handling and logging
   - Validate chunk quality metrics

5. **Observability**
   - Log chunking decisions (why chunks were created)
   - Track chunk size distribution
   - Monitor semantic coherence scores
   - Report chunking strategy effectiveness

---

## Acceptance Criteria

### AC1: Semantic Boundary Detection
- [ ] Embedding-based similarity calculation implemented
- [ ] Boundary detection identifies semantic shifts
- [ ] Configurable similarity threshold (0.0-1.0)
- [ ] Boundary detection works with embedding service
- [ ] Performance: <50ms per boundary comparison

### AC2: Document Structure Preservation
- [ ] Markdown headers recognized and respected
- [ ] Section boundaries preserved
- [ ] Paragraphs remain intact (not split mid-paragraph)
- [ ] Code blocks kept as single chunks
- [ ] Tables handled as atomic units
- [ ] Lists properly structured

### AC3: Dockling Integration
- [ ] Dockling library integrated for PDF processing
- [ ] Document layout analysis working
- [ ] Hybrid chunking leverages dockling capabilities
- [ ] Fallback to basic chunking on dockling errors
- [ ] Support for PDF, DOCX, and other formats via dockling

### AC4: Configurable Chunk Sizes
- [ ] Minimum, maximum, and target chunk sizes configurable
- [ ] Token counting accurate (using tiktoken)
- [ ] Chunks respect size constraints (with exceptions for atomic content)
- [ ] Chunk overlap configurable
- [ ] Statistics on chunk size distribution available

### AC5: Metadata Tracking
- [ ] Document structure hierarchy tracked
- [ ] Chunk position in document recorded
- [ ] Source document references maintained
- [ ] Chunking method logged for each chunk
- [ ] Parent-child relationships stored for nested content

### AC6: Multiple Strategies
- [ ] Semantic chunking strategy implemented
- [ ] Structural chunking strategy implemented
- [ ] Hybrid chunking strategy implemented
- [ ] Fixed-size chunking strategy (baseline) implemented
- [ ] Strategy selection via configuration
- [ ] Easy to add new strategies (strategy pattern)

### AC7: Quality Metrics
- [ ] Intra-chunk semantic similarity calculated
- [ ] Chunk coherence scores available
- [ ] Size distribution metrics tracked
- [ ] Comparison metrics between strategies
- [ ] Quality benchmarks documented

### AC8: Testing
- [ ] Unit tests for all chunking strategies (>90% coverage)
- [ ] Integration tests with real documents
- [ ] Performance benchmarks meet requirements
- [ ] Quality tests validate chunk coherence
- [ ] Edge case handling tested (empty docs, malformed content)

---

## Technical Specifications

### File Structure
```
rag_factory/
├── strategies/
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── base.py                   # Base chunker interface
│   │   ├── semantic_chunker.py       # Semantic boundary detection
│   │   ├── structural_chunker.py     # Document structure-based
│   │   ├── hybrid_chunker.py         # Combined approach
│   │   ├── fixed_size_chunker.py     # Baseline fixed-size
│   │   ├── dockling_chunker.py       # Dockling integration
│   │   ├── config.py                 # Chunking configuration
│   │   └── utils.py                  # Shared utilities
│
tests/
├── unit/
│   └── strategies/
│       └── chunking/
│           ├── test_semantic_chunker.py
│           ├── test_structural_chunker.py
│           ├── test_hybrid_chunker.py
│           ├── test_fixed_size_chunker.py
│           └── test_dockling_chunker.py
│
├── integration/
│   └── strategies/
│       └── test_chunking_integration.py
│
├── fixtures/
│   └── documents/
│       ├── sample.md
│       ├── sample.pdf
│       ├── sample.txt
│       └── sample_complex.md
```

### Dependencies
```python
# requirements.txt additions
dockling>=1.0.0                  # Advanced document parsing
tiktoken==0.5.2                  # Token counting
beautifulsoup4==4.12.0           # HTML parsing
pypdf2==3.0.0                    # PDF text extraction (fallback)
python-magic==0.4.27             # File type detection
```

### Base Chunker Interface
```python
# rag_factory/strategies/chunking/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class ChunkingMethod(Enum):
    """Enumeration of chunking methods."""
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    HYBRID = "hybrid"
    FIXED_SIZE = "fixed_size"
    DOCKLING = "dockling"

@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    chunk_id: str
    source_document_id: str
    position: int  # Position in document (0-indexed)
    start_char: int
    end_char: int
    section_hierarchy: List[str]  # e.g., ["Chapter 1", "Section 1.1"]
    chunking_method: ChunkingMethod
    token_count: int
    coherence_score: Optional[float] = None
    parent_chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    """A document chunk with text and metadata."""
    text: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None

@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""
    method: ChunkingMethod = ChunkingMethod.HYBRID
    min_chunk_size: int = 128  # tokens
    max_chunk_size: int = 1024  # tokens
    target_chunk_size: int = 512  # tokens
    chunk_overlap: int = 50  # tokens

    # Semantic chunking settings
    similarity_threshold: float = 0.7  # 0.0-1.0
    use_embeddings: bool = True

    # Structural chunking settings
    respect_headers: bool = True
    respect_paragraphs: bool = True
    keep_code_blocks_intact: bool = True
    keep_tables_intact: bool = True

    # Dockling settings
    use_dockling: bool = True
    dockling_fallback: bool = True

    # General settings
    compute_coherence_scores: bool = True
    preserve_metadata: bool = True

    # Additional config
    extra_config: Dict[str, Any] = field(default_factory=dict)

class IChunker(ABC):
    """Abstract base class for document chunking strategies."""

    def __init__(self, config: ChunkingConfig):
        """Initialize chunker with configuration."""
        self.config = config

    @abstractmethod
    def chunk_document(self, document: str, document_id: str) -> List[Chunk]:
        """
        Chunk a document into semantically coherent pieces.

        Args:
            document: The document text to chunk
            document_id: Unique identifier for the document

        Returns:
            List of Chunk objects with text and metadata
        """
        pass

    @abstractmethod
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[List[Chunk]]:
        """
        Chunk multiple documents in batch.

        Args:
            documents: List of dicts with 'text' and 'id' keys

        Returns:
            List of chunk lists, one per document
        """
        pass

    def validate_chunks(self, chunks: List[Chunk]) -> bool:
        """Validate that chunks meet quality criteria."""
        for chunk in chunks:
            token_count = chunk.metadata.token_count
            if token_count < self.config.min_chunk_size:
                # Allow small chunks for atomic content
                if not self._is_atomic_content(chunk.text):
                    return False
            if token_count > self.config.max_chunk_size:
                # Allow oversized chunks for atomic content
                if not self._is_atomic_content(chunk.text):
                    return False
        return True

    def _is_atomic_content(self, text: str) -> bool:
        """Check if content should be kept as atomic unit (code, table, etc.)."""
        # Check for code blocks
        if text.strip().startswith("```") or text.strip().startswith("    "):
            return True
        # Check for tables
        if "|" in text and text.count("|") > 3:
            return True
        return False

    def get_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about chunk distribution."""
        if not chunks:
            return {}

        sizes = [c.metadata.token_count for c in chunks]
        coherence_scores = [c.metadata.coherence_score for c in chunks
                          if c.metadata.coherence_score is not None]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(sizes) / len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "avg_coherence": sum(coherence_scores) / len(coherence_scores) if coherence_scores else None,
            "chunking_method": self.config.method.value
        }
```

### Semantic Chunker Implementation
```python
# rag_factory/strategies/chunking/semantic_chunker.py
from typing import List, Dict, Any, Optional
import tiktoken
import numpy as np
from .base import IChunker, Chunk, ChunkMetadata, ChunkingConfig, ChunkingMethod
from ...services.embedding.service import EmbeddingService

class SemanticChunker(IChunker):
    """
    Chunks documents based on semantic similarity between segments.
    Uses embeddings to detect topic shifts and create semantically coherent chunks.
    """

    def __init__(self, config: ChunkingConfig, embedding_service: EmbeddingService):
        super().__init__(config)
        self.embedding_service = embedding_service
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk_document(self, document: str, document_id: str) -> List[Chunk]:
        """Chunk document using semantic boundary detection."""
        # Step 1: Split into sentences
        sentences = self._split_into_sentences(document)

        if not sentences:
            return []

        # Step 2: Group sentences into segments for embedding
        segments = self._create_segments(sentences, segment_size=3)

        # Step 3: Generate embeddings for segments
        segment_texts = [" ".join(seg) for seg in segments]
        embedding_result = self.embedding_service.embed(segment_texts)
        embeddings = embedding_result.embeddings

        # Step 4: Calculate similarity between consecutive segments
        boundaries = self._detect_boundaries(embeddings)

        # Step 5: Create chunks from boundaries
        chunks = self._create_chunks_from_boundaries(
            sentences, boundaries, document_id
        )

        # Step 6: Compute coherence scores if configured
        if self.config.compute_coherence_scores:
            chunks = self._compute_coherence_scores(chunks)

        return chunks

    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[List[Chunk]]:
        """Chunk multiple documents."""
        return [self.chunk_document(doc["text"], doc["id"]) for doc in documents]

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting (can be improved with nltk)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_segments(self, sentences: List[str], segment_size: int = 3) -> List[List[str]]:
        """Group sentences into overlapping segments for embedding."""
        segments = []
        for i in range(len(sentences)):
            segment = sentences[i:i + segment_size]
            if segment:
                segments.append(segment)
        return segments

    def _detect_boundaries(self, embeddings: List[List[float]]) -> List[int]:
        """
        Detect semantic boundaries based on embedding similarity.

        Returns list of indices where boundaries should be placed.
        """
        boundaries = [0]  # Start with first sentence

        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(embeddings[i], embeddings[i + 1])

            # If similarity drops below threshold, it's a boundary
            if similarity < self.config.similarity_threshold:
                boundaries.append(i + 1)

        boundaries.append(len(embeddings))  # End boundary
        return boundaries

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _create_chunks_from_boundaries(
        self,
        sentences: List[str],
        boundaries: List[int],
        document_id: str
    ) -> List[Chunk]:
        """Create chunks from detected boundaries."""
        chunks = []

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)

            # Count tokens
            token_count = len(self.tokenizer.encode(chunk_text))

            # Create metadata
            metadata = ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{i}",
                source_document_id=document_id,
                position=i,
                start_char=0,  # Would need to track actual character positions
                end_char=len(chunk_text),
                section_hierarchy=[],
                chunking_method=ChunkingMethod.SEMANTIC,
                token_count=token_count
            )

            chunks.append(Chunk(text=chunk_text, metadata=metadata))

        # Filter and merge chunks based on size constraints
        chunks = self._adjust_chunk_sizes(chunks)

        return chunks

    def _adjust_chunk_sizes(self, chunks: List[Chunk]) -> List[Chunk]:
        """Adjust chunk sizes to meet min/max constraints."""
        adjusted = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]

            # If chunk is too small, try to merge with next
            if current_chunk.metadata.token_count < self.config.min_chunk_size:
                if i + 1 < len(chunks):
                    # Merge with next chunk
                    next_chunk = chunks[i + 1]
                    merged_text = current_chunk.text + " " + next_chunk.text
                    merged_tokens = current_chunk.metadata.token_count + next_chunk.metadata.token_count

                    current_chunk.text = merged_text
                    current_chunk.metadata.token_count = merged_tokens
                    current_chunk.metadata.end_char = next_chunk.metadata.end_char
                    i += 1  # Skip next chunk since we merged it

            # If chunk is too large, split it
            elif current_chunk.metadata.token_count > self.config.max_chunk_size:
                # Use fixed-size splitting for oversized chunks
                sub_chunks = self._split_large_chunk(current_chunk)
                adjusted.extend(sub_chunks)
                i += 1
                continue

            adjusted.append(current_chunk)
            i += 1

        return adjusted

    def _split_large_chunk(self, chunk: Chunk) -> List[Chunk]:
        """Split a large chunk into smaller chunks."""
        # Simple fixed-size splitting for oversized chunks
        sentences = self._split_into_sentences(chunk.text)
        sub_chunks = []
        current_text = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))

            if current_tokens + sentence_tokens > self.config.target_chunk_size:
                # Create chunk from accumulated sentences
                if current_text:
                    text = " ".join(current_text)
                    metadata = ChunkMetadata(
                        chunk_id=f"{chunk.metadata.chunk_id}_sub_{len(sub_chunks)}",
                        source_document_id=chunk.metadata.source_document_id,
                        position=chunk.metadata.position,
                        start_char=0,
                        end_char=len(text),
                        section_hierarchy=chunk.metadata.section_hierarchy,
                        chunking_method=ChunkingMethod.SEMANTIC,
                        token_count=current_tokens,
                        parent_chunk_id=chunk.metadata.chunk_id
                    )
                    sub_chunks.append(Chunk(text=text, metadata=metadata))

                current_text = [sentence]
                current_tokens = sentence_tokens
            else:
                current_text.append(sentence)
                current_tokens += sentence_tokens

        # Add remaining text
        if current_text:
            text = " ".join(current_text)
            metadata = ChunkMetadata(
                chunk_id=f"{chunk.metadata.chunk_id}_sub_{len(sub_chunks)}",
                source_document_id=chunk.metadata.source_document_id,
                position=chunk.metadata.position,
                start_char=0,
                end_char=len(text),
                section_hierarchy=chunk.metadata.section_hierarchy,
                chunking_method=ChunkingMethod.SEMANTIC,
                token_count=current_tokens,
                parent_chunk_id=chunk.metadata.chunk_id
            )
            sub_chunks.append(Chunk(text=text, metadata=metadata))

        return sub_chunks

    def _compute_coherence_scores(self, chunks: List[Chunk]) -> List[Chunk]:
        """Compute intra-chunk coherence scores."""
        for chunk in chunks:
            # Split chunk into sentences and compute internal similarity
            sentences = self._split_into_sentences(chunk.text)
            if len(sentences) > 1:
                # Get embeddings for sentences
                result = self.embedding_service.embed(sentences)
                embeddings = result.embeddings

                # Calculate average pairwise similarity
                similarities = []
                for i in range(len(embeddings) - 1):
                    sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
                    similarities.append(sim)

                chunk.metadata.coherence_score = sum(similarities) / len(similarities) if similarities else 1.0
            else:
                chunk.metadata.coherence_score = 1.0

        return chunks
```

### Structural Chunker Implementation
```python
# rag_factory/strategies/chunking/structural_chunker.py
from typing import List, Dict, Any, Optional
import re
import tiktoken
from .base import IChunker, Chunk, ChunkMetadata, ChunkingConfig, ChunkingMethod

class StructuralChunker(IChunker):
    """
    Chunks documents based on document structure (headers, paragraphs, etc.).
    Preserves the natural organization of the document.
    """

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk_document(self, document: str, document_id: str) -> List[Chunk]:
        """Chunk document based on structural elements."""
        # Detect document type
        if self._is_markdown(document):
            return self._chunk_markdown(document, document_id)
        else:
            return self._chunk_plain_text(document, document_id)

    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[List[Chunk]]:
        """Chunk multiple documents."""
        return [self.chunk_document(doc["text"], doc["id"]) for doc in documents]

    def _is_markdown(self, text: str) -> bool:
        """Check if document appears to be markdown."""
        # Simple heuristic: look for markdown headers
        return bool(re.search(r'^#{1,6}\s+', text, re.MULTILINE))

    def _chunk_markdown(self, document: str, document_id: str) -> List[Chunk]:
        """Chunk markdown document preserving structure."""
        chunks = []

        # Split by headers
        sections = self._split_by_headers(document)

        for i, section in enumerate(sections):
            header = section.get("header", "")
            content = section.get("content", "")
            level = section.get("level", 0)
            hierarchy = section.get("hierarchy", [])

            # Combine header with content
            full_text = f"{header}\n{content}".strip() if header else content.strip()

            if not full_text:
                continue

            token_count = len(self.tokenizer.encode(full_text))

            # If section is too large, split further
            if token_count > self.config.max_chunk_size:
                sub_chunks = self._split_large_section(
                    full_text, document_id, i, hierarchy
                )
                chunks.extend(sub_chunks)
            else:
                metadata = ChunkMetadata(
                    chunk_id=f"{document_id}_chunk_{i}",
                    source_document_id=document_id,
                    position=i,
                    start_char=0,
                    end_char=len(full_text),
                    section_hierarchy=hierarchy,
                    chunking_method=ChunkingMethod.STRUCTURAL,
                    token_count=token_count
                )
                chunks.append(Chunk(text=full_text, metadata=metadata))

        return chunks

    def _split_by_headers(self, document: str) -> List[Dict[str, Any]]:
        """Split markdown document by headers, preserving hierarchy."""
        lines = document.split("\n")
        sections = []
        current_section = {"content": [], "header": "", "level": 0, "hierarchy": []}
        hierarchy_stack = []

        for line in lines:
            # Check for markdown header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                # Save previous section if it has content
                if current_section["content"] or current_section["header"]:
                    current_section["content"] = "\n".join(current_section["content"])
                    sections.append(current_section)

                # Start new section
                level = len(header_match.group(1))
                header_text = header_match.group(2).strip()

                # Update hierarchy stack
                while hierarchy_stack and hierarchy_stack[-1]["level"] >= level:
                    hierarchy_stack.pop()

                hierarchy_stack.append({"level": level, "text": header_text})
                hierarchy = [h["text"] for h in hierarchy_stack]

                current_section = {
                    "content": [],
                    "header": line,
                    "level": level,
                    "hierarchy": hierarchy
                }
            else:
                # Add line to current section
                current_section["content"].append(line)

        # Add final section
        if current_section["content"] or current_section["header"]:
            current_section["content"] = "\n".join(current_section["content"])
            sections.append(current_section)

        return sections

    def _split_large_section(
        self,
        text: str,
        document_id: str,
        section_idx: int,
        hierarchy: List[str]
    ) -> List[Chunk]:
        """Split a large section into smaller chunks by paragraphs."""
        # Split by paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_text = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = len(self.tokenizer.encode(para))

            # Check if adding this paragraph would exceed target size
            if current_tokens + para_tokens > self.config.target_chunk_size and current_text:
                # Create chunk from accumulated paragraphs
                chunk_text = "\n\n".join(current_text)
                metadata = ChunkMetadata(
                    chunk_id=f"{document_id}_chunk_{section_idx}_{len(chunks)}",
                    source_document_id=document_id,
                    position=section_idx,
                    start_char=0,
                    end_char=len(chunk_text),
                    section_hierarchy=hierarchy,
                    chunking_method=ChunkingMethod.STRUCTURAL,
                    token_count=current_tokens
                )
                chunks.append(Chunk(text=chunk_text, metadata=metadata))

                current_text = [para]
                current_tokens = para_tokens
            else:
                current_text.append(para)
                current_tokens += para_tokens

        # Add remaining paragraphs
        if current_text:
            chunk_text = "\n\n".join(current_text)
            metadata = ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{section_idx}_{len(chunks)}",
                source_document_id=document_id,
                position=section_idx,
                start_char=0,
                end_char=len(chunk_text),
                section_hierarchy=hierarchy,
                chunking_method=ChunkingMethod.STRUCTURAL,
                token_count=current_tokens
            )
            chunks.append(Chunk(text=chunk_text, metadata=metadata))

        return chunks

    def _chunk_plain_text(self, document: str, document_id: str) -> List[Chunk]:
        """Chunk plain text document by paragraphs."""
        paragraphs = [p.strip() for p in document.split("\n\n") if p.strip()]

        chunks = []
        current_text = []
        current_tokens = 0

        for i, para in enumerate(paragraphs):
            para_tokens = len(self.tokenizer.encode(para))

            if current_tokens + para_tokens > self.config.target_chunk_size and current_text:
                chunk_text = "\n\n".join(current_text)
                metadata = ChunkMetadata(
                    chunk_id=f"{document_id}_chunk_{len(chunks)}",
                    source_document_id=document_id,
                    position=len(chunks),
                    start_char=0,
                    end_char=len(chunk_text),
                    section_hierarchy=[],
                    chunking_method=ChunkingMethod.STRUCTURAL,
                    token_count=current_tokens
                )
                chunks.append(Chunk(text=chunk_text, metadata=metadata))

                current_text = [para]
                current_tokens = para_tokens
            else:
                current_text.append(para)
                current_tokens += para_tokens

        # Add remaining text
        if current_text:
            chunk_text = "\n\n".join(current_text)
            metadata = ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{len(chunks)}",
                source_document_id=document_id,
                position=len(chunks),
                start_char=0,
                end_char=len(chunk_text),
                section_hierarchy=[],
                chunking_method=ChunkingMethod.STRUCTURAL,
                token_count=current_tokens
            )
            chunks.append(Chunk(text=chunk_text, metadata=metadata))

        return chunks
```

---

## Unit Tests

### Test File Location
`tests/unit/strategies/chunking/test_semantic_chunker.py`
`tests/unit/strategies/chunking/test_structural_chunker.py`
`tests/unit/strategies/chunking/test_hybrid_chunker.py`

### Test Cases

#### TC4.1.1: Semantic Chunker Tests
```python
import pytest
from unittest.mock import Mock, MagicMock
from rag_factory.strategies.chunking.semantic_chunker import SemanticChunker
from rag_factory.strategies.chunking.base import ChunkingConfig, ChunkingMethod

@pytest.fixture
def mock_embedding_service():
    service = Mock()
    # Mock embedding result
    service.embed.return_value = Mock(
        embeddings=[[0.1] * 10 for _ in range(5)]
    )
    return service

@pytest.fixture
def chunking_config():
    return ChunkingConfig(
        method=ChunkingMethod.SEMANTIC,
        target_chunk_size=512,
        min_chunk_size=128,
        max_chunk_size=1024,
        similarity_threshold=0.7
    )

def test_semantic_chunker_initialization(chunking_config, mock_embedding_service):
    """Test semantic chunker initializes correctly."""
    chunker = SemanticChunker(chunking_config, mock_embedding_service)
    assert chunker.config == chunking_config
    assert chunker.embedding_service == mock_embedding_service

def test_chunk_document_basic(chunking_config, mock_embedding_service):
    """Test basic document chunking."""
    chunker = SemanticChunker(chunking_config, mock_embedding_service)

    document = """
    This is the first sentence. This is the second sentence.
    This is the third sentence. This is the fourth sentence.
    """

    chunks = chunker.chunk_document(document, "doc_1")

    assert len(chunks) > 0
    assert all(chunk.metadata.source_document_id == "doc_1" for chunk in chunks)
    assert all(chunk.metadata.chunking_method == ChunkingMethod.SEMANTIC for chunk in chunks)

def test_split_into_sentences(chunking_config, mock_embedding_service):
    """Test sentence splitting."""
    chunker = SemanticChunker(chunking_config, mock_embedding_service)

    text = "First sentence. Second sentence! Third sentence?"
    sentences = chunker._split_into_sentences(text)

    assert len(sentences) == 3
    assert sentences[0] == "First sentence."
    assert sentences[1] == "Second sentence!"
    assert sentences[2] == "Third sentence?"

def test_cosine_similarity(chunking_config, mock_embedding_service):
    """Test cosine similarity calculation."""
    chunker = SemanticChunker(chunking_config, mock_embedding_service)

    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]

    similarity = chunker._cosine_similarity(vec1, vec2)
    assert similarity == pytest.approx(1.0)

    vec3 = [0.0, 1.0, 0.0]
    similarity = chunker._cosine_similarity(vec1, vec3)
    assert similarity == pytest.approx(0.0)

def test_detect_boundaries_high_similarity(chunking_config, mock_embedding_service):
    """Test boundary detection with high similarity (no boundaries)."""
    chunker = SemanticChunker(chunking_config, mock_embedding_service)

    # All embeddings identical (high similarity)
    embeddings = [[0.5, 0.5] for _ in range(5)]
    boundaries = chunker._detect_boundaries(embeddings)

    # Should only have start and end boundaries
    assert 0 in boundaries
    assert len(embeddings) in boundaries

def test_detect_boundaries_low_similarity(chunking_config, mock_embedding_service):
    """Test boundary detection with low similarity (creates boundaries)."""
    config = ChunkingConfig(similarity_threshold=0.9)
    chunker = SemanticChunker(config, mock_embedding_service)

    # Embeddings with low similarity
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],  # Low similarity -> boundary
        [0.0, 0.9],
        [0.1, 0.9]
    ]

    boundaries = chunker._detect_boundaries(embeddings)

    # Should detect boundary where similarity drops
    assert len(boundaries) > 2  # More than just start and end

def test_chunk_size_adjustment_merge_small(chunking_config, mock_embedding_service):
    """Test merging chunks that are too small."""
    from rag_factory.strategies.chunking.base import Chunk, ChunkMetadata

    chunker = SemanticChunker(chunking_config, mock_embedding_service)

    # Create small chunks
    chunks = [
        Chunk(
            text="Small chunk 1",
            metadata=ChunkMetadata(
                chunk_id="c1",
                source_document_id="doc1",
                position=0,
                start_char=0,
                end_char=13,
                section_hierarchy=[],
                chunking_method=ChunkingMethod.SEMANTIC,
                token_count=50  # Below min_chunk_size (128)
            )
        ),
        Chunk(
            text="Small chunk 2",
            metadata=ChunkMetadata(
                chunk_id="c2",
                source_document_id="doc1",
                position=1,
                start_char=0,
                end_char=13,
                section_hierarchy=[],
                chunking_method=ChunkingMethod.SEMANTIC,
                token_count=60  # Below min_chunk_size (128)
            )
        )
    ]

    adjusted = chunker._adjust_chunk_sizes(chunks)

    # Should merge into one chunk
    assert len(adjusted) < len(chunks) or adjusted[0].metadata.token_count >= chunking_config.min_chunk_size

def test_coherence_score_computation(chunking_config, mock_embedding_service):
    """Test coherence score calculation."""
    # Mock embeddings for coherence calculation
    mock_embedding_service.embed.return_value = Mock(
        embeddings=[[0.9, 0.1], [0.85, 0.15], [0.8, 0.2]]
    )

    config = ChunkingConfig(compute_coherence_scores=True)
    chunker = SemanticChunker(config, mock_embedding_service)

    from rag_factory.strategies.chunking.base import Chunk, ChunkMetadata

    chunks = [
        Chunk(
            text="First sentence. Second sentence. Third sentence.",
            metadata=ChunkMetadata(
                chunk_id="c1",
                source_document_id="doc1",
                position=0,
                start_char=0,
                end_char=48,
                section_hierarchy=[],
                chunking_method=ChunkingMethod.SEMANTIC,
                token_count=10
            )
        )
    ]

    result = chunker._compute_coherence_scores(chunks)

    # Should have coherence score
    assert result[0].metadata.coherence_score is not None
    assert 0.0 <= result[0].metadata.coherence_score <= 1.0

def test_empty_document_handling(chunking_config, mock_embedding_service):
    """Test handling of empty documents."""
    chunker = SemanticChunker(chunking_config, mock_embedding_service)

    chunks = chunker.chunk_document("", "doc_1")
    assert chunks == []

def test_single_sentence_document(chunking_config, mock_embedding_service):
    """Test document with single sentence."""
    mock_embedding_service.embed.return_value = Mock(embeddings=[[0.5] * 10])

    chunker = SemanticChunker(chunking_config, mock_embedding_service)

    chunks = chunker.chunk_document("Single sentence.", "doc_1")

    assert len(chunks) >= 1
    assert chunks[0].text == "Single sentence."
```

#### TC4.1.2: Structural Chunker Tests
```python
import pytest
from rag_factory.strategies.chunking.structural_chunker import StructuralChunker
from rag_factory.strategies.chunking.base import ChunkingConfig, ChunkingMethod

@pytest.fixture
def chunking_config():
    return ChunkingConfig(
        method=ChunkingMethod.STRUCTURAL,
        target_chunk_size=512,
        respect_headers=True,
        respect_paragraphs=True
    )

def test_structural_chunker_initialization(chunking_config):
    """Test structural chunker initializes correctly."""
    chunker = StructuralChunker(chunking_config)
    assert chunker.config == chunking_config

def test_is_markdown_detection(chunking_config):
    """Test markdown detection."""
    chunker = StructuralChunker(chunking_config)

    markdown_text = "# Header\n\nSome content"
    assert chunker._is_markdown(markdown_text) == True

    plain_text = "Just plain text without headers"
    assert chunker._is_markdown(plain_text) == False

def test_chunk_markdown_with_headers(chunking_config):
    """Test chunking markdown document with headers."""
    chunker = StructuralChunker(chunking_config)

    markdown = """# Header 1

Content under header 1.

## Header 1.1

Content under header 1.1.

## Header 1.2

Content under header 1.2.

# Header 2

Content under header 2.
"""

    chunks = chunker.chunk_document(markdown, "doc_1")

    assert len(chunks) > 0
    assert all(chunk.metadata.chunking_method == ChunkingMethod.STRUCTURAL for chunk in chunks)

    # Check that hierarchy is preserved
    assert any(len(chunk.metadata.section_hierarchy) > 0 for chunk in chunks)

def test_split_by_headers_hierarchy(chunking_config):
    """Test header hierarchy preservation."""
    chunker = StructuralChunker(chunking_config)

    markdown = """# Level 1

Content 1

## Level 2

Content 2

### Level 3

Content 3

## Another Level 2

More content
"""

    sections = chunker._split_by_headers(markdown)

    # Verify hierarchy is tracked
    assert len(sections) > 0

    # Find the Level 3 section
    level_3_sections = [s for s in sections if s["level"] == 3]
    if level_3_sections:
        assert "Level 1" in level_3_sections[0]["hierarchy"]
        assert "Level 2" in level_3_sections[0]["hierarchy"]
        assert "Level 3" in level_3_sections[0]["hierarchy"]

def test_chunk_plain_text_by_paragraphs(chunking_config):
    """Test chunking plain text by paragraphs."""
    chunker = StructuralChunker(chunking_config)

    text = """First paragraph with some content.

Second paragraph with more content.

Third paragraph with even more content.

Fourth paragraph."""

    chunks = chunker.chunk_document(text, "doc_1")

    assert len(chunks) > 0
    assert all(chunk.metadata.chunking_method == ChunkingMethod.STRUCTURAL for chunk in chunks)

def test_large_section_splitting(chunking_config):
    """Test splitting large sections that exceed max size."""
    # Configure small chunk size for testing
    config = ChunkingConfig(
        method=ChunkingMethod.STRUCTURAL,
        target_chunk_size=50,  # Small size to force splitting
        max_chunk_size=100
    )
    chunker = StructuralChunker(config)

    # Create a large section
    large_section = "\n\n".join([f"Paragraph {i} with some content." for i in range(20)])

    chunks = chunker._split_large_section(
        large_section,
        "doc_1",
        0,
        ["Test Section"]
    )

    # Should split into multiple chunks
    assert len(chunks) > 1

    # All chunks should have same hierarchy
    assert all(chunk.metadata.section_hierarchy == ["Test Section"] for chunk in chunks)

def test_empty_markdown_document(chunking_config):
    """Test empty markdown document."""
    chunker = StructuralChunker(chunking_config)

    chunks = chunker.chunk_document("", "doc_1")

    # Should handle gracefully
    assert isinstance(chunks, list)

def test_markdown_with_code_blocks(chunking_config):
    """Test markdown with code blocks."""
    chunker = StructuralChunker(chunking_config)

    markdown = """# Code Example

Here is some code:

```python
def hello():
    print("Hello, world!")
```

More content after code.
"""

    chunks = chunker.chunk_document(markdown, "doc_1")

    assert len(chunks) > 0
    # Code block should be preserved in chunk
    assert any("```python" in chunk.text for chunk in chunks)

def test_multiple_documents_batch(chunking_config):
    """Test chunking multiple documents in batch."""
    chunker = StructuralChunker(chunking_config)

    documents = [
        {"text": "# Doc 1\n\nContent 1", "id": "doc_1"},
        {"text": "# Doc 2\n\nContent 2", "id": "doc_2"},
        {"text": "Plain text document", "id": "doc_3"}
    ]

    results = chunker.chunk_documents(documents)

    assert len(results) == 3
    assert all(isinstance(chunks, list) for chunks in results)

    # Verify document IDs are preserved
    assert results[0][0].metadata.source_document_id == "doc_1"
    assert results[1][0].metadata.source_document_id == "doc_2"
    assert results[2][0].metadata.source_document_id == "doc_3"
```

---

## Integration Tests

### Test File Location
`tests/integration/strategies/test_chunking_integration.py`

### Test Scenarios

#### IS4.1.1: End-to-End Chunking Workflow
```python
import pytest
import os
from rag_factory.strategies.chunking.semantic_chunker import SemanticChunker
from rag_factory.strategies.chunking.structural_chunker import StructuralChunker
from rag_factory.strategies.chunking.base import ChunkingConfig, ChunkingMethod
from rag_factory.services.embedding import EmbeddingService, EmbeddingServiceConfig

@pytest.mark.integration
def test_semantic_chunking_with_real_embeddings():
    """Test semantic chunking with real embedding service."""
    # Setup embedding service
    embed_config = EmbeddingServiceConfig(
        provider="local",
        model="all-MiniLM-L6-v2"
    )
    embedding_service = EmbeddingService(embed_config)

    # Setup chunker
    chunk_config = ChunkingConfig(
        method=ChunkingMethod.SEMANTIC,
        target_chunk_size=256,
        min_chunk_size=64,
        similarity_threshold=0.7,
        compute_coherence_scores=True
    )
    chunker = SemanticChunker(chunk_config, embedding_service)

    # Test document with topic shifts
    document = """
    Machine learning is a subset of artificial intelligence.
    It focuses on building systems that learn from data.
    Neural networks are a key component of deep learning.

    Python is a popular programming language for data science.
    It has many libraries like NumPy, Pandas, and Scikit-learn.
    These tools make it easy to work with data.

    Climate change is one of the biggest challenges facing humanity.
    Rising temperatures are causing glaciers to melt.
    We need to reduce carbon emissions urgently.
    """

    chunks = chunker.chunk_document(document, "test_doc")

    # Assertions
    assert len(chunks) > 0
    assert all(chunk.metadata.chunking_method == ChunkingMethod.SEMANTIC for chunk in chunks)
    assert all(chunk.metadata.coherence_score is not None for chunk in chunks)

    # Should detect topic shifts (ML -> Python -> Climate)
    assert len(chunks) >= 2

    # Get stats
    stats = chunker.get_stats(chunks)
    print(f"\nChunking stats: {stats}")

    assert stats["total_chunks"] == len(chunks)
    assert stats["avg_coherence"] is not None

@pytest.mark.integration
def test_structural_chunking_markdown_document():
    """Test structural chunking with real markdown document."""
    chunk_config = ChunkingConfig(
        method=ChunkingMethod.STRUCTURAL,
        target_chunk_size=512,
        respect_headers=True
    )
    chunker = StructuralChunker(chunk_config)

    # Complex markdown document
    markdown = """# Introduction to RAG

Retrieval-Augmented Generation (RAG) is a technique that combines retrieval with generation.

## Key Components

### Vector Database

The vector database stores embeddings of document chunks.
It enables semantic search over large document collections.

### Embedding Model

Embeddings convert text into dense vectors.
These vectors capture semantic meaning.

### LLM

The language model generates responses based on retrieved context.

## Benefits

RAG provides several advantages:
- Reduces hallucinations
- Enables knowledge grounding
- Allows dynamic knowledge updates

## Challenges

Some challenges include:
- Chunk size optimization
- Relevance ranking
- Context window limitations

# Conclusion

RAG is a powerful technique for building knowledge-grounded applications.
"""

    chunks = chunker.chunk_document(markdown, "rag_doc")

    # Assertions
    assert len(chunks) > 0

    # Should preserve header hierarchy
    hierarchies = [chunk.metadata.section_hierarchy for chunk in chunks]
    assert any(len(h) > 1 for h in hierarchies)  # Nested sections

    # Check that headers are in chunks
    assert any("Introduction to RAG" in h for h in hierarchies if h)

    # Verify all chunks have metadata
    for chunk in chunks:
        assert chunk.metadata.token_count > 0
        assert chunk.metadata.source_document_id == "rag_doc"

@pytest.mark.integration
def test_chunking_real_document_from_file():
    """Test chunking a real document loaded from file."""
    # Create a test document
    test_file = "/tmp/test_document.md"
    with open(test_file, "w") as f:
        f.write("""# Sample Technical Document

## Overview

This is a technical document about software engineering.

## Section 1: Design Patterns

Design patterns are reusable solutions to common problems.

### Factory Pattern

The factory pattern provides an interface for creating objects.

### Observer Pattern

The observer pattern defines one-to-many dependencies.

## Section 2: Testing

Testing is crucial for software quality.

### Unit Testing

Unit tests verify individual components.

### Integration Testing

Integration tests verify component interactions.
""")

    # Test with structural chunker
    chunk_config = ChunkingConfig(
        method=ChunkingMethod.STRUCTURAL,
        target_chunk_size=256
    )
    chunker = StructuralChunker(chunk_config)

    with open(test_file, "r") as f:
        document = f.read()

    chunks = chunker.chunk_document(document, "tech_doc")

    assert len(chunks) > 0

    # Cleanup
    os.remove(test_file)

@pytest.mark.integration
def test_chunk_quality_metrics():
    """Test chunk quality metrics calculation."""
    embed_config = EmbeddingServiceConfig(
        provider="local",
        model="all-MiniLM-L6-v2"
    )
    embedding_service = EmbeddingService(embed_config)

    chunk_config = ChunkingConfig(
        method=ChunkingMethod.SEMANTIC,
        compute_coherence_scores=True
    )
    chunker = SemanticChunker(chunk_config, embedding_service)

    document = """
    Artificial intelligence is transforming the world.
    Machine learning algorithms can learn from data.
    Deep learning uses neural networks with many layers.
    These technologies are being applied in many domains.
    """

    chunks = chunker.chunk_document(document, "ai_doc")

    # All chunks should have coherence scores
    assert all(chunk.metadata.coherence_score is not None for chunk in chunks)

    # Coherence scores should be between 0 and 1
    assert all(0.0 <= chunk.metadata.coherence_score <= 1.0 for chunk in chunks)

    # Get stats
    stats = chunker.get_stats(chunks)
    assert "avg_coherence" in stats
    assert stats["avg_coherence"] is not None

@pytest.mark.integration
def test_compare_chunking_strategies():
    """Compare different chunking strategies on same document."""
    document = """# Machine Learning Guide

Machine learning is a field of AI that focuses on building systems that learn from data.

## Supervised Learning

In supervised learning, models are trained on labeled data.
The model learns to map inputs to outputs based on example pairs.

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data.
Common techniques include clustering and dimensionality reduction.

## Reinforcement Learning

Reinforcement learning trains agents through rewards and penalties.
The agent learns to maximize cumulative reward over time.
"""

    # Test structural chunking
    structural_config = ChunkingConfig(method=ChunkingMethod.STRUCTURAL)
    structural_chunker = StructuralChunker(structural_config)
    structural_chunks = structural_chunker.chunk_document(document, "ml_doc")

    # Test semantic chunking
    embed_config = EmbeddingServiceConfig(provider="local", model="all-MiniLM-L6-v2")
    embedding_service = EmbeddingService(embed_config)
    semantic_config = ChunkingConfig(method=ChunkingMethod.SEMANTIC)
    semantic_chunker = SemanticChunker(semantic_config, embedding_service)
    semantic_chunks = semantic_chunker.chunk_document(document, "ml_doc")

    # Both should produce chunks
    assert len(structural_chunks) > 0
    assert len(semantic_chunks) > 0

    # Get stats for comparison
    structural_stats = structural_chunker.get_stats(structural_chunks)
    semantic_stats = semantic_chunker.get_stats(semantic_chunks)

    print(f"\nStructural chunking: {structural_stats}")
    print(f"Semantic chunking: {semantic_stats}")

    # Verify different strategies produce different results
    # (number of chunks may differ)
    assert structural_stats["total_chunks"] >= 1
    assert semantic_stats["total_chunks"] >= 1

@pytest.mark.integration
def test_large_document_performance():
    """Test performance on large document."""
    import time

    # Generate large document
    large_doc = "\n\n".join([
        f"# Section {i}\n\nThis is content for section {i}. " * 10
        for i in range(50)
    ])

    chunk_config = ChunkingConfig(
        method=ChunkingMethod.STRUCTURAL,
        target_chunk_size=512
    )
    chunker = StructuralChunker(chunk_config)

    start = time.time()
    chunks = chunker.chunk_document(large_doc, "large_doc")
    duration = time.time() - start

    # Should process quickly
    assert duration < 5.0  # Less than 5 seconds

    # Should produce many chunks
    assert len(chunks) > 10

    # Calculate throughput
    stats = chunker.get_stats(chunks)
    print(f"\nProcessed {stats['total_chunks']} chunks in {duration:.2f}s")
```

---

## Performance Benchmarks

```python
# tests/benchmarks/test_chunking_performance.py

import pytest
import time
from rag_factory.strategies.chunking.semantic_chunker import SemanticChunker
from rag_factory.strategies.chunking.structural_chunker import StructuralChunker
from rag_factory.strategies.chunking.base import ChunkingConfig, ChunkingMethod
from rag_factory.services.embedding import EmbeddingService, EmbeddingServiceConfig

@pytest.mark.benchmark
def test_structural_chunking_performance():
    """Benchmark structural chunking speed."""
    config = ChunkingConfig(method=ChunkingMethod.STRUCTURAL)
    chunker = StructuralChunker(config)

    # Generate test document
    document = "\n\n".join([f"Paragraph {i} with content." for i in range(1000)])

    start = time.time()
    chunks = chunker.chunk_document(document, "perf_test")
    duration = time.time() - start

    chunks_per_second = len(chunks) / duration

    print(f"\nStructural chunking: {len(chunks)} chunks in {duration:.3f}s ({chunks_per_second:.0f} chunks/s)")

    # Should meet performance target
    assert chunks_per_second > 100, f"Performance: {chunks_per_second:.0f} chunks/s (expected >100)"

@pytest.mark.benchmark
def test_semantic_chunking_performance():
    """Benchmark semantic chunking speed."""
    embed_config = EmbeddingServiceConfig(
        provider="local",
        model="all-MiniLM-L6-v2"
    )
    embedding_service = EmbeddingService(embed_config)

    config = ChunkingConfig(
        method=ChunkingMethod.SEMANTIC,
        compute_coherence_scores=False  # Disable for performance test
    )
    chunker = SemanticChunker(config, embedding_service)

    document = ". ".join([f"Sentence {i}" for i in range(100)])

    start = time.time()
    chunks = chunker.chunk_document(document, "perf_test")
    duration = time.time() - start

    print(f"\nSemantic chunking: {len(chunks)} chunks in {duration:.3f}s")

    # Semantic chunking is slower due to embeddings
    assert duration < 10.0, f"Took {duration:.2f}s (expected <10s)"

@pytest.mark.benchmark
def test_batch_document_processing():
    """Benchmark batch processing of multiple documents."""
    config = ChunkingConfig(method=ChunkingMethod.STRUCTURAL)
    chunker = StructuralChunker(config)

    # Create 100 documents
    documents = [
        {"text": f"# Document {i}\n\nContent for document {i}.", "id": f"doc_{i}"}
        for i in range(100)
    ]

    start = time.time()
    results = chunker.chunk_documents(documents)
    duration = time.time() - start

    total_chunks = sum(len(chunks) for chunks in results)
    docs_per_second = len(documents) / duration

    print(f"\nBatch processing: {len(documents)} docs, {total_chunks} chunks in {duration:.3f}s ({docs_per_second:.0f} docs/s)")

    assert docs_per_second > 10, f"Performance: {docs_per_second:.0f} docs/s (expected >10)"
```

---

## Definition of Done

- [ ] Base chunker interface defined
- [ ] Semantic chunking strategy implemented
- [ ] Structural chunking strategy implemented
- [ ] Hybrid chunking strategy implemented (combines semantic + structural)
- [ ] Dockling integration completed
- [ ] Fixed-size baseline chunker implemented
- [ ] Metadata tracking working correctly
- [ ] Chunk quality metrics implemented
- [ ] All unit tests pass (>90% coverage)
- [ ] All integration tests pass
- [ ] Performance benchmarks meet requirements (>100 chunks/second)
- [ ] Configuration system working
- [ ] Documentation complete with examples
- [ ] Code reviewed
- [ ] No linting errors

---

## Setup Instructions

### Installation

```bash
# Install dependencies
pip install dockling tiktoken beautifulsoup4 pypdf2 python-magic sentence-transformers

# Verify dockling installation
python -c "import dockling; print('Dockling installed successfully')"
```

### Configuration

```yaml
# config.yaml
chunking:
  method: "hybrid"  # semantic, structural, hybrid, fixed_size, dockling

  min_chunk_size: 128
  max_chunk_size: 1024
  target_chunk_size: 512
  chunk_overlap: 50

  # Semantic settings
  similarity_threshold: 0.7
  use_embeddings: true

  # Structural settings
  respect_headers: true
  respect_paragraphs: true
  keep_code_blocks_intact: true
  keep_tables_intact: true

  # Quality metrics
  compute_coherence_scores: true
```

### Usage Example

```python
from rag_factory.strategies.chunking import SemanticChunker, StructuralChunker
from rag_factory.strategies.chunking.base import ChunkingConfig, ChunkingMethod
from rag_factory.services.embedding import EmbeddingService, EmbeddingServiceConfig

# Setup embedding service for semantic chunking
embed_config = EmbeddingServiceConfig(
    provider="local",
    model="all-MiniLM-L6-v2"
)
embedding_service = EmbeddingService(embed_config)

# Configure semantic chunker
chunk_config = ChunkingConfig(
    method=ChunkingMethod.SEMANTIC,
    target_chunk_size=512,
    similarity_threshold=0.7,
    compute_coherence_scores=True
)

chunker = SemanticChunker(chunk_config, embedding_service)

# Chunk a document
document = """
# Introduction

This is a sample document...

## Section 1

Content for section 1...
"""

chunks = chunker.chunk_document(document, "sample_doc")

# Print results
for chunk in chunks:
    print(f"Chunk {chunk.metadata.chunk_id}:")
    print(f"  Tokens: {chunk.metadata.token_count}")
    print(f"  Coherence: {chunk.metadata.coherence_score:.3f}")
    print(f"  Hierarchy: {chunk.metadata.section_hierarchy}")
    print(f"  Preview: {chunk.text[:100]}...")
    print()

# Get statistics
stats = chunker.get_stats(chunks)
print(f"Total chunks: {stats['total_chunks']}")
print(f"Average size: {stats['avg_chunk_size']:.0f} tokens")
print(f"Average coherence: {stats['avg_coherence']:.3f}")
```

---

## Notes for Developers

1. **Start with Structural Chunking**: It's simpler and faster. Use semantic chunking when quality is critical.

2. **Dockling Integration**: Dockling is powerful for PDFs and complex documents. Always enable fallback to basic chunking.

3. **Chunk Size Tuning**: The optimal chunk size depends on your use case:
   - Smaller chunks (256-512): Better precision, more chunks to process
   - Larger chunks (512-1024): More context, fewer chunks

4. **Similarity Threshold**: For semantic chunking, tune the threshold:
   - Higher threshold (0.8-0.9): Fewer, larger chunks
   - Lower threshold (0.5-0.7): More, smaller chunks

5. **Performance**: Structural chunking is fast. Semantic chunking requires embeddings and is slower but produces higher quality chunks.

6. **Metadata**: Always preserve metadata. It's crucial for tracking and debugging.

7. **Testing**: Test with various document types (markdown, PDF, plain text) to ensure robustness.

8. **Coherence Scores**: Enable coherence scoring in development to validate chunk quality. Disable in production if performance is critical.

9. **Hybrid Approach**: Combine structural and semantic chunking for best results - use structure as first pass, then semantic refinement.

10. **Edge Cases**: Handle empty documents, single-sentence documents, and malformed content gracefully with try-catch blocks.
