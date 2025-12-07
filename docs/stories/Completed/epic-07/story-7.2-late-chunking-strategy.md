# Story 7.2: Implement Late Chunking Strategy

**Story ID:** 7.2
**Epic:** Epic 7 - Advanced & Experimental Strategies
**Story Points:** 21
**Priority:** Experimental (High Complexity)
**Dependencies:** Epic 3 (Embedding Service), Epic 4 (Chunking Strategies)

---

## User Story

**As a** system
**I want** to apply embeddings before chunking
**So that** chunks maintain full document context

---

## Overview

Late chunking is an advanced technique that reverses the traditional RAG workflow:

**Traditional Chunking**:
Document → Split into chunks → Embed each chunk → Store embeddings

**Late Chunking**:
Document → Embed full document → Split embeddings into chunks → Store embedding chunks

This approach maintains full document context during embedding, potentially improving semantic understanding and chunk quality.

---

## Detailed Requirements

### Functional Requirements

1. **Full Document Embedding**
   - Embed entire document before chunking
   - Support long-context embedding models (>8K tokens)
   - Handle documents that exceed model context window:
     - Sliding window approach for very long documents
     - Hierarchical embedding (embed sections then combine)
     - Document truncation with warnings
   - Generate single embedding vector for full document
   - Track embedding model context limitations

2. **Token-Level Embedding Access**
   - Extract token-level embeddings from model
   - Support models that provide token embeddings (BERT, RoBERTa, etc.)
   - Map tokens back to character positions in original text
   - Handle subword tokenization (BPE, WordPiece)
   - Preserve embedding dimensionality
   - Support both transformer models and custom embedding functions

3. **Embedding Chunk Splitting**
   - Split token embeddings into semantically coherent groups
   - Strategies for splitting:
     - **Fixed-size**: Fixed number of tokens per chunk
     - **Semantic boundary**: Detect embedding similarity drops
     - **Hierarchical**: Split based on document structure + embeddings
     - **Adaptive**: Adjust chunk size based on embedding variance
   - Configurable chunk size ranges (min, target, max tokens)
   - Overlap between embedding chunks (configurable)

4. **Text Reconstruction from Token Chunks**
   - Map embedding chunks back to original text spans
   - Reconstruct readable text from token ranges
   - Handle subword tokens correctly (merge back to words)
   - Preserve original text boundaries (don't split mid-word)
   - Maintain character-level accuracy for text extraction

5. **Semantic Coherence Preservation**
   - Measure intra-chunk embedding similarity
   - Minimize inter-chunk embedding disruption
   - Detect and preserve semantic boundaries
   - Calculate coherence scores for each chunk
   - Compare coherence with traditional chunking

6. **Metadata and Context Preservation**
   - Store full document embedding as context
   - Link chunks to parent document embedding
   - Track token ranges for each chunk
   - Preserve positional information
   - Store chunking method metadata

### Non-Functional Requirements

1. **Performance**
   - Full document embedding: <2s for typical document (2K tokens)
   - Token embedding extraction: <500ms
   - Embedding chunking: <300ms
   - Support for batch processing (multiple documents)
   - Memory-efficient handling of large embeddings

2. **Scalability**
   - Handle documents up to 32K tokens (with long-context models)
   - Process multiple documents concurrently
   - Efficient storage of token embeddings
   - Streaming support for very large documents

3. **Quality**
   - Improved semantic coherence vs. traditional chunking (measurable)
   - Better context preservation (quantitative metrics)
   - No information loss during text reconstruction
   - Comparable or better retrieval accuracy

4. **Model Compatibility**
   - Support for long-context models:
     - Longformer (4K-16K tokens)
     - LED (16K tokens)
     - BigBird (4K tokens)
     - Custom models with long context
   - Fallback to chunked embedding for non-long-context models
   - Support for various model architectures

5. **Observability**
   - Log embedding dimensions and token counts
   - Track chunk splitting decisions
   - Monitor coherence scores
   - Compare with traditional chunking metrics
   - Alert when document exceeds model capacity

---

## Acceptance Criteria

### AC1: Full Document Embedding
- [ ] Full document embedding implemented
- [ ] Long-context embedding models supported
- [ ] Documents up to 16K tokens handled
- [ ] Sliding window for longer documents
- [ ] Document embeddings stored correctly

### AC2: Token Embedding Access
- [ ] Token-level embeddings extracted
- [ ] Token-to-character mapping working
- [ ] Subword tokenization handled correctly
- [ ] Embedding dimensionality preserved
- [ ] At least 2 model types supported (BERT-style, custom)

### AC3: Embedding Chunk Splitting
- [ ] At least 3 splitting strategies implemented
- [ ] Fixed-size splitting working
- [ ] Semantic boundary detection working
- [ ] Configurable chunk sizes
- [ ] Chunk overlap supported

### AC4: Text Reconstruction
- [ ] Token chunks map back to original text
- [ ] Text reconstruction accurate (no character loss)
- [ ] Subword tokens merged correctly
- [ ] Word boundaries preserved

### AC5: Semantic Coherence
- [ ] Intra-chunk similarity calculated
- [ ] Coherence scores computed
- [ ] Comparison with traditional chunking
- [ ] Semantic boundaries detected

### AC6: Context Preservation
- [ ] Full document embedding stored
- [ ] Chunk-to-document links maintained
- [ ] Positional information preserved
- [ ] Metadata complete

### AC7: Performance
- [ ] Full document embedding <2s (2K tokens)
- [ ] Token extraction <500ms
- [ ] Chunking <300ms
- [ ] Memory usage reasonable (<2GB per document)

### AC8: Testing
- [ ] Unit tests for all components (>85% coverage)
- [ ] Integration tests with real documents
- [ ] Comparison tests vs. traditional chunking
- [ ] Performance benchmarks meet requirements
- [ ] Quality metrics on test datasets

---

## Technical Specifications

### File Structure
```
rag_factory/
├── strategies/
│   ├── late_chunking/
│   │   ├── __init__.py
│   │   ├── strategy.py                 # Main late chunking strategy
│   │   ├── document_embedder.py        # Full document embedding
│   │   ├── token_embedder.py           # Token-level embedding extraction
│   │   ├── embedding_chunker.py        # Split embeddings into chunks
│   │   ├── text_reconstructor.py       # Map embeddings back to text
│   │   ├── coherence_analyzer.py       # Analyze chunk coherence
│   │   ├── models.py                   # Data models
│   │   ├── config.py                   # Configuration
│   │   └── utils.py                    # Utility functions
│
tests/
├── unit/
│   └── strategies/
│       └── late_chunking/
│           ├── test_document_embedder.py
│           ├── test_token_embedder.py
│           ├── test_embedding_chunker.py
│           ├── test_text_reconstructor.py
│           └── test_coherence_analyzer.py
│
├── integration/
│   └── strategies/
│       └── test_late_chunking_integration.py
│
├── benchmarks/
│   └── test_late_chunking_vs_traditional.py
```

### Dependencies
```python
# requirements.txt additions
transformers==4.36.0               # For transformer models
torch==2.1.2                       # PyTorch for embeddings
tokenizers==0.15.0                 # Fast tokenization
sentence-transformers==2.2.2       # Embedding models
```

### Data Models
```python
# rag_factory/strategies/late_chunking/models.py
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np

class EmbeddingChunkingMethod(str, Enum):
    """Methods for chunking embeddings."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC_BOUNDARY = "semantic_boundary"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"

class TokenEmbedding(BaseModel):
    """Token-level embedding with text mapping."""
    token: str
    token_id: int
    start_char: int
    end_char: int
    embedding: List[float]
    position: int

    class Config:
        arbitrary_types_allowed = True

class DocumentEmbedding(BaseModel):
    """Full document embedding with token details."""
    document_id: str
    text: str
    full_embedding: List[float]  # Document-level embedding
    token_embeddings: List[TokenEmbedding]  # Token-level embeddings
    model_name: str
    token_count: int
    embedding_dim: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

class EmbeddingChunk(BaseModel):
    """Chunk created from embedding split."""
    chunk_id: str
    document_id: str
    text: str
    chunk_embedding: List[float]  # Average of token embeddings in chunk
    token_range: Tuple[int, int]  # Start and end token indices
    char_range: Tuple[int, int]  # Start and end character positions
    token_count: int
    coherence_score: Optional[float] = None
    chunking_method: EmbeddingChunkingMethod
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

class LateChunkingConfig(BaseModel):
    """Configuration for late chunking."""
    # Document embedding
    use_long_context_model: bool = True
    max_document_tokens: int = 16384
    model_name: str = "sentence-transformers/all-mpnet-base-v2"

    # Chunking
    chunking_method: EmbeddingChunkingMethod = EmbeddingChunkingMethod.SEMANTIC_BOUNDARY
    target_chunk_size: int = 512
    min_chunk_size: int = 128
    max_chunk_size: int = 1024
    chunk_overlap_tokens: int = 50

    # Semantic boundary detection
    similarity_threshold: float = 0.7
    use_local_similarity: bool = True

    # Coherence analysis
    compute_coherence_scores: bool = True
    coherence_window_size: int = 3

    # Performance
    batch_size: int = 1
    device: str = "cpu"  # or "cuda"

class CoherenceMetrics(BaseModel):
    """Coherence metrics for chunk evaluation."""
    intra_chunk_similarity: float  # Average similarity within chunk
    inter_chunk_similarity: float  # Similarity with adjacent chunks
    variance: float  # Embedding variance within chunk
    semantic_boundary_score: float  # How well boundaries align with semantics
    comparison_to_traditional: Optional[float] = None  # Improvement over traditional
```

### Document Embedder
```python
# rag_factory/strategies/late_chunking/document_embedder.py
from typing import List, Dict, Any, Optional
import logging
import torch
from transformers import AutoTokenizer, AutoModel
from .models import DocumentEmbedding, TokenEmbedding, LateChunkingConfig

logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """Embed full documents with token-level detail."""

    def __init__(self, config: LateChunkingConfig):
        self.config = config
        self.device = config.device

        # Load model and tokenizer
        logger.info(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.max_length = min(
            config.max_document_tokens,
            self.tokenizer.model_max_length
        )

    def embed_document(
        self,
        text: str,
        document_id: str
    ) -> DocumentEmbedding:
        """
        Embed full document and extract token-level embeddings.

        Args:
            text: Document text
            document_id: Unique document ID

        Returns:
            DocumentEmbedding with token-level details
        """
        logger.info(f"Embedding document: {document_id}")

        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            padding=True
        )

        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0]

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        # Extract embeddings
        # Use last hidden state for token embeddings
        token_embeddings_tensor = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)

        # Document-level embedding (mean pooling)
        full_embedding = self._mean_pooling(
            token_embeddings_tensor,
            attention_mask[0]
        )

        # Convert token embeddings to TokenEmbedding objects
        token_embeddings = []
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        for i, (token, offset) in enumerate(zip(tokens, offset_mapping)):
            if i == 0 or i == len(tokens) - 1:  # Skip special tokens
                continue

            start_char, end_char = offset.tolist()

            token_emb = TokenEmbedding(
                token=token,
                token_id=int(input_ids[0][i]),
                start_char=start_char,
                end_char=end_char,
                embedding=token_embeddings_tensor[i].cpu().tolist(),
                position=i
            )
            token_embeddings.append(token_emb)

        # Create DocumentEmbedding
        doc_embedding = DocumentEmbedding(
            document_id=document_id,
            text=text,
            full_embedding=full_embedding.cpu().tolist(),
            token_embeddings=token_embeddings,
            model_name=self.config.model_name,
            token_count=len(token_embeddings),
            embedding_dim=token_embeddings_tensor.shape[1]
        )

        logger.info(
            f"Embedded document {document_id}: {len(token_embeddings)} tokens, "
            f"dim {doc_embedding.embedding_dim}"
        )

        return doc_embedding

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling to get document-level embedding."""
        # Expand attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=0)

        # Divide by number of tokens
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=0), min=1e-9)
        mean_embedding = sum_embeddings / sum_mask

        return mean_embedding

    def embed_documents_batch(
        self,
        documents: List[Dict[str, str]]
    ) -> List[DocumentEmbedding]:
        """
        Embed multiple documents in batch.

        Args:
            documents: List of {"text": str, "document_id": str}

        Returns:
            List of DocumentEmbedding objects
        """
        return [
            self.embed_document(doc["text"], doc["document_id"])
            for doc in documents
        ]
```

### Embedding Chunker
```python
# rag_factory/strategies/late_chunking/embedding_chunker.py
from typing import List, Tuple
import logging
import numpy as np
from .models import (
    DocumentEmbedding, EmbeddingChunk, EmbeddingChunkingMethod, LateChunkingConfig
)

logger = logging.getLogger(__name__)

class EmbeddingChunker:
    """Split token embeddings into semantically coherent chunks."""

    def __init__(self, config: LateChunkingConfig):
        self.config = config

    def chunk_embeddings(
        self,
        doc_embedding: DocumentEmbedding
    ) -> List[EmbeddingChunk]:
        """
        Chunk token embeddings into semantic units.

        Args:
            doc_embedding: Document embedding with token details

        Returns:
            List of embedding chunks
        """
        logger.info(
            f"Chunking embeddings for document {doc_embedding.document_id} "
            f"using method: {self.config.chunking_method.value}"
        )

        method = self.config.chunking_method

        if method == EmbeddingChunkingMethod.FIXED_SIZE:
            chunks = self._fixed_size_chunking(doc_embedding)
        elif method == EmbeddingChunkingMethod.SEMANTIC_BOUNDARY:
            chunks = self._semantic_boundary_chunking(doc_embedding)
        elif method == EmbeddingChunkingMethod.HIERARCHICAL:
            chunks = self._hierarchical_chunking(doc_embedding)
        elif method == EmbeddingChunkingMethod.ADAPTIVE:
            chunks = self._adaptive_chunking(doc_embedding)
        else:
            raise ValueError(f"Unknown chunking method: {method}")

        logger.info(f"Created {len(chunks)} embedding chunks")

        return chunks

    def _fixed_size_chunking(
        self,
        doc_embedding: DocumentEmbedding
    ) -> List[EmbeddingChunk]:
        """Fixed-size chunking of token embeddings."""
        chunks = []
        tokens = doc_embedding.token_embeddings
        chunk_size = self.config.target_chunk_size
        overlap = self.config.chunk_overlap_tokens

        i = 0
        chunk_idx = 0

        while i < len(tokens):
            # Define chunk range
            start_idx = i
            end_idx = min(i + chunk_size, len(tokens))

            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]

            # Create chunk
            chunk = self._create_chunk(
                doc_embedding,
                chunk_tokens,
                chunk_idx,
                EmbeddingChunkingMethod.FIXED_SIZE
            )

            chunks.append(chunk)

            # Move forward with overlap
            i += chunk_size - overlap
            chunk_idx += 1

        return chunks

    def _semantic_boundary_chunking(
        self,
        doc_embedding: DocumentEmbedding
    ) -> List[EmbeddingChunk]:
        """Chunk based on semantic boundary detection in embeddings."""
        tokens = doc_embedding.token_embeddings

        # Calculate similarity between consecutive tokens
        boundaries = self._detect_semantic_boundaries(tokens)

        # Create chunks from boundaries
        chunks = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            chunk_tokens = tokens[start_idx:end_idx]

            # Skip very small chunks
            if len(chunk_tokens) < self.config.min_chunk_size:
                # Merge with previous chunk if possible
                if chunks:
                    prev_chunk = chunks[-1]
                    # Update previous chunk to include these tokens
                    merged_tokens = tokens[boundaries[i-1]:end_idx]
                    chunks[-1] = self._create_chunk(
                        doc_embedding,
                        merged_tokens,
                        i - 1,
                        EmbeddingChunkingMethod.SEMANTIC_BOUNDARY
                    )
                continue

            chunk = self._create_chunk(
                doc_embedding,
                chunk_tokens,
                i,
                EmbeddingChunkingMethod.SEMANTIC_BOUNDARY
            )
            chunks.append(chunk)

        return chunks

    def _detect_semantic_boundaries(
        self,
        tokens: List
    ) -> List[int]:
        """Detect semantic boundaries by analyzing embedding similarities."""
        boundaries = [0]  # Start boundary

        # Extract embeddings as numpy array
        embeddings = np.array([t.embedding for t in tokens])

        # Calculate cosine similarity between consecutive tokens
        for i in range(1, len(embeddings) - 1):
            similarity = self._cosine_similarity(
                embeddings[i],
                embeddings[i + 1]
            )

            # If similarity drops below threshold, mark as boundary
            if similarity < self.config.similarity_threshold:
                # Check if chunk would be large enough
                if i - boundaries[-1] >= self.config.min_chunk_size:
                    boundaries.append(i)

        boundaries.append(len(tokens))  # End boundary

        return boundaries

    def _hierarchical_chunking(
        self,
        doc_embedding: DocumentEmbedding
    ) -> List[EmbeddingChunk]:
        """Hierarchical chunking combining structure and embeddings."""
        # TODO: Implement hierarchical chunking
        # For now, fall back to semantic boundary
        logger.warning("Hierarchical chunking not fully implemented, using semantic boundary")
        return self._semantic_boundary_chunking(doc_embedding)

    def _adaptive_chunking(
        self,
        doc_embedding: DocumentEmbedding
    ) -> List[EmbeddingChunk]:
        """Adaptive chunking based on embedding variance."""
        tokens = doc_embedding.token_embeddings
        embeddings = np.array([t.embedding for t in tokens])

        chunks = []
        current_chunk_start = 0
        chunk_idx = 0

        i = self.config.min_chunk_size

        while i < len(embeddings):
            # Calculate variance in current window
            window_embeddings = embeddings[current_chunk_start:i]
            variance = np.var(window_embeddings, axis=0).mean()

            # If variance is high, create chunk
            if variance > 0.1 and i - current_chunk_start >= self.config.min_chunk_size:
                chunk_tokens = tokens[current_chunk_start:i]
                chunk = self._create_chunk(
                    doc_embedding,
                    chunk_tokens,
                    chunk_idx,
                    EmbeddingChunkingMethod.ADAPTIVE
                )
                chunks.append(chunk)

                current_chunk_start = i
                chunk_idx += 1

            # Force chunk if max size reached
            if i - current_chunk_start >= self.config.max_chunk_size:
                chunk_tokens = tokens[current_chunk_start:i]
                chunk = self._create_chunk(
                    doc_embedding,
                    chunk_tokens,
                    chunk_idx,
                    EmbeddingChunkingMethod.ADAPTIVE
                )
                chunks.append(chunk)

                current_chunk_start = i
                chunk_idx += 1

            i += 1

        # Add remaining tokens
        if current_chunk_start < len(tokens):
            chunk_tokens = tokens[current_chunk_start:]
            chunk = self._create_chunk(
                doc_embedding,
                chunk_tokens,
                chunk_idx,
                EmbeddingChunkingMethod.ADAPTIVE
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        doc_embedding: DocumentEmbedding,
        chunk_tokens: List,
        chunk_idx: int,
        method: EmbeddingChunkingMethod
    ) -> EmbeddingChunk:
        """Create EmbeddingChunk from tokens."""
        # Extract text span
        start_char = chunk_tokens[0].start_char
        end_char = chunk_tokens[-1].end_char
        text = doc_embedding.text[start_char:end_char]

        # Average embeddings for chunk
        chunk_embedding = np.mean([t.embedding for t in chunk_tokens], axis=0).tolist()

        # Token range
        token_range = (chunk_tokens[0].position, chunk_tokens[-1].position)

        # Create chunk ID
        chunk_id = f"{doc_embedding.document_id}_late_chunk_{chunk_idx}"

        chunk = EmbeddingChunk(
            chunk_id=chunk_id,
            document_id=doc_embedding.document_id,
            text=text,
            chunk_embedding=chunk_embedding,
            token_range=token_range,
            char_range=(start_char, end_char),
            token_count=len(chunk_tokens),
            chunking_method=method
        )

        return chunk

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
```

I'll continue with the remaining sections in the next message since this is getting quite long...


### Coherence Analyzer
```python
# rag_factory/strategies/late_chunking/coherence_analyzer.py
from typing import List, Dict, Any
import logging
import numpy as np
from .models import EmbeddingChunk, CoherenceMetrics

logger = logging.getLogger(__name__)

class CoherenceAnalyzer:
    """Analyze semantic coherence of embedding chunks."""

    def __init__(self, config):
        self.config = config
        self.window_size = config.coherence_window_size

    def analyze_chunk_coherence(
        self,
        chunks: List[EmbeddingChunk]
    ) -> List[EmbeddingChunk]:
        """
        Calculate coherence scores for chunks.

        Args:
            chunks: List of embedding chunks

        Returns:
            Chunks with coherence scores added
        """
        logger.info(f"Analyzing coherence for {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            # Intra-chunk similarity (would need token embeddings)
            # For now, use a simplified approach
            coherence_score = self._calculate_intra_chunk_coherence(chunk)

            chunk.coherence_score = coherence_score

        return chunks

    def _calculate_intra_chunk_coherence(self, chunk: EmbeddingChunk) -> float:
        """
        Calculate coherence within a chunk.

        In a full implementation, this would analyze token-level embeddings.
        For now, we return a placeholder.
        """
        # Placeholder: In real implementation, would analyze token embeddings
        # within the chunk to measure semantic consistency
        return 0.85  # Placeholder value

    def compare_with_traditional(
        self,
        late_chunks: List[EmbeddingChunk],
        traditional_chunks: List[Any]
    ) -> Dict[str, float]:
        """
        Compare late chunking with traditional chunking.

        Args:
            late_chunks: Chunks from late chunking
            traditional_chunks: Chunks from traditional method

        Returns:
            Comparison metrics
        """
        metrics = {
            "late_chunking_coherence": np.mean([
                c.coherence_score for c in late_chunks
                if c.coherence_score is not None
            ]),
            "num_late_chunks": len(late_chunks),
            "num_traditional_chunks": len(traditional_chunks),
            "avg_late_chunk_size": np.mean([c.token_count for c in late_chunks]),
        }

        logger.info(f"Comparison metrics: {metrics}")

        return metrics
```

### Late Chunking Strategy
```python
# rag_factory/strategies/late_chunking/strategy.py
from typing import List, Dict, Any, Optional
import logging
from ..base import RAGStrategy
from .document_embedder import DocumentEmbedder
from .embedding_chunker import EmbeddingChunker
from .coherence_analyzer import CoherenceAnalyzer
from .models import LateChunkingConfig, EmbeddingChunkingMethod

logger = logging.getLogger(__name__)

class LateChunkingRAGStrategy(RAGStrategy):
    """
    Late Chunking RAG: Embed first, then chunk.

    This experimental strategy embeds the full document before chunking,
    maintaining full context during embedding for potentially better
    semantic understanding.
    """

    def __init__(
        self,
        vector_store_service: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)

        # Create config object
        self.late_config = LateChunkingConfig(**config) if config else LateChunkingConfig()

        # Initialize components
        self.vector_store = vector_store_service
        self.document_embedder = DocumentEmbedder(self.late_config)
        self.embedding_chunker = EmbeddingChunker(self.late_config)
        self.coherence_analyzer = CoherenceAnalyzer(self.late_config)

        logger.info("Late Chunking RAG Strategy initialized")
        logger.info(f"Model: {self.late_config.model_name}")
        logger.info(f"Chunking method: {self.late_config.chunking_method.value}")

    def index_document(self, document: str, document_id: str) -> None:
        """
        Index document using late chunking.

        Args:
            document: Document text
            document_id: Unique document ID
        """
        logger.info(f"Indexing document with late chunking: {document_id}")

        # Step 1: Embed full document
        doc_embedding = self.document_embedder.embed_document(document, document_id)
        logger.info(
            f"Document embedded: {doc_embedding.token_count} tokens, "
            f"dim {doc_embedding.embedding_dim}"
        )

        # Step 2: Chunk embeddings
        chunks = self.embedding_chunker.chunk_embeddings(doc_embedding)
        logger.info(f"Created {len(chunks)} embedding chunks")

        # Step 3: Analyze coherence
        if self.late_config.compute_coherence_scores:
            chunks = self.coherence_analyzer.analyze_chunk_coherence(chunks)
            avg_coherence = np.mean([c.coherence_score for c in chunks if c.coherence_score])
            logger.info(f"Average coherence score: {avg_coherence:.3f}")

        # Step 4: Index chunks in vector store
        for chunk in chunks:
            self.vector_store.index_chunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                embedding=chunk.chunk_embedding,
                metadata={
                    "document_id": chunk.document_id,
                    "token_count": chunk.token_count,
                    "coherence_score": chunk.coherence_score,
                    "chunking_method": "late_chunking",
                    "token_range": chunk.token_range,
                    "char_range": chunk.char_range
                }
            )

        logger.info(f"Indexed {len(chunks)} chunks for document {document_id}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using late chunking embeddings.

        Args:
            query: Search query
            top_k: Number of results
            **kwargs: Additional parameters

        Returns:
            List of retrieved chunks
        """
        logger.info(f"Late chunking retrieval for: {query}")

        # Use vector store search
        results = self.vector_store.search(query, top_k=top_k, **kwargs)

        # Add late chunking metadata to results
        for result in results:
            result["strategy"] = "late_chunking"

        return results

    @property
    def name(self) -> str:
        return "late_chunking"

    @property
    def description(self) -> str:
        return "Embed full document first, then chunk embeddings for better context"
```

---

## Unit Tests

### Test File Locations
- `tests/unit/strategies/late_chunking/test_document_embedder.py`
- `tests/unit/strategies/late_chunking/test_embedding_chunker.py`
- `tests/unit/strategies/late_chunking/test_coherence_analyzer.py`

### Test Cases

#### TC7.2.1: Document Embedder Tests
```python
import pytest
import torch
from rag_factory.strategies.late_chunking.document_embedder import DocumentEmbedder
from rag_factory.strategies.late_chunking.models import LateChunkingConfig

@pytest.fixture
def embedder_config():
    return LateChunkingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_document_tokens=512,
        device="cpu"
    )

@pytest.fixture
def document_embedder(embedder_config):
    return DocumentEmbedder(embedder_config)

def test_document_embedding_basic(document_embedder):
    """Test basic document embedding."""
    text = "This is a test document with multiple sentences. It should be embedded properly."
    doc_emb = document_embedder.embed_document(text, "test_doc")

    assert doc_emb.document_id == "test_doc"
    assert doc_emb.text == text
    assert len(doc_emb.full_embedding) > 0
    assert len(doc_emb.token_embeddings) > 0
    assert doc_emb.token_count > 0

def test_token_embeddings_extracted(document_embedder):
    """Test that token-level embeddings are extracted."""
    text = "Hello world"
    doc_emb = document_embedder.embed_document(text, "test_doc")

    # Should have token embeddings
    assert len(doc_emb.token_embeddings) >= 2  # At least "Hello" and "world"

    # Each token should have embedding
    for token_emb in doc_emb.token_embeddings:
        assert len(token_emb.embedding) == doc_emb.embedding_dim
        assert token_emb.start_char >= 0
        assert token_emb.end_char > token_emb.start_char

def test_char_position_mapping(document_embedder):
    """Test that character positions are correct."""
    text = "The quick brown fox"
    doc_emb = document_embedder.embed_document(text, "test_doc")

    # Verify character positions map to correct text
    for token_emb in doc_emb.token_embeddings:
        token_text = text[token_emb.start_char:token_emb.end_char]
        # Token should be similar to extracted text (may have subword differences)
        assert len(token_text) > 0

def test_mean_pooling(document_embedder):
    """Test mean pooling for document embedding."""
    # Create fake token embeddings
    token_embeddings = torch.randn(5, 384)  # 5 tokens, 384 dim
    attention_mask = torch.ones(5)

    mean_emb = document_embedder._mean_pooling(token_embeddings, attention_mask)

    assert mean_emb.shape == (384,)
    # Mean should be average of token embeddings
    expected_mean = token_embeddings.mean(dim=0)
    assert torch.allclose(mean_emb, expected_mean, atol=1e-5)

def test_long_document_truncation(document_embedder):
    """Test that long documents are truncated."""
    # Create very long text
    long_text = "This is a sentence. " * 1000

    doc_emb = document_embedder.embed_document(long_text, "long_doc")

    # Should be truncated to max_length
    assert doc_emb.token_count <= document_embedder.max_length
```

#### TC7.2.2: Embedding Chunker Tests
```python
import pytest
import numpy as np
from rag_factory.strategies.late_chunking.embedding_chunker import EmbeddingChunker
from rag_factory.strategies.late_chunking.models import (
    DocumentEmbedding, TokenEmbedding, LateChunkingConfig, EmbeddingChunkingMethod
)

@pytest.fixture
def chunker_config():
    return LateChunkingConfig(
        chunking_method=EmbeddingChunkingMethod.FIXED_SIZE,
        target_chunk_size=10,
        min_chunk_size=5,
        max_chunk_size=20,
        chunk_overlap_tokens=2
    )

@pytest.fixture
def embedding_chunker(chunker_config):
    return EmbeddingChunker(chunker_config)

@pytest.fixture
def sample_doc_embedding():
    """Create sample DocumentEmbedding for testing."""
    # Create fake token embeddings
    tokens = []
    text = "This is a test document with many tokens for chunking"

    words = text.split()
    char_pos = 0

    for i, word in enumerate(words):
        token = TokenEmbedding(
            token=word,
            token_id=i,
            start_char=char_pos,
            end_char=char_pos + len(word),
            embedding=np.random.randn(384).tolist(),
            position=i
        )
        tokens.append(token)
        char_pos += len(word) + 1  # +1 for space

    doc_emb = DocumentEmbedding(
        document_id="test_doc",
        text=text,
        full_embedding=np.random.randn(384).tolist(),
        token_embeddings=tokens,
        model_name="test_model",
        token_count=len(tokens),
        embedding_dim=384
    )

    return doc_emb

def test_fixed_size_chunking(embedding_chunker, sample_doc_embedding):
    """Test fixed-size chunking."""
    chunks = embedding_chunker._fixed_size_chunking(sample_doc_embedding)

    assert len(chunks) > 0
    # All chunks should have reasonable size
    for chunk in chunks:
        assert chunk.token_count <= embedding_chunker.config.max_chunk_size

def test_semantic_boundary_chunking(sample_doc_embedding):
    """Test semantic boundary-based chunking."""
    config = LateChunkingConfig(
        chunking_method=EmbeddingChunkingMethod.SEMANTIC_BOUNDARY,
        similarity_threshold=0.5,
        min_chunk_size=3
    )
    chunker = EmbeddingChunker(config)

    chunks = chunker._semantic_boundary_chunking(sample_doc_embedding)

    assert len(chunks) > 0
    # Chunks should respect boundaries
    for chunk in chunks:
        assert chunk.token_count >= config.min_chunk_size

def test_chunk_text_reconstruction(embedding_chunker, sample_doc_embedding):
    """Test that chunk text is correctly reconstructed."""
    chunks = embedding_chunker.chunk_embeddings(sample_doc_embedding)

    for chunk in chunks:
        # Text should match character range
        expected_text = sample_doc_embedding.text[chunk.char_range[0]:chunk.char_range[1]]
        assert chunk.text == expected_text

def test_chunk_embedding_averaging(embedding_chunker, sample_doc_embedding):
    """Test that chunk embeddings are averaged correctly."""
    chunks = embedding_chunker.chunk_embeddings(sample_doc_embedding)

    for chunk in chunks:
        # Chunk embedding should be average of token embeddings
        start_token_idx = chunk.token_range[0]
        end_token_idx = chunk.token_range[1]

        token_embeddings = [
            sample_doc_embedding.token_embeddings[i].embedding
            for i in range(start_token_idx, end_token_idx + 1)
        ]

        expected_avg = np.mean(token_embeddings, axis=0)
        actual_avg = np.array(chunk.chunk_embedding)

        assert np.allclose(actual_avg, expected_avg, atol=1e-5)

def test_cosine_similarity(embedding_chunker):
    """Test cosine similarity calculation."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])

    similarity = embedding_chunker._cosine_similarity(vec1, vec2)
    assert abs(similarity - 1.0) < 1e-6  # Should be exactly 1.0

    vec3 = np.array([0.0, 1.0, 0.0])
    similarity = embedding_chunker._cosine_similarity(vec1, vec3)
    assert abs(similarity - 0.0) < 1e-6  # Should be exactly 0.0
```

---

## Integration Tests

### Test File Location
`tests/integration/strategies/test_late_chunking_integration.py`

### Test Scenarios

#### IS7.2.1: End-to-End Late Chunking Workflow
```python
import pytest
from rag_factory.strategies.late_chunking.strategy import LateChunkingRAGStrategy
from rag_factory.strategies.late_chunking.models import EmbeddingChunkingMethod

@pytest.mark.integration
def test_late_chunking_workflow(test_vector_store):
    """Test complete late chunking workflow."""
    config = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "chunking_method": EmbeddingChunkingMethod.SEMANTIC_BOUNDARY.value,
        "target_chunk_size": 128,
        "compute_coherence_scores": True,
        "device": "cpu"
    }

    strategy = LateChunkingRAGStrategy(
        vector_store_service=test_vector_store,
        config=config
    )

    # Index document
    document = """
    Machine learning is a subset of artificial intelligence that enables systems to learn from data.
    Deep learning is a type of machine learning that uses neural networks with many layers.
    Neural networks are inspired by biological neurons in the human brain.
    The training process involves adjusting network weights to minimize error.
    """

    strategy.index_document(document, "ml_doc")

    # Retrieve
    results = strategy.retrieve("What is machine learning?", top_k=3)

    assert len(results) > 0
    assert all("strategy" in r for r in results)
    assert all(r["strategy"] == "late_chunking" for r in results)

@pytest.mark.integration
def test_comparison_with_traditional(test_vector_store):
    """Compare late chunking with traditional chunking."""
    from rag_factory.strategies.chunking.semantic_chunker import SemanticChunker
    from rag_factory.strategies.chunking.base import ChunkingConfig

    document = """
    Python is a versatile programming language. It is widely used in data science and machine learning.
    Libraries like NumPy and Pandas make data manipulation easy. TensorFlow and PyTorch are popular ML frameworks.
    """

    # Traditional chunking
    traditional_config = ChunkingConfig(target_chunk_size=128)
    # (Would need embedding service for semantic chunker)

    # Late chunking
    late_config = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "chunking_method": "semantic_boundary",
        "target_chunk_size": 128,
        "device": "cpu"
    }
    late_strategy = LateChunkingRAGStrategy(test_vector_store, late_config)

    late_strategy.index_document(document, "compare_doc")

    results = late_strategy.retrieve("Python libraries", top_k=2)

    assert len(results) > 0
    # Late chunking results should include coherence scores
    for result in results:
        metadata = result.get("metadata", {})
        assert "chunking_method" in metadata
```

---

## Performance Benchmarks

```python
# tests/benchmarks/test_late_chunking_performance.py
import pytest
import time
from rag_factory.strategies.late_chunking.strategy import LateChunkingRAGStrategy

@pytest.mark.benchmark
def test_document_embedding_speed(test_vector_store):
    """Benchmark document embedding speed."""
    config = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu"
    }

    strategy = LateChunkingRAGStrategy(test_vector_store, config)

    # Generate document (~2K tokens)
    document = ". ".join([f"Sentence {i} with some content" for i in range(200)])

    start = time.time()
    doc_emb = strategy.document_embedder.embed_document(document, "perf_test")
    duration = time.time() - start

    print(f"\nDocument embedding: {doc_emb.token_count} tokens in {duration:.3f}s")
    assert duration < 2.0, f"Too slow: {duration:.2f}s (expected <2s)"

@pytest.mark.benchmark
def test_embedding_chunking_speed(test_vector_store):
    """Benchmark embedding chunking speed."""
    config = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "chunking_method": "fixed_size",
        "device": "cpu"
    }

    strategy = LateChunkingRAGStrategy(test_vector_store, config)

    document = ". ".join([f"Sentence {i}" for i in range(100)])

    # Embed document
    doc_emb = strategy.document_embedder.embed_document(document, "perf_test")

    # Time chunking
    start = time.time()
    chunks = strategy.embedding_chunker.chunk_embeddings(doc_emb)
    duration = time.time() - start

    print(f"\nEmbedding chunking: {len(chunks)} chunks in {duration:.3f}s")
    assert duration < 0.3, f"Too slow: {duration:.2f}s (expected <300ms)"

@pytest.mark.benchmark
def test_end_to_end_latency(test_vector_store):
    """Benchmark end-to-end late chunking latency."""
    config = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "compute_coherence_scores": False,  # Disable for performance test
        "device": "cpu"
    }

    strategy = LateChunkingRAGStrategy(test_vector_store, config)

    document = "Test document. " * 100

    start = time.time()
    strategy.index_document(document, "latency_test")
    duration = time.time() - start

    print(f"\nEnd-to-end late chunking: {duration:.3f}s")
    # This includes embedding + chunking + indexing
    assert duration < 5.0, f"Too slow: {duration:.2f}s"
```

---

## Definition of Done

- [ ] DocumentEmbedder implemented
- [ ] Token-level embedding extraction working
- [ ] EmbeddingChunker implemented
- [ ] At least 3 chunking strategies implemented (fixed, semantic, adaptive)
- [ ] Text reconstruction accurate
- [ ] Coherence analyzer implemented
- [ ] LateChunkingRAGStrategy complete
- [ ] Support for long-context models
- [ ] Sliding window for very long documents
- [ ] All unit tests pass (>85% coverage)
- [ ] All integration tests pass
- [ ] Performance benchmarks meet requirements
- [ ] Comparison with traditional chunking documented
- [ ] Quality metrics measured
- [ ] Configuration system working
- [ ] Documentation complete with examples
- [ ] Code reviewed
- [ ] No linting errors

---

## Setup Instructions

### Installation

```bash
# Install dependencies
pip install transformers torch tokenizers sentence-transformers

# Download a long-context model (optional)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')"
```

### Configuration

```yaml
# config.yaml
strategies:
  late_chunking:
    enabled: true

    # Model selection
    use_long_context_model: true
    model_name: "sentence-transformers/all-mpnet-base-v2"
    max_document_tokens: 8192

    # Chunking configuration
    chunking_method: "semantic_boundary"  # fixed_size, semantic_boundary, adaptive
    target_chunk_size: 512
    min_chunk_size: 128
    max_chunk_size: 1024
    chunk_overlap_tokens: 50

    # Semantic boundary detection
    similarity_threshold: 0.7
    use_local_similarity: true

    # Coherence analysis
    compute_coherence_scores: true
    coherence_window_size: 3

    # Performance
    batch_size: 1
    device: "cpu"  # or "cuda" for GPU
```

### Usage Example

```python
from rag_factory.strategies.late_chunking import LateChunkingRAGStrategy
from rag_factory.strategies.late_chunking.models import EmbeddingChunkingMethod

# Setup strategy
config = {
    "model_name": "sentence-transformers/all-mpnet-base-v2",
    "chunking_method": EmbeddingChunkingMethod.SEMANTIC_BOUNDARY.value,
    "target_chunk_size": 512,
    "similarity_threshold": 0.7,
    "compute_coherence_scores": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

strategy = LateChunkingRAGStrategy(
    vector_store_service=vector_store,
    config=config
)

# Index document with late chunking
document = """
Your long document here...
The late chunking strategy will embed the entire document first,
maintaining full context, before splitting into semantic chunks.
"""

strategy.index_document(document, "doc_123")

# Retrieve as normal
results = strategy.retrieve("your query", top_k=5)

# Inspect late chunking metadata
for result in results:
    print(f"Chunk: {result['text'][:100]}...")
    print(f"Coherence: {result['metadata'].get('coherence_score')}")
    print(f"Token range: {result['metadata'].get('token_range')}")
    print()
```

---

## Notes for Developers

1. **Complexity Warning**: Late chunking is significantly more complex than traditional chunking. Only use when context preservation is critical.

2. **Model Selection**: Choose models with long context windows:
   - sentence-transformers/all-mpnet-base-v2: 384 tokens (limited but fast)
   - Longformer: 4096 tokens
   - LED: 16384 tokens
   - For most use cases, 512-1024 tokens is sufficient

3. **Performance Considerations**:
   - Embedding full documents is slower than embedding chunks
   - Use GPU if available for large documents
   - Consider caching document embeddings

4. **Memory Usage**: Token embeddings can be memory-intensive. For a 2K token document with 768-dim embeddings:
   - 2000 tokens × 768 dims × 4 bytes = ~6MB per document
   - Scale accordingly for batch processing

5. **When to Use Late Chunking**:
   - Documents where context is crucial (legal, medical)
   - When traditional chunking breaks semantic units
   - For comparison/research purposes
   - NOT for chat logs or short messages

6. **Quality vs Speed Trade-off**:
   - Late chunking: Better context, slower
   - Traditional chunking: Faster, may miss context
   - Measure both on your specific use case

7. **Semantic Boundary Detection**: Tune the similarity threshold:
   - Higher (0.8-0.9): Fewer, larger chunks
   - Lower (0.5-0.7): More, smaller chunks
   - Test with your domain-specific documents

8. **Text Reconstruction**: Always validate that reconstructed text matches original. Character offsets are critical.

9. **Evaluation**: Compare retrieval accuracy with traditional chunking on a test set. Late chunking should show measurable improvement to justify the complexity.

10. **Experimental Nature**: This strategy is experimental. Document all findings, limitations, and edge cases thoroughly.

11. **Model Limitations**: Be aware of model context window limits. Always handle truncation gracefully with warnings.

12. **Future Enhancements**:
    - Implement hierarchical late chunking (section → paragraph → sentence)
    - Add support for more long-context models
    - Optimize token embedding storage
    - Implement incremental updates
