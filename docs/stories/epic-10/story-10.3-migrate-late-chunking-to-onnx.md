# Story 10.3: Migrate Late Chunking to ONNX

**Story ID:** 10.3
**Epic:** Epic 10 - Lightweight Dependencies Implementation
**Story Points:** 8
**Priority:** Medium
**Dependencies:** Story 10.1 (ONNX embeddings), Story 10.2 (tiktoken)

---

## User Story

**As a** developer
**I want** late chunking strategy to use ONNX models
**So that** this experimental feature doesn't require PyTorch

---

## Detailed Requirements

### Functional Requirements

1. **Document Embedder Migration**
   - Update `DocumentEmbedder` to use ONNX runtime instead of PyTorch
   - Replace transformers model loading with ONNX models
   - Maintain token-level embedding extraction capability
   - Support long-context ONNX models (up to 8192 tokens)
   - Preserve embedding quality and accuracy
   - Handle model loading and caching

2. **Tokenization Update**
   - Replace transformers tokenizer with tiktoken
   - Maintain accurate token-to-text mapping
   - Support special tokens handling
   - Preserve tokenization consistency
   - Handle edge cases (very long documents, special characters)

3. **Token-Level Embeddings**
   - Extract token-level embeddings from ONNX models
   - Maintain embedding alignment with tokens
   - Support different pooling strategies (none, mean, max)
   - Handle variable-length sequences
   - Preserve embedding dimensionality

4. **Long-Context Support**
   - Support ONNX models with long context windows
   - Handle documents up to 8192 tokens
   - Implement efficient chunking for very long documents
   - Maintain context across chunks
   - Optimize memory usage for long sequences

5. **Embedding Chunker Update**
   - Update chunking logic to work with ONNX embeddings
   - Maintain all splitting strategies (fixed-size, semantic, adaptive)
   - Preserve chunk boundary detection
   - Ensure coherence analysis still works
   - Optimize performance

6. **Testing and Validation**
   - Test with various document lengths
   - Verify chunking quality maintained
   - Benchmark performance
   - Test with different ONNX models

### Non-Functional Requirements

1. **Performance**
   - Document embedding: <500ms for 2048 tokens
   - Memory usage: <1GB for long documents
   - Model load time: <5 seconds
   - Chunking speed: comparable to original implementation

2. **Quality**
   - Embedding similarity to PyTorch: >99%
   - Chunking quality: maintained or improved
   - Token alignment: 100% accurate
   - No loss of functionality

3. **Compatibility**
   - Works with ONNX embedding models
   - Compatible with tiktoken encodings
   - Supports all original features
   - Backward compatible configuration

4. **Resource Efficiency**
   - No PyTorch dependency (~2.5GB saved)
   - No transformers dependency (~500MB saved)
   - Total savings: ~3GB installation size
   - Faster installation and startup

---

## Acceptance Criteria

### AC1: Document Embedder Migration
- [ ] `DocumentEmbedder` uses ONNX runtime
- [ ] PyTorch code completely removed
- [ ] Token-level embedding extraction working
- [ ] Long-context models supported (up to 8192 tokens)
- [ ] Model loading and caching working
- [ ] Embedding quality validated (>99% similarity)

### AC2: Tokenization Update
- [ ] Transformers tokenizer replaced with tiktoken
- [ ] Token-to-text mapping accurate
- [ ] Special tokens handled correctly
- [ ] Edge cases handled properly
- [ ] Tokenization tests passing

### AC3: Token-Level Embeddings
- [ ] Token-level embeddings extracted correctly
- [ ] Embedding alignment with tokens verified
- [ ] Different pooling strategies working
- [ ] Variable-length sequences supported
- [ ] Embedding dimensions correct

### AC4: Long-Context Support
- [ ] Documents up to 8192 tokens supported
- [ ] Very long documents chunked efficiently
- [ ] Context maintained across chunks
- [ ] Memory usage optimized
- [ ] Performance targets met

### AC5: Embedding Chunker
- [ ] All splitting strategies working
- [ ] Fixed-size splitting working
- [ ] Semantic boundary detection working
- [ ] Adaptive splitting working
- [ ] Coherence analysis working

### AC6: Testing
- [ ] Unit tests updated for ONNX
- [ ] Integration tests passing
- [ ] Quality validation tests passing
- [ ] Performance benchmarks met
- [ ] All tests passing without PyTorch/transformers

---

## Technical Specifications

### File Structure
```
rag_factory/
├── strategies/
│   ├── late_chunking/
│   │   ├── __init__.py
│   │   ├── strategy.py
│   │   ├── document_embedder.py    # UPDATED: Use ONNX
│   │   ├── embedding_chunker.py
│   │   ├── coherence_analyzer.py
│   │   └── config.py

tests/
├── unit/
│   └── strategies/
│       └── late_chunking/
│           ├── test_document_embedder.py    # UPDATED
│           ├── test_embedding_chunker.py
│           └── test_strategy.py
│
└── integration/
    └── strategies/
        └── test_late_chunking_integration.py
```

### Updated Document Embedder
```python
# rag_factory/strategies/late_chunking/document_embedder.py
from typing import List, Optional, Tuple
import numpy as np
import onnxruntime as ort
from pathlib import Path
import logging
from rag_factory.utils.tokenization import Tokenizer

logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """
    Embeds entire documents at token level using ONNX models.
    Replaces PyTorch-based implementation.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_path: Optional[str] = None,
        max_length: int = 8192,
        encoding: str = "cl100k_base"
    ):
        """
        Initialize document embedder.

        Args:
            model_name: HuggingFace model name
            model_path: Path to local ONNX model
            max_length: Maximum sequence length
            encoding: Tiktoken encoding name
        """
        self.model_name = model_name
        self.max_length = max_length

        # Initialize tokenizer (using tiktoken)
        self.tokenizer = Tokenizer(encoding_name=encoding)

        # Load ONNX model
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = self._get_model_path(model_name)

        self.session = self._create_session(self.model_path)

        # Get embedding dimension
        self.embedding_dim = self._get_embedding_dimension()

        logger.info(
            f"Initialized DocumentEmbedder with model: {model_name}, "
            f"max_length: {max_length}, embedding_dim: {self.embedding_dim}"
        )

    def _get_model_path(self, model_name: str) -> Path:
        """Get path to ONNX model."""
        from huggingface_hub import hf_hub_download

        cache_dir = Path.home() / ".cache" / "rag_factory" / "onnx_models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            model_file = hf_hub_download(
                repo_id=model_name,
                filename="model.onnx",
                cache_dir=str(cache_dir)
            )
            return Path(model_file)
        except Exception as e:
            raise ValueError(
                f"Could not download ONNX model '{model_name}'. "
                f"Please convert the model to ONNX format first. "
                f"Error: {e}"
            )

    def _create_session(self, model_path: Path) -> ort.InferenceSession:
        """Create ONNX Runtime session."""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )

        return session

    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension from model."""
        output_meta = self.session.get_outputs()[0]
        shape = output_meta.shape
        return shape[-1]  # Last dimension is embedding dim

    def embed_document(
        self,
        text: str,
        return_tokens: bool = True
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        """
        Embed entire document at token level.

        Args:
            text: Document text
            return_tokens: Whether to return token strings

        Returns:
            Tuple of (token_embeddings, tokens)
            - token_embeddings: [num_tokens, embedding_dim]
            - tokens: List of token strings (if return_tokens=True)
        """
        # Tokenize
        token_ids = self.tokenizer.encode(text)

        # Truncate if needed
        if len(token_ids) > self.max_length:
            logger.warning(
                f"Document has {len(token_ids)} tokens, "
                f"truncating to {self.max_length}"
            )
            token_ids = token_ids[:self.max_length]

        # Prepare inputs
        input_ids = np.array([token_ids], dtype=np.int64)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)

        # Run inference
        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        )

        # Extract token-level embeddings
        # Output shape: [batch_size, seq_length, embedding_dim]
        token_embeddings = outputs[0][0]  # Remove batch dimension

        # Get token strings if requested
        tokens = None
        if return_tokens:
            tokens = self._decode_tokens(token_ids)

        return token_embeddings, tokens

    def _decode_tokens(self, token_ids: List[int]) -> List[str]:
        """
        Decode token IDs to token strings.

        Args:
            token_ids: List of token IDs

        Returns:
            List of token strings
        """
        # Decode each token individually
        tokens = []
        for token_id in token_ids:
            try:
                token_str = self.tokenizer.decode([token_id])
                tokens.append(token_str)
            except Exception:
                tokens.append(f"<unk_{token_id}>")

        return tokens

    def chunk_embeddings(
        self,
        token_embeddings: np.ndarray,
        tokens: List[str],
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[Tuple[np.ndarray, List[str], int, int]]:
        """
        Chunk token embeddings into smaller pieces.

        Args:
            token_embeddings: Token-level embeddings [num_tokens, dim]
            tokens: Token strings
            chunk_size: Tokens per chunk
            overlap: Overlapping tokens

        Returns:
            List of (chunk_embeddings, chunk_tokens, start_idx, end_idx)
        """
        num_tokens = len(token_embeddings)
        chunks = []

        start = 0
        while start < num_tokens:
            end = min(start + chunk_size, num_tokens)

            chunk_emb = token_embeddings[start:end]
            chunk_tok = tokens[start:end]

            chunks.append((chunk_emb, chunk_tok, start, end))

            # Move to next chunk with overlap
            start = end - overlap
            if start >= num_tokens:
                break

        return chunks

    def pool_embeddings(
        self,
        token_embeddings: np.ndarray,
        method: str = "mean"
    ) -> np.ndarray:
        """
        Pool token embeddings to document embedding.

        Args:
            token_embeddings: Token-level embeddings [num_tokens, dim]
            method: Pooling method ("mean", "max", "first")

        Returns:
            Document embedding [dim]
        """
        if method == "mean":
            return np.mean(token_embeddings, axis=0)
        elif method == "max":
            return np.max(token_embeddings, axis=0)
        elif method == "first":
            return token_embeddings[0]
        else:
            raise ValueError(f"Unknown pooling method: {method}")
```

---

## Testing Strategy

### Unit Tests
```python
# tests/unit/strategies/late_chunking/test_document_embedder.py
import pytest
import numpy as np
from unittest.mock import Mock, patch
from rag_factory.strategies.late_chunking.document_embedder import DocumentEmbedder


class TestDocumentEmbedder:
    """Test ONNX-based document embedder."""

    @pytest.fixture
    def mock_session(self):
        """Mock ONNX session."""
        session = Mock()
        session.get_outputs.return_value = [
            Mock(shape=[1, 512, 384])  # [batch, seq, dim]
        ]
        return session

    @pytest.fixture
    def embedder(self, mock_session):
        """Create embedder with mocked session."""
        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            with patch.object(DocumentEmbedder, "_get_model_path"):
                embedder = DocumentEmbedder()
                embedder.session = mock_session
                return embedder

    def test_initialization(self, embedder):
        """Test embedder initialization."""
        assert embedder.embedding_dim == 384
        assert embedder.max_length == 8192

    def test_embed_document(self, embedder, mock_session):
        """Test document embedding."""
        # Mock output
        mock_embeddings = np.random.randn(1, 10, 384).astype(np.float32)
        mock_session.run.return_value = [mock_embeddings]

        text = "This is a test document."
        token_embeddings, tokens = embedder.embed_document(text)

        assert token_embeddings.shape[1] == 384  # Embedding dim
        assert tokens is not None
        assert len(tokens) == token_embeddings.shape[0]

    def test_long_document_truncation(self, embedder, mock_session):
        """Test truncation of long documents."""
        # Create very long text
        long_text = "word " * 10000

        mock_embeddings = np.random.randn(1, embedder.max_length, 384).astype(np.float32)
        mock_session.run.return_value = [mock_embeddings]

        token_embeddings, tokens = embedder.embed_document(long_text)

        # Should be truncated to max_length
        assert token_embeddings.shape[0] <= embedder.max_length

    def test_chunk_embeddings(self, embedder):
        """Test chunking of token embeddings."""
        # Create mock embeddings
        token_embeddings = np.random.randn(100, 384)
        tokens = [f"token_{i}" for i in range(100)]

        chunks = embedder.chunk_embeddings(
            token_embeddings,
            tokens,
            chunk_size=30,
            overlap=5
        )

        assert len(chunks) > 1
        for chunk_emb, chunk_tok, start, end in chunks:
            assert chunk_emb.shape[0] == len(chunk_tok)
            assert chunk_emb.shape[1] == 384
            assert end - start == len(chunk_tok)

    def test_pool_embeddings_mean(self, embedder):
        """Test mean pooling."""
        token_embeddings = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])

        pooled = embedder.pool_embeddings(token_embeddings, method="mean")

        expected = np.array([4.0, 5.0, 6.0])  # Mean of columns
        np.testing.assert_array_almost_equal(pooled, expected)

    def test_pool_embeddings_max(self, embedder):
        """Test max pooling."""
        token_embeddings = np.array([
            [1.0, 5.0, 3.0],
            [4.0, 2.0, 6.0],
            [7.0, 8.0, 1.0]
        ])

        pooled = embedder.pool_embeddings(token_embeddings, method="max")

        expected = np.array([7.0, 8.0, 6.0])  # Max of columns
        np.testing.assert_array_almost_equal(pooled, expected)

    def test_pool_embeddings_first(self, embedder):
        """Test first token pooling."""
        token_embeddings = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])

        pooled = embedder.pool_embeddings(token_embeddings, method="first")

        expected = np.array([1.0, 2.0, 3.0])  # First row
        np.testing.assert_array_almost_equal(pooled, expected)
```

### Integration Tests
```python
# tests/integration/strategies/test_late_chunking_integration.py
import pytest
import numpy as np
from rag_factory.strategies.late_chunking import LateChunkingStrategy


@pytest.mark.integration
class TestLateChunkingIntegration:
    """Integration tests for late chunking with ONNX."""

    @pytest.fixture(scope="class")
    def strategy(self):
        """Create late chunking strategy."""
        return LateChunkingStrategy(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=512,
            overlap=50
        )

    def test_embed_and_chunk_document(self, strategy):
        """Test full pipeline: embed and chunk."""
        text = """
        This is a test document with multiple sentences.
        It should be embedded at the token level.
        Then chunked into smaller pieces.
        Each chunk should maintain semantic coherence.
        """

        # Embed document
        token_embeddings, tokens = strategy.embedder.embed_document(text)

        assert token_embeddings.shape[1] == strategy.embedder.embedding_dim
        assert len(tokens) == token_embeddings.shape[0]

        # Chunk embeddings
        chunks = strategy.embedder.chunk_embeddings(
            token_embeddings,
            tokens,
            chunk_size=20,
            overlap=5
        )

        assert len(chunks) > 0

    def test_quality_vs_pytorch(self, strategy):
        """Test that ONNX embeddings are similar to PyTorch."""
        # This would compare against saved PyTorch embeddings
        # For now, just verify embeddings are reasonable

        text = "The quick brown fox jumps over the lazy dog."
        token_embeddings, _ = strategy.embedder.embed_document(text)

        # Check embeddings are normalized
        norms = np.linalg.norm(token_embeddings, axis=1)
        assert np.all(norms > 0)  # Non-zero

        # Check reasonable range
        assert np.all(np.abs(token_embeddings) < 10)  # Not exploding

    def test_long_document_handling(self, strategy):
        """Test handling of long documents."""
        # Create long document
        long_text = "This is a sentence. " * 500

        token_embeddings, tokens = strategy.embedder.embed_document(long_text)

        assert token_embeddings.shape[0] > 0
        assert token_embeddings.shape[0] <= strategy.embedder.max_length

    def test_performance(self, strategy):
        """Test embedding performance."""
        import time

        text = "This is a test document. " * 100  # ~2000 tokens

        start = time.time()
        token_embeddings, tokens = strategy.embedder.embed_document(text)
        elapsed = time.time() - start

        # Should be < 500ms for 2048 tokens
        assert elapsed < 0.5
```

---

## Implementation Plan

### Phase 1: Document Embedder Migration (Days 1-3)
1. Update `DocumentEmbedder` to use ONNX runtime
2. Replace PyTorch model loading
3. Implement token-level embedding extraction
4. Add long-context support
5. Test and validate

### Phase 2: Tokenization Update (Day 4)
1. Replace transformers tokenizer with tiktoken
2. Update token-to-text mapping
3. Handle special tokens
4. Test tokenization accuracy

### Phase 3: Integration (Days 5-6)
1. Update embedding chunker
2. Ensure all splitting strategies work
3. Validate coherence analysis
4. Integration testing

### Phase 4: Testing and Validation (Days 7-8)
1. Write/update unit tests
2. Write integration tests
3. Validate quality (>99% similarity)
4. Performance benchmarks
5. Documentation updates

---

## Risks and Mitigation

### Risk: Embedding Quality Loss
**Impact:** High
**Probability:** Low
**Mitigation:**
- Validate against PyTorch version
- Test with multiple models
- Document any differences

### Risk: Token Alignment Issues
**Impact:** Medium
**Probability:** Medium
**Mitigation:**
- Thorough testing of tokenization
- Validate token-to-embedding mapping
- Handle edge cases

### Risk: Performance Degradation
**Impact:** Medium
**Probability:** Low
**Mitigation:**
- Benchmark early
- Optimize ONNX runtime settings
- Use efficient numpy operations

---

## Success Metrics

- [ ] ONNX embeddings match PyTorch (>99% similarity)
- [ ] All late chunking features working
- [ ] Performance targets met (<500ms for 2048 tokens)
- [ ] No PyTorch/transformers dependencies
- [ ] All tests passing
- [ ] Documentation updated

---

## Dependencies

**Blocked by:** Story 10.1 (ONNX embeddings), Story 10.2 (tiktoken)
**Blocks:** None
**Related:** Epic 7 (experimental strategies)
