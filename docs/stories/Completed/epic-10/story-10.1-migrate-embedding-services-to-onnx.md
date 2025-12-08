# Story 10.1: Migrate Embedding Services to ONNX

**Story ID:** 10.1
**Epic:** Epic 10 - Lightweight Dependencies Implementation
**Story Points:** 8
**Priority:** High
**Dependencies:** Epic 7 (experimental strategies), Epic 4 (reranking)

---

## User Story

**As a** developer
**I want** embedding services to use ONNX runtime instead of PyTorch
**So that** I can deploy without heavy ML dependencies

---

## Detailed Requirements

### Functional Requirements

1. **ONNX Provider Enhancement**
   - Update `onnx_local.py` provider to be the primary local embedding option
   - Support ONNX model loading from HuggingFace Hub
   - Implement automatic model download and caching
   - Support multiple ONNX model formats (standard, quantized, optimized)
   - Handle model metadata and configuration
   - Provide model validation on load

2. **PyTorch Dependency Removal**
   - Remove torch imports from embedding providers
   - Replace torch tensor operations with numpy
   - Update all embedding-related code to use ONNX runtime
   - Ensure no torch dependencies remain in core embedding path
   - Add runtime checks with helpful error messages if torch is referenced

3. **Model Conversion Support**
   - Provide conversion utilities for PyTorch → ONNX
   - Support conversion of popular embedding models (sentence-transformers, etc.)
   - Validate converted models for accuracy
   - Document conversion process and best practices
   - Provide example conversion scripts

4. **Embedding Quality Maintenance**
   - Ensure ONNX embeddings match PyTorch quality (>99% similarity)
   - Validate embedding dimensions and normalization
   - Test with multiple model architectures
   - Benchmark embedding quality across common models
   - Document any quality differences or limitations

5. **Performance Optimization**
   - Optimize ONNX runtime settings for CPU inference
   - Implement batch processing for multiple documents
   - Add caching for frequently used embeddings
   - Monitor and optimize memory usage
   - Ensure inference speed meets targets (<100ms per document)

6. **Documentation Updates**
   - Document ONNX model usage and configuration
   - Provide model conversion guide
   - List supported ONNX models and sources
   - Add troubleshooting guide for common issues
   - Include performance tuning recommendations

### Non-Functional Requirements

1. **Performance**
   - Embedding speed: <100ms per document (CPU)
   - Memory usage: <500MB for model + inference
   - Model load time: <5 seconds
   - Batch processing: support up to 32 documents simultaneously

2. **Reliability**
   - Handle model loading failures gracefully
   - Validate model compatibility before loading
   - Provide clear error messages for unsupported models
   - Automatic fallback to default models if custom model fails

3. **Compatibility**
   - Support Linux, macOS, and Windows
   - Work on CPU-only systems (no GPU required)
   - Compatible with Python 3.8+
   - Support both x86 and ARM architectures

4. **Maintainability**
   - Clear separation between ONNX and other providers
   - Well-documented model loading process
   - Comprehensive error handling
   - Extensive logging for debugging

5. **Resource Efficiency**
   - Installation size: ONNX runtime ~200MB (vs PyTorch ~2.5GB)
   - Installation time: <2 minutes (vs 20+ minutes with torch)
   - Memory footprint: <1GB total for embedding service
   - No CUDA dependencies required

---

## Acceptance Criteria

### AC1: ONNX Provider Implementation
- [ ] `onnx_local.py` provider is fully functional
- [ ] Supports loading models from HuggingFace Hub
- [ ] Supports loading local ONNX models
- [ ] Model caching implemented and working
- [ ] Model validation on load working
- [ ] Supports quantized and optimized ONNX models

### AC2: PyTorch Removal
- [ ] All torch imports removed from embedding providers
- [ ] Torch dependency removed from requirements.txt
- [ ] All tensor operations use numpy instead
- [ ] No torch code in core embedding path
- [ ] Runtime checks prevent accidental torch usage

### AC3: Model Conversion
- [ ] Conversion utility script created
- [ ] Supports sentence-transformers models
- [ ] Supports HuggingFace transformers models
- [ ] Conversion validation implemented
- [ ] Example conversions documented
- [ ] Conversion guide published

### AC4: Quality Validation
- [ ] Embedding dimensions correct
- [ ] Normalization working correctly
- [ ] Tested with at least 3 different model architectures
- [ ] Quality benchmarks documented

### AC5: Performance
- [ ] Embedding speed <100ms per document on CPU
- [ ] Memory usage <500MB for model + inference
- [ ] Model load time <5 seconds
- [ ] Batch processing working for up to 32 documents
- [ ] Performance benchmarks documented

### AC6: Documentation
- [ ] ONNX model usage guide written
- [ ] Model conversion guide published
- [ ] Supported models list created
- [ ] Troubleshooting guide added
- [ ] Performance tuning guide included
- [ ] API documentation updated

### AC7: Testing
- [ ] Unit tests for ONNX provider with mocked models
- [ ] Integration tests with real ONNX models
- [ ] Performance benchmarks implemented
- [ ] Quality validation tests passing
- [ ] Cross-platform tests (Linux, macOS, Windows)
- [ ] All tests passing without torch dependency

---

## Technical Specifications

### File Structure
```
rag_factory/
├── services/
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── onnx_local.py         # Enhanced ONNX provider (PRIMARY)
│   │   ├── openai.py              # API provider
│   │   ├── cohere.py              # API provider
│   │   └── config.py
│   │
│   └── utils/
│       ├── onnx_utils.py          # ONNX utilities
│       └── model_converter.py    # PyTorch → ONNX conversion

scripts/
├── convert_model_to_onnx.py      # Model conversion script
└── validate_onnx_model.py        # Model validation script

tests/
├── unit/
│   └── services/
│       └── embeddings/
│           ├── test_onnx_local.py
│           └── test_model_converter.py
│
├── integration/
│   └── services/
│       └── test_onnx_embeddings_integration.py
│
└── performance/
    └── benchmark_onnx_embeddings.py
```

### Dependencies
```python
# requirements.txt - Lightweight dependencies
onnx>=1.15.0                    # ~15MB - Model format
onnxruntime>=1.16.3             # ~200MB - CPU-optimized runtime
numpy>=1.24.0                   # ~15MB - Array operations
optimum>=1.16.0                 # Model conversion utilities
huggingface-hub>=0.20.0         # Model downloads

# REMOVED (saving ~3GB):
# torch>=2.1.2                  # ~2.5GB + CUDA
# transformers>=4.36.0          # ~500MB
```

### ONNX Provider Implementation
```python
# rag_factory/services/embeddings/onnx_local.py
from typing import List, Optional, Dict, Any
import numpy as np
import onnxruntime as ort
from pathlib import Path
import logging
from huggingface_hub import hf_hub_download
from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class ONNXLocalEmbedding(EmbeddingProvider):
    """
    ONNX-based local embedding provider.
    Uses ONNX Runtime for CPU-optimized inference without PyTorch.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize ONNX embedding provider.

        Args:
            model_name: HuggingFace model name (will auto-download ONNX version)
            model_path: Path to local ONNX model file
            cache_dir: Directory for model caching
            **kwargs: Additional ONNX runtime options
        """
        super().__init__()
        self.model_name = model_name
        self.cache_dir = cache_dir or Path.home() / ".cache" / "rag_factory" / "onnx_models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = self._download_model(model_name)

        # Initialize ONNX Runtime session
        self.session = self._create_session(self.model_path, **kwargs)

        # Get model metadata
        self.embedding_dim = self._get_embedding_dimension()

        logger.info(
            f"Loaded ONNX model: {model_name}, "
            f"embedding_dim: {self.embedding_dim}"
        )

    def _download_model(self, model_name: str) -> Path:
        """
        Download ONNX model from HuggingFace Hub.

        Args:
            model_name: HuggingFace model identifier

        Returns:
            Path to downloaded model file
        """
        logger.info(f"Downloading ONNX model: {model_name}")

        try:
            # Try to download ONNX version
            model_file = hf_hub_download(
                repo_id=model_name,
                filename="model.onnx",
                cache_dir=str(self.cache_dir)
            )
            return Path(model_file)

        except Exception as e:
            logger.error(f"Failed to download ONNX model: {e}")
            raise ValueError(
                f"Could not download ONNX model '{model_name}'. "
                f"Please ensure the model has an ONNX version available, "
                f"or convert it using the conversion script. "
                f"See documentation for model conversion guide."
            )

    def _create_session(
        self,
        model_path: Path,
        providers: Optional[List[str]] = None,
        **session_options
    ) -> ort.InferenceSession:
        """
        Create ONNX Runtime inference session.

        Args:
            model_path: Path to ONNX model
            providers: Execution providers (default: CPU)
            **session_options: Additional session options

        Returns:
            ONNX Runtime session
        """
        # Default to CPU provider (no GPU required)
        if providers is None:
            providers = ["CPUExecutionProvider"]

        # Create session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Apply custom session options
        for key, value in session_options.items():
            setattr(sess_options, key, value)

        # Create session
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers
        )

        logger.info(f"Created ONNX session with providers: {session.get_providers()}")
        return session

    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension from model output."""
        # Get output shape from model metadata
        output_meta = self.session.get_outputs()[0]
        shape = output_meta.shape

        # Last dimension is usually embedding dimension
        # Shape is typically [batch_size, sequence_length, embedding_dim]
        # or [batch_size, embedding_dim]
        if len(shape) >= 2:
            return shape[-1]
        else:
            raise ValueError(f"Unexpected output shape: {shape}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for documents.

        Args:
            texts: List of text documents

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.debug(f"Embedding {len(texts)} documents")

        # Tokenize texts (using simple tokenization for now)
        # In production, would use tiktoken or similar
        input_ids, attention_mask = self._tokenize(texts)

        # Run inference
        outputs = self.session.run(
            None,  # Get all outputs
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        )

        # Extract embeddings (usually first output)
        embeddings = outputs[0]

        # Mean pooling if needed
        if len(embeddings.shape) == 3:  # [batch, seq_len, dim]
            embeddings = self._mean_pooling(embeddings, attention_mask)

        # Normalize embeddings
        embeddings = self._normalize(embeddings)

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            text: Query text

        Returns:
            Embedding vector
        """
        return self.embed_documents([text])[0]

    def _tokenize(self, texts: List[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Tokenize texts for model input.

        Args:
            texts: List of texts

        Returns:
            Tuple of (input_ids, attention_mask)
        """
        # Simplified tokenization
        # In production, use tiktoken or proper tokenizer
        # This is a placeholder - actual implementation would use
        # the model's tokenizer configuration

        max_length = 512
        vocab_size = 30522  # BERT vocab size

        # Simple word-based tokenization (placeholder)
        input_ids = []
        attention_mask = []

        for text in texts:
            # Simplified: just use character codes (NOT PRODUCTION READY)
            # Real implementation would use proper tokenizer
            tokens = [ord(c) % vocab_size for c in text[:max_length]]
            padding = [0] * (max_length - len(tokens))

            input_ids.append(tokens + padding)
            attention_mask.append([1] * len(tokens) + padding)

        return (
            np.array(input_ids, dtype=np.int64),
            np.array(attention_mask, dtype=np.int64)
        )

    def _mean_pooling(
        self,
        embeddings: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply mean pooling to token embeddings.

        Args:
            embeddings: Token embeddings [batch, seq_len, dim]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Pooled embeddings [batch, dim]
        """
        # Expand attention mask to match embedding dimensions
        attention_mask_expanded = np.expand_dims(attention_mask, -1)

        # Sum embeddings, weighted by attention mask
        sum_embeddings = np.sum(embeddings * attention_mask_expanded, axis=1)

        # Sum attention mask to get counts
        sum_mask = np.sum(attention_mask_expanded, axis=1)
        sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)  # Avoid division by zero

        # Mean pooling
        return sum_embeddings / sum_mask

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length.

        Args:
            embeddings: Embedding vectors

        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)  # Avoid division by zero
        return embeddings / norms

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim
```

### Model Conversion Utility
```python
# scripts/convert_model_to_onnx.py
"""
Convert PyTorch embedding models to ONNX format.

Usage:
    python scripts/convert_model_to_onnx.py \\
        --model-name sentence-transformers/all-MiniLM-L6-v2 \\
        --output-dir ./onnx_models \\
        --quantize
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import torch
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_onnx(
    model_name: str,
    output_dir: str,
    quantize: bool = False,
    optimize: bool = True
) -> Path:
    """
    Convert a PyTorch model to ONNX format.

    Args:
        model_name: HuggingFace model name
        output_dir: Output directory for ONNX model
        quantize: Whether to apply quantization
        optimize: Whether to optimize the model

    Returns:
        Path to converted model
    """
    output_path = Path(output_dir) / model_name.replace("/", "_")
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting {model_name} to ONNX...")

    # Load and convert model
    model = ORTModelForFeatureExtraction.from_pretrained(
        model_name,
        export=True,
        provider="CPUExecutionProvider"
    )

    # Save ONNX model
    model.save_pretrained(output_path)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_path)

    logger.info(f"Model saved to {output_path}")

    # Validate conversion
    validate_conversion(model_name, output_path)

    return output_path


def validate_conversion(
    original_model: str,
    onnx_path: Path,
    test_text: str = "This is a test sentence."
) -> bool:
    """
    Validate that ONNX model produces similar embeddings to original.

    Args:
        original_model: Original model name
        onnx_path: Path to ONNX model
        test_text: Test text for validation

    Returns:
        True if validation passes
    """
    logger.info("Validating conversion...")

    # Load original model
    from sentence_transformers import SentenceTransformer
    original = SentenceTransformer(original_model)
    original_embedding = original.encode([test_text])[0]

    # Load ONNX model
    onnx_model = ORTModelForFeatureExtraction.from_pretrained(onnx_path)
    tokenizer = AutoTokenizer.from_pretrained(onnx_path)

    inputs = tokenizer(test_text, return_tensors="pt")
    with torch.no_grad():
        outputs = onnx_model(**inputs)
        onnx_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    # Compare embeddings
    similarity = cosine_similarity(original_embedding, onnx_embedding)
    logger.info(f"Embedding similarity: {similarity:.4f}")

    if similarity > 0.99:
        logger.info("✓ Validation passed!")
        return True
    else:
        logger.warning(f"⚠ Validation warning: similarity {similarity:.4f} < 0.99")
        return False


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    import numpy as np
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    parser = argparse.ArgumentParser(description="Convert models to ONNX")
    parser.add_argument("--model-name", required=True, help="HuggingFace model name")
    parser.add_argument("--output-dir", default="./onnx_models", help="Output directory")
    parser.add_argument("--quantize", action="store_true", help="Apply quantization")
    parser.add_argument("--optimize", action="store_true", default=True, help="Optimize model")

    args = parser.parse_args()

    convert_to_onnx(
        model_name=args.model_name,
        output_dir=args.output_dir,
        quantize=args.quantize,
        optimize=args.optimize
    )


if __name__ == "__main__":
    main()
```

---

## Testing Strategy

### Unit Tests
```python
# tests/unit/services/embeddings/test_onnx_local.py
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from rag_factory.services.embeddings.onnx_local import ONNXLocalEmbedding


class TestONNXLocalEmbedding:
    """Test ONNX local embedding provider."""

    @pytest.fixture
    def mock_session(self):
        """Mock ONNX Runtime session."""
        session = Mock()
        session.get_outputs.return_value = [
            Mock(shape=[1, 384])  # Embedding dimension 384
        ]
        session.get_providers.return_value = ["CPUExecutionProvider"]
        return session

    @pytest.fixture
    def embedding_provider(self, mock_session):
        """Create embedding provider with mocked session."""
        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            with patch.object(ONNXLocalEmbedding, "_download_model"):
                provider = ONNXLocalEmbedding(model_name="test-model")
                provider.session = mock_session
                return provider

    def test_initialization(self, embedding_provider):
        """Test provider initialization."""
        assert embedding_provider.model_name == "test-model"
        assert embedding_provider.embedding_dim == 384

    def test_embed_documents(self, embedding_provider, mock_session):
        """Test document embedding."""
        # Mock session output
        mock_embeddings = np.random.randn(2, 384).astype(np.float32)
        mock_session.run.return_value = [mock_embeddings]

        texts = ["Document 1", "Document 2"]
        embeddings = embedding_provider.embed_documents(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
        assert isinstance(embeddings[0][0], float)

    def test_embed_query(self, embedding_provider, mock_session):
        """Test query embedding."""
        mock_embedding = np.random.randn(1, 384).astype(np.float32)
        mock_session.run.return_value = [mock_embedding]

        embedding = embedding_provider.embed_query("Test query")

        assert len(embedding) == 384
        assert isinstance(embedding[0], float)

    def test_normalization(self, embedding_provider):
        """Test embedding normalization."""
        embeddings = np.array([[3.0, 4.0], [1.0, 0.0]])
        normalized = embedding_provider._normalize(embeddings)

        # Check unit length
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0])

    def test_mean_pooling(self, embedding_provider):
        """Test mean pooling."""
        embeddings = np.array([
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],  # Batch 1
            [[2.0, 3.0], [4.0, 5.0], [0.0, 0.0]]   # Batch 2
        ])
        attention_mask = np.array([
            [1, 1, 0],  # First two tokens valid
            [1, 1, 0]
        ])

        pooled = embedding_provider._mean_pooling(embeddings, attention_mask)

        # Should average first two tokens
        expected = np.array([
            [2.0, 3.0],  # (1+3)/2, (2+4)/2
            [3.0, 4.0]   # (2+4)/2, (3+5)/2
        ])
        np.testing.assert_array_almost_equal(pooled, expected)

    def test_empty_input(self, embedding_provider):
        """Test handling of empty input."""
        embeddings = embedding_provider.embed_documents([])
        assert embeddings == []

    def test_model_download_failure(self):
        """Test handling of model download failure."""
        with patch("huggingface_hub.hf_hub_download", side_effect=Exception("Download failed")):
            with pytest.raises(ValueError, match="Could not download ONNX model"):
                ONNXLocalEmbedding(model_name="nonexistent-model")
```

### Integration Tests
```python
# tests/integration/services/test_onnx_embeddings_integration.py
import pytest
from rag_factory.services.embeddings.onnx_local import ONNXLocalEmbedding


@pytest.mark.integration
class TestONNXEmbeddingsIntegration:
    """Integration tests with real ONNX models."""

    @pytest.fixture(scope="class")
    def embedding_provider(self):
        """Create real ONNX embedding provider."""
        # Use a small, fast model for testing
        return ONNXLocalEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def test_embed_single_document(self, embedding_provider):
        """Test embedding a single document."""
        text = "This is a test document."
        embedding = embedding_provider.embed_query(text)

        assert len(embedding) == embedding_provider.dimension
        assert all(isinstance(x, float) for x in embedding)

        # Check normalization (should be unit vector)
        import numpy as np
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

    def test_embed_multiple_documents(self, embedding_provider):
        """Test embedding multiple documents."""
        texts = [
            "First document about machine learning.",
            "Second document about natural language processing.",
            "Third document about deep learning."
        ]

        embeddings = embedding_provider.embed_documents(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == embedding_provider.dimension for emb in embeddings)

    def test_semantic_similarity(self, embedding_provider):
        """Test that similar texts have similar embeddings."""
        import numpy as np

        text1 = "The cat sits on the mat."
        text2 = "A cat is sitting on a mat."
        text3 = "Python is a programming language."

        emb1 = np.array(embedding_provider.embed_query(text1))
        emb2 = np.array(embedding_provider.embed_query(text2))
        emb3 = np.array(embedding_provider.embed_query(text3))

        # Similar texts should have higher similarity
        sim_12 = np.dot(emb1, emb2)
        sim_13 = np.dot(emb1, emb3)

        assert sim_12 > sim_13
        assert sim_12 > 0.7  # Should be quite similar

    def test_batch_consistency(self, embedding_provider):
        """Test that batch and individual embeddings match."""
        import numpy as np

        texts = ["Test 1", "Test 2", "Test 3"]

        # Batch embedding
        batch_embeddings = embedding_provider.embed_documents(texts)

        # Individual embeddings
        individual_embeddings = [
            embedding_provider.embed_query(text)
            for text in texts
        ]

        # Should be very similar (allowing for small numerical differences)
        for batch_emb, ind_emb in zip(batch_embeddings, individual_embeddings):
            similarity = np.dot(batch_emb, ind_emb)
            assert similarity > 0.99

    def test_performance(self, embedding_provider):
        """Test embedding performance."""
        import time

        text = "This is a performance test document."

        # Warm up
        embedding_provider.embed_query(text)

        # Measure performance
        start = time.time()
        for _ in range(10):
            embedding_provider.embed_query(text)
        elapsed = time.time() - start

        avg_time = elapsed / 10
        assert avg_time < 0.1  # Should be < 100ms per document
```

### Performance Benchmarks
```python
# tests/performance/benchmark_onnx_embeddings.py
import time
import numpy as np
from rag_factory.services.embeddings.onnx_local import ONNXLocalEmbedding


def benchmark_embedding_speed():
    """Benchmark embedding speed."""
    provider = ONNXLocalEmbedding()

    # Test documents of varying lengths
    short_text = "Short document."
    medium_text = "This is a medium length document. " * 10
    long_text = "This is a longer document with more content. " * 50

    texts = {
        "short": short_text,
        "medium": medium_text,
        "long": long_text
    }

    results = {}

    for name, text in texts.items():
        # Warm up
        provider.embed_query(text)

        # Benchmark
        times = []
        for _ in range(100):
            start = time.time()
            provider.embed_query(text)
            times.append(time.time() - start)

        results[name] = {
            "mean": np.mean(times) * 1000,  # ms
            "std": np.std(times) * 1000,
            "p95": np.percentile(times, 95) * 1000,
            "p99": np.percentile(times, 99) * 1000
        }

    print("\\nEmbedding Speed Benchmark:")
    print("-" * 60)
    for name, stats in results.items():
        print(f"{name:10s}: {stats['mean']:.2f}ms ± {stats['std']:.2f}ms "
              f"(p95: {stats['p95']:.2f}ms, p99: {stats['p99']:.2f}ms)")

    # Verify performance targets
    assert results["short"]["p95"] < 100, "Short text p95 should be < 100ms"
    assert results["medium"]["p95"] < 100, "Medium text p95 should be < 100ms"


def benchmark_batch_processing():
    """Benchmark batch processing."""
    provider = ONNXLocalEmbedding()

    batch_sizes = [1, 4, 8, 16, 32]
    text = "This is a test document for batch processing."

    results = {}

    for batch_size in batch_sizes:
        texts = [text] * batch_size

        # Warm up
        provider.embed_documents(texts)

        # Benchmark
        times = []
        for _ in range(50):
            start = time.time()
            provider.embed_documents(texts)
            times.append(time.time() - start)

        results[batch_size] = {
            "total_mean": np.mean(times) * 1000,
            "per_doc": (np.mean(times) / batch_size) * 1000
        }

    print("\\nBatch Processing Benchmark:")
    print("-" * 60)
    for batch_size, stats in results.items():
        print(f"Batch {batch_size:2d}: {stats['total_mean']:.2f}ms total, "
              f"{stats['per_doc']:.2f}ms per document")


if __name__ == "__main__":
    benchmark_embedding_speed()
    benchmark_batch_processing()
```

---

## Documentation

### ONNX Model Usage Guide
```markdown
# Using ONNX Models for Embeddings

## Quick Start

```python
from rag_factory.services.embeddings import ONNXLocalEmbedding

# Initialize with default model
embedder = ONNXLocalEmbedding()

# Embed documents
texts = ["Document 1", "Document 2"]
embeddings = embedder.embed_documents(texts)

# Embed query
query_embedding = embedder.embed_query("What is RAG?")
```

## Supported Models

### Pre-converted ONNX Models
- `sentence-transformers/all-MiniLM-L6-v2` (default)
- `sentence-transformers/all-mpnet-base-v2`
- `BAAI/bge-small-en-v1.5`

### Using Custom Models
```python
# Use local ONNX model
embedder = ONNXLocalEmbedding(
    model_path="/path/to/model.onnx"
)
```

## Performance Tuning

### CPU Optimization
```python
embedder = ONNXLocalEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    intra_op_num_threads=4,  # Parallel ops within layer
    inter_op_num_threads=2   # Parallel layers
)
```

### Memory Management
```python
# For memory-constrained environments
embedder = ONNXLocalEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    enable_cpu_mem_arena=False  # Reduce memory usage
)
```

## Troubleshooting

### Model Not Found
If you get "Could not download ONNX model", the model may not have an ONNX version.
Convert it using our conversion script:

```bash
python scripts/convert_model_to_onnx.py \\
    --model-name your-model-name \\
    --output-dir ./onnx_models
```

### Slow Performance
- Ensure you're using CPU-optimized ONNX runtime
- Adjust thread settings based on your CPU
- Consider using quantized models for faster inference
```

---

## Implementation Plan

### Phase 1: Core ONNX Provider (Days 1-2)
1. Enhance `onnx_local.py` provider
2. Implement model loading from HuggingFace Hub
3. Add model caching
4. Implement tokenization (placeholder for now)
5. Add mean pooling and normalization

### Phase 2: PyTorch Removal (Day 3)
1. Remove torch imports from all embedding providers
2. Replace torch operations with numpy
3. Update requirements.txt
4. Add runtime checks

### Phase 3: Model Conversion (Days 4-5)
1. Create conversion utility script
2. Implement validation
3. Convert common models
4. Document conversion process

### Phase 4: Testing (Days 6-7)
1. Write unit tests
2. Write integration tests
3. Create performance benchmarks
4. Validate quality

### Phase 5: Documentation (Day 8)
1. Write usage guide
2. Create conversion guide
3. Document supported models
4. Add troubleshooting guide

---

## Risks and Mitigation

### Risk: ONNX Model Availability
**Impact:** Medium
**Probability:** Medium
**Mitigation:**
- Provide conversion tools
- Document conversion process
- Pre-convert popular models

### Risk: Tokenization Complexity
**Impact:** High
**Probability:** Medium
**Mitigation:**
- Use tiktoken for OpenAI-compatible models
- Provide fallback tokenization
- Document tokenizer requirements

### Risk: Performance Degradation
**Impact:** Medium
**Probability:** Low
**Mitigation:**
- Benchmark early and often
- Optimize ONNX runtime settings
- Use quantized models if needed

### Risk: Quality Loss
**Impact:** High
**Probability:** Low
**Mitigation:**
- Validate embeddings match PyTorch (>99%)
- Test with multiple models
- Document any limitations

---

## Success Metrics

- [ ] Installation size reduced by >90% (~3GB → ~235MB)
- [ ] Installation time reduced by >90% (20min → <2min)
- [ ] Embedding quality maintained (>99% similarity to PyTorch)
- [ ] Performance targets met (<100ms per document)
- [ ] All tests passing without torch dependency
- [ ] Documentation complete and published
- [ ] At least 3 models converted and validated

---

## Dependencies

**Blocked by:** None
**Blocks:** Stories 10.3, 10.5 (other strategies using embeddings)
**Related:** Story 10.2 (tokenization)
