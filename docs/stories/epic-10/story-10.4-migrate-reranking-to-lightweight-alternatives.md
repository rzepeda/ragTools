# Story 10.4: Migrate Reranking to Lightweight Alternatives

**Story ID:** 10.4
**Epic:** Epic 10 - Lightweight Dependencies Implementation
**Story Points:** 8
**Priority:** High
**Dependencies:** Story 10.1 (ONNX embeddings)

---

## User Story

**As a** developer
**I want** reranking without PyTorch dependencies
**So that** I can use reranking in lightweight deployments

---

## Detailed Requirements

### Functional Requirements

1. **Cohere Reranking (Primary)**
   - Implement Cohere reranking as primary option
   - Support Cohere's rerank API
   - Handle API authentication and configuration
   - Implement batch reranking
   - Add error handling and retries
   - Track API usage and costs

2. **Cosine Similarity Reranker**
   - Implement lightweight cosine similarity reranker
   - Use numpy for vector operations (no PyTorch)
   - Support different similarity metrics (cosine, dot product, euclidean)
   - Implement efficient batch processing
   - Provide normalization options
   - Add score calibration

3. **Optional PyTorch Rerankers**
   - Make BGE reranker optional (requires torch)
   - Make Cross-Encoder reranker optional (requires torch)
   - Add runtime dependency checks
   - Provide helpful error messages when unavailable
   - Document installation for optional rerankers
   - Ensure graceful degradation

4. **Auto-Selection Logic**
   - Automatically select available reranker
   - Priority: Cohere > Cosine 
   - Allow manual reranker selection
   - Provide reranker capability detection
   - Log selected reranker
   - Handle fallback scenarios

5. **Reranking Strategy Updates**
   - Update reranking strategy to use new rerankers
   - Maintain all existing functionality
   - Support multiple reranking passes
   - Preserve score normalization
   - Ensure backward compatibility
   - Update configuration options

6. **Quality and Performance**
   - Validate reranking quality
   - Benchmark performance
   - Document quality trade-offs
   - Provide tuning recommendations

### Non-Functional Requirements

1. **Performance**
   - Cohere API: <200ms per batch (network dependent)
   - Cosine similarity: <10ms per batch (100 documents)
   - Batch processing: support up to 100 documents
   - Memory efficient: <100MB overhead

2. **Reliability**
   - Handle API failures gracefully
   - Implement retry logic for API calls
   - Fallback to cosine if API unavailable
   - Validate inputs before processing
   - Comprehensive error handling

3. **Cost Efficiency**
   - Track Cohere API usage
   - Implement caching where appropriate
   - Batch requests efficiently
   - Provide cost estimation
   - Document pricing implications

4. **Compatibility**
   - Works without PyTorch
   - Compatible with all embedding providers
   - Supports all RAG strategies
   - Backward compatible configuration

5. **Resource Efficiency**
   - No PyTorch dependency (~2.5GB saved)
   - Minimal additional dependencies
   - Fast initialization (<100ms)
   - Low memory footprint

---

## Acceptance Criteria

### AC1: Cohere Reranking
- [ ] Cohere reranker implemented
- [ ] API authentication working
- [ ] Batch reranking supported
- [ ] Error handling and retries working
- [ ] Usage tracking implemented
- [ ] Documentation complete

### AC2: Cosine Similarity Reranker
- [ ] Cosine similarity reranker implemented
- [ ] Multiple similarity metrics supported
- [ ] Batch processing efficient
- [ ] Score normalization working
- [ ] Performance targets met (<10ms)
- [ ] Quality validated

### AC3: Optional PyTorch Rerankers
- [ ] BGE reranker made optional
- [ ] Cross-Encoder reranker made optional
- [ ] Runtime checks implemented
- [ ] Helpful error messages provided
- [ ] Installation documented
- [ ] Graceful degradation working

### AC4: Auto-Selection
- [ ] Auto-selection logic implemented
- [ ] Correct priority order (Cohere > Cosine > PyTorch)
- [ ] Manual selection supported
- [ ] Capability detection working
- [ ] Selection logged
- [ ] Fallback scenarios handled

### AC5: Strategy Updates
- [ ] Reranking strategy updated
- [ ] All features working
- [ ] Multiple reranking passes supported
- [ ] Score normalization preserved
- [ ] Backward compatibility maintained
- [ ] Configuration updated

### AC6: Quality and Performance
- [ ] Quality validated vs PyTorch rerankers
- [ ] Performance benchmarked
- [ ] Comparison documented
- [ ] Trade-offs documented
- [ ] Tuning guide provided

### AC7: Testing
- [ ] Unit tests for all rerankers
- [ ] Integration tests passing
- [ ] Performance benchmarks created
- [ ] Quality validation tests passing
- [ ] All tests passing without PyTorch

---

## Technical Specifications

### File Structure
```
rag_factory/
├── services/
│   ├── reranking/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── cohere_reranker.py        # NEW: Primary option
│   │   ├── cosine_reranker.py        # NEW: Lightweight fallback
│   │   ├── bge_reranker.py           # UPDATED: Optional
│   │   ├── cross_encoder_reranker.py # UPDATED: Optional
│   │   └── config.py
│   │
│   └── utils/
│       └── reranker_selector.py      # NEW: Auto-selection logic

tests/
├── unit/
│   └── services/
│       └── reranking/
│           ├── test_cohere_reranker.py
│           ├── test_cosine_reranker.py
│           ├── test_reranker_selector.py
│           └── test_optional_rerankers.py
│
└── integration/
    └── services/
        └── test_reranking_integration.py
```

### Dependencies
```python
# requirements.txt - Lightweight dependencies
cohere>=4.47                    # Cohere API client
numpy>=1.24.0                   # Vector operations

# Optional dependencies (in requirements-optional.txt)
# torch>=2.1.2                  # For BGE, Cross-Encoder (optional)
# sentence-transformers>=2.2.0  # For Cross-Encoder (optional)
```

### Cohere Reranker
```python
# rag_factory/services/reranking/cohere_reranker.py
from typing import List, Dict, Any, Optional
import logging
from cohere import Client
from .base import Reranker, RerankResult

logger = logging.getLogger(__name__)


class CohereReranker(Reranker):
    """
    Reranker using Cohere's rerank API.
    High quality, no local dependencies, API-based.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-english-v3.0",
        top_n: Optional[int] = None,
        max_chunks_per_doc: Optional[int] = None
    ):
        """
        Initialize Cohere reranker.

        Args:
            api_key: Cohere API key (or set COHERE_API_KEY env var)
            model: Rerank model name
            top_n: Return top N results (None = all)
            max_chunks_per_doc: Max chunks per document
        """
        super().__init__()
        self.client = Client(api_key=api_key)
        self.model = model
        self.top_n = top_n
        self.max_chunks_per_doc = max_chunks_per_doc

        logger.info(f"Initialized CohereReranker with model: {model}")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> List[RerankResult]:
        """
        Rerank documents using Cohere API.

        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of results to return

        Returns:
            List of reranked results with scores
        """
        if not documents:
            return []

        try:
            # Call Cohere rerank API
            response = self.client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_n=top_k,
                max_chunks_per_doc=self.max_chunks_per_doc
            )

            # Convert to RerankResult
            results = []
            for result in response.results:
                results.append(RerankResult(
                    index=result.index,
                    document=documents[result.index],
                    score=result.relevance_score,
                    metadata={"model": self.model}
                ))

            logger.info(
                f"Reranked {len(documents)} documents, "
                f"returned top {len(results)}"
            )

            return results

        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            raise

    @property
    def name(self) -> str:
        """Get reranker name."""
        return f"cohere-{self.model}"

    @property
    def requires_api_key(self) -> bool:
        """Whether this reranker requires an API key."""
        return True
```

### Cosine Similarity Reranker
```python
# rag_factory/services/reranking/cosine_reranker.py
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from .base import Reranker, RerankResult

logger = logging.getLogger(__name__)


class CosineReranker(Reranker):
    """
    Lightweight reranker using cosine similarity.
    No external dependencies, fast, works offline.
    """

    def __init__(
        self,
        embedding_provider: Any,
        metric: str = "cosine",
        normalize: bool = True
    ):
        """
        Initialize cosine reranker.

        Args:
            embedding_provider: Provider for generating embeddings
            metric: Similarity metric ("cosine", "dot", "euclidean")
            normalize: Whether to normalize embeddings
        """
        super().__init__()
        self.embedding_provider = embedding_provider
        self.metric = metric
        self.normalize = normalize

        logger.info(f"Initialized CosineReranker with metric: {metric}")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> List[RerankResult]:
        """
        Rerank documents using cosine similarity.

        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of results to return

        Returns:
            List of reranked results with scores
        """
        if not documents:
            return []

        # Embed query
        query_embedding = np.array(
            self.embedding_provider.embed_query(query)
        )

        # Embed documents
        doc_embeddings = np.array(
            self.embedding_provider.embed_documents(documents)
        )

        # Normalize if requested
        if self.normalize:
            query_embedding = self._normalize(query_embedding)
            doc_embeddings = self._normalize_batch(doc_embeddings)

        # Calculate similarities
        if self.metric == "cosine":
            scores = self._cosine_similarity(query_embedding, doc_embeddings)
        elif self.metric == "dot":
            scores = self._dot_product(query_embedding, doc_embeddings)
        elif self.metric == "euclidean":
            scores = self._euclidean_similarity(query_embedding, doc_embeddings)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Sort by score
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        # Create results
        results = []
        for idx in ranked_indices:
            results.append(RerankResult(
                index=int(idx),
                document=documents[idx],
                score=float(scores[idx]),
                metadata={"metric": self.metric}
            ))

        logger.info(
            f"Reranked {len(documents)} documents using {self.metric}, "
            f"returned top {len(results)}"
        )

        return results

    def _cosine_similarity(
        self,
        query: np.ndarray,
        documents: np.ndarray
    ) -> np.ndarray:
        """Calculate cosine similarity."""
        # Assuming normalized vectors, cosine = dot product
        return np.dot(documents, query)

    def _dot_product(
        self,
        query: np.ndarray,
        documents: np.ndarray
    ) -> np.ndarray:
        """Calculate dot product."""
        return np.dot(documents, query)

    def _euclidean_similarity(
        self,
        query: np.ndarray,
        documents: np.ndarray
    ) -> np.ndarray:
        """Calculate euclidean similarity (inverse distance)."""
        distances = np.linalg.norm(documents - query, axis=1)
        # Convert distance to similarity (smaller distance = higher similarity)
        return 1.0 / (1.0 + distances)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a single vector."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _normalize_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize a batch of vectors."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)  # Avoid division by zero
        return vectors / norms

    @property
    def name(self) -> str:
        """Get reranker name."""
        return f"cosine-{self.metric}"

    @property
    def requires_api_key(self) -> bool:
        """Whether this reranker requires an API key."""
        return False
```

### Reranker Selector
```python
# rag_factory/services/utils/reranker_selector.py
from typing import Optional, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)


class RerankerSelector:
    """
    Automatically select the best available reranker.
    Priority: Cohere > Cosine > PyTorch-based (if available)
    """

    @staticmethod
    def select_reranker(
        embedding_provider: Any,
        config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Select the best available reranker.

        Args:
            embedding_provider: Embedding provider for cosine reranker
            config: Configuration options

        Returns:
            Reranker instance
        """
        config = config or {}

        # Check for manual selection
        if "reranker_type" in config:
            return RerankerSelector._create_reranker(
                config["reranker_type"],
                embedding_provider,
                config
            )

        # Auto-select based on availability
        # 1. Try Cohere (if API key available)
        if RerankerSelector._is_cohere_available():
            logger.info("Selected Cohere reranker (API-based)")
            from rag_factory.services.reranking.cohere_reranker import CohereReranker
            return CohereReranker(
                api_key=config.get("cohere_api_key"),
                model=config.get("cohere_model", "rerank-english-v3.0")
            )

        # 2. Fall back to Cosine similarity
        logger.info("Selected Cosine similarity reranker (lightweight)")
        from rag_factory.services.reranking.cosine_reranker import CosineReranker
        return CosineReranker(
            embedding_provider=embedding_provider,
            metric=config.get("similarity_metric", "cosine")
        )

    @staticmethod
    def _is_cohere_available() -> bool:
        """Check if Cohere API is available."""
        # Check for API key
        api_key = os.getenv("COHERE_API_KEY")
        if api_key:
            return True

        # Check if cohere package is installed
        try:
            import cohere
            return True
        except ImportError:
            return False

    @staticmethod
    def _is_torch_available() -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False

    @staticmethod
    def _create_reranker(
        reranker_type: str,
        embedding_provider: Any,
        config: Dict[str, Any]
    ) -> Any:
        """Create specific reranker type."""
        if reranker_type == "cohere":
            from rag_factory.services.reranking.cohere_reranker import CohereReranker
            return CohereReranker(
                api_key=config.get("cohere_api_key"),
                model=config.get("cohere_model", "rerank-english-v3.0")
            )

        elif reranker_type == "cosine":
            from rag_factory.services.reranking.cosine_reranker import CosineReranker
            return CosineReranker(
                embedding_provider=embedding_provider,
                metric=config.get("similarity_metric", "cosine")
            )

        elif reranker_type == "bge":
            if not RerankerSelector._is_torch_available():
                raise ImportError(
                    "BGE reranker requires PyTorch. "
                    "Install with: pip install torch sentence-transformers"
                )
            from rag_factory.services.reranking.bge_reranker import BGEReranker
            return BGEReranker(model_name=config.get("bge_model"))

        elif reranker_type == "cross-encoder":
            if not RerankerSelector._is_torch_available():
                raise ImportError(
                    "Cross-Encoder reranker requires PyTorch. "
                    "Install with: pip install torch sentence-transformers"
                )
            from rag_factory.services.reranking.cross_encoder_reranker import CrossEncoderReranker
            return CrossEncoderReranker(model_name=config.get("cross_encoder_model"))

        else:
            raise ValueError(f"Unknown reranker type: {reranker_type}")

    @staticmethod
    def get_available_rerankers() -> Dict[str, bool]:
        """Get dict of available rerankers."""
        return {
            "cohere": RerankerSelector._is_cohere_available(),
            "cosine": True,  # Always available
            "bge": RerankerSelector._is_torch_available(),
            "cross-encoder": RerankerSelector._is_torch_available()
        }
```

---

## Testing Strategy

### Unit Tests
```python
# tests/unit/services/reranking/test_cosine_reranker.py
import pytest
import numpy as np
from unittest.mock import Mock
from rag_factory.services.reranking.cosine_reranker import CosineReranker


class TestCosineReranker:
    """Test cosine similarity reranker."""

    @pytest.fixture
    def mock_embedder(self):
        """Mock embedding provider."""
        embedder = Mock()
        embedder.embed_query.return_value = [1.0, 0.0, 0.0]
        embedder.embed_documents.return_value = [
            [1.0, 0.0, 0.0],  # Perfect match
            [0.5, 0.5, 0.0],  # Partial match
            [0.0, 1.0, 0.0]   # Orthogonal
        ]
        return embedder

    def test_rerank_cosine(self, mock_embedder):
        """Test cosine similarity reranking."""
        reranker = CosineReranker(mock_embedder, metric="cosine")

        documents = ["Doc 1", "Doc 2", "Doc 3"]
        results = reranker.rerank("query", documents, top_k=3)

        assert len(results) == 3
        # First result should be perfect match
        assert results[0].index == 0
        assert results[0].score > results[1].score

    def test_rerank_top_k(self, mock_embedder):
        """Test top_k limiting."""
        reranker = CosineReranker(mock_embedder)

        documents = ["Doc 1", "Doc 2", "Doc 3"]
        results = reranker.rerank("query", documents, top_k=2)

        assert len(results) == 2

    def test_normalization(self, mock_embedder):
        """Test vector normalization."""
        reranker = CosineReranker(mock_embedder, normalize=True)

        vector = np.array([3.0, 4.0])  # Length 5
        normalized = reranker._normalize(vector)

        # Should be unit length
        assert abs(np.linalg.norm(normalized) - 1.0) < 1e-6

    def test_empty_documents(self, mock_embedder):
        """Test handling of empty document list."""
        reranker = CosineReranker(mock_embedder)

        results = reranker.rerank("query", [], top_k=10)

        assert results == []
```

---

## Implementation Plan

### Phase 1: Cohere Reranker (Days 1-2)
1. Implement Cohere reranker
2. Add API authentication
3. Implement batch processing
4. Add error handling
5. Test and validate

### Phase 2: Cosine Reranker (Days 2-3)
1. Implement cosine similarity reranker
2. Add multiple metrics
3. Optimize performance
4. Test and validate

### Phase 3: Optional Rerankers (Days 4-5)
1. Make BGE reranker optional
2. Make Cross-Encoder optional
3. Add runtime checks
4. Update documentation

### Phase 4: Auto-Selection (Day 6)
1. Implement reranker selector
2. Add capability detection
3. Implement fallback logic
4. Test selection logic

### Phase 5: Integration and Testing (Days 7-8)
1. Update reranking strategy
2. Integration testing
3. Quality validation
4. Performance benchmarks
5. Documentation

---

## Success Metrics

- [ ] Cohere reranker working
- [ ] Cosine reranker working
- [ ] No PyTorch required for basic reranking
- [ ] Auto-selection working correctly
- [ ] Performance targets met
- [ ] Quality comparable to PyTorch rerankers
- [ ] All tests passing

---

## Dependencies

**Blocked by:** Story 10.1 (ONNX embeddings)
**Blocks:** None
**Related:** Epic 4 (reranking strategies)
