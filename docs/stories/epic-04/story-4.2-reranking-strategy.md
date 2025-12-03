# Story 4.2: Implement Re-ranking Strategy

**Story ID:** 4.2
**Epic:** Epic 4 - Priority RAG Strategies
**Story Points:** 13
**Priority:** Critical
**Dependencies:** Epic 2 (Vector Database), Epic 3 (Embedding Service)

---

## User Story

**As a** system
**I want** a two-step retrieval with re-ranking
**So that** I can retrieve many chunks but return only the most relevant

---

## Detailed Requirements

### Functional Requirements

1. **Two-Step Retrieval Process**
   - **Step 1 (Broad Retrieval)**: Retrieve large set of candidates (50-100 chunks) using vector similarity
   - **Step 2 (Re-ranking)**: Re-rank candidates using cross-encoder model
   - Return top-k re-ranked results (configurable, 5-10 chunks)
   - Support configurable retrieval and re-ranking sizes

2. **Multi-Model Re-ranker Support**
   - **Cross-Encoder Models**: Sentence-transformers cross-encoders
   - **Cohere Rerank API**: Integration with Cohere's rerank endpoint
   - **BGE Reranker**: Support for BGE reranker models
   - **Custom Rerankers**: Pluggable architecture for custom models
   - Model selection via configuration

3. **Scoring and Ranking**
   - Generate relevance scores for query-document pairs
   - Normalize scores across different models (0.0-1.0)
   - Support score thresholding (filter low-relevance results)
   - Preserve original vector similarity scores for comparison
   - Track ranking changes (position shifts after re-ranking)

4. **Performance Optimization**
   - Batch re-ranking for efficiency
   - Caching of re-ranking results
   - Parallel processing for large candidate sets
   - Configurable batch sizes
   - Early stopping for low-scoring candidates

5. **Ranking Metrics and Logging**
   - Log original vs. re-ranked positions
   - Track score distributions
   - Measure re-ranking impact (NDCG, MRR)
   - Record which candidates were promoted/demoted
   - Performance timing metrics

6. **Fallback Strategies**
   - Graceful degradation if re-ranker fails
   - Fallback to vector similarity ranking
   - Timeout handling for slow re-ranking
   - Error recovery with partial results

### Non-Functional Requirements

1. **Performance**
   - Re-rank 100 candidates in <2 seconds
   - Support concurrent re-ranking requests
   - Batch processing >50 pairs/second
   - Cache hit rate >70% for repeated queries

2. **Quality**
   - Re-ranking should improve relevance over vector search alone
   - Measurable improvement in ranking metrics (NDCG, MRR)
   - Consistent results across different re-ranker models
   - Support A/B testing of different re-rankers

3. **Scalability**
   - Handle 1000+ concurrent re-ranking requests
   - Support re-ranking of up to 500 candidates
   - Horizontal scaling support
   - Efficient memory usage

4. **Reliability**
   - Handle API failures gracefully (Cohere)
   - Automatic retries with exponential backoff
   - Comprehensive error logging
   - Health checks for re-ranker models

5. **Observability**
   - Log all re-ranking decisions
   - Track model performance metrics
   - Monitor API usage and costs
   - Debugging information for ranking analysis

---

## Acceptance Criteria

### AC1: Two-Step Retrieval
- [ ] Broad retrieval retrieves 50-100 candidates (configurable)
- [ ] Re-ranking processes all candidates
- [ ] Top-k results returned after re-ranking (configurable)
- [ ] Original and re-ranked scores preserved
- [ ] Integration with vector database for initial retrieval

### AC2: Multi-Model Support
- [ ] Cross-encoder integration (sentence-transformers)
- [ ] Cohere Rerank API integration
- [ ] BGE reranker support
- [ ] Model selection via configuration
- [ ] Easy to add new re-ranker models

### AC3: Scoring System
- [ ] Relevance scores generated for all pairs
- [ ] Scores normalized to 0.0-1.0 range
- [ ] Score thresholding implemented
- [ ] Original vector scores preserved
- [ ] Ranking position changes tracked

### AC4: Performance Optimization
- [ ] Batch re-ranking implemented
- [ ] Re-ranking cache working
- [ ] Parallel processing for large sets
- [ ] Configurable batch sizes
- [ ] Performance meets <2s for 100 candidates

### AC5: Metrics and Logging
- [ ] Original vs re-ranked positions logged
- [ ] Score distributions tracked
- [ ] NDCG and MRR metrics calculated
- [ ] Promotion/demotion tracking
- [ ] Timing metrics recorded

### AC6: Fallback and Error Handling
- [ ] Fallback to vector ranking on errors
- [ ] Timeout handling implemented
- [ ] Graceful degradation working
- [ ] Partial results returned when needed
- [ ] All errors logged with context

### AC7: Testing
- [ ] Unit tests for all re-ranker implementations (>90% coverage)
- [ ] Integration tests with real models
- [ ] Performance benchmarks meet requirements
- [ ] Quality tests show improvement over baseline
- [ ] A/B testing framework working

---

## Technical Specifications

### File Structure
```
rag_factory/
├── strategies/
│   ├── reranking/
│   │   ├── __init__.py
│   │   ├── base.py                      # Base re-ranker interface
│   │   ├── reranker_service.py          # Main re-ranking service
│   │   ├── cross_encoder_reranker.py    # Cross-encoder implementation
│   │   ├── cohere_reranker.py           # Cohere API integration
│   │   ├── bge_reranker.py              # BGE reranker
│   │   ├── cache.py                     # Re-ranking cache
│   │   ├── config.py                    # Re-ranking configuration
│   │   └── metrics.py                   # Ranking metrics (NDCG, MRR)
│
tests/
├── unit/
│   └── strategies/
│       └── reranking/
│           ├── test_reranker_service.py
│           ├── test_cross_encoder_reranker.py
│           ├── test_cohere_reranker.py
│           ├── test_bge_reranker.py
│           ├── test_cache.py
│           └── test_metrics.py
│
├── integration/
│   └── strategies/
│       └── test_reranking_integration.py
```

### Dependencies
```python
# requirements.txt additions
sentence-transformers==2.3.1    # Cross-encoder models
cohere==4.47                    # Cohere Rerank API
transformers==4.36.0            # For BGE and custom models
torch==2.1.0                    # PyTorch for model inference
numpy==1.24.0                   # Numerical operations
scipy==1.11.0                   # Ranking metrics
```

### Base Re-ranker Interface
```python
# rag_factory/strategies/reranking/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

class RerankerModel(Enum):
    """Enumeration of supported re-ranker models."""
    CROSS_ENCODER = "cross_encoder"
    COHERE = "cohere"
    BGE = "bge"
    CUSTOM = "custom"

@dataclass
class RerankResult:
    """Result from re-ranking operation."""
    document_id: str
    original_rank: int
    reranked_rank: int
    original_score: float  # Vector similarity score
    rerank_score: float    # Re-ranker relevance score
    normalized_score: float  # Normalized score (0.0-1.0)

@dataclass
class RerankResponse:
    """Response from re-ranking service."""
    query: str
    results: List[RerankResult]
    total_candidates: int
    reranked_count: int
    top_k: int
    model_used: str
    execution_time_ms: float
    cache_hit: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RerankConfig:
    """Configuration for re-ranking."""
    model: RerankerModel = RerankerModel.CROSS_ENCODER
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval settings
    initial_retrieval_size: int = 100  # Number of candidates to retrieve
    top_k: int = 10  # Number of results to return after re-ranking

    # Scoring settings
    score_threshold: float = 0.0  # Minimum score to include
    normalize_scores: bool = True

    # Performance settings
    batch_size: int = 32
    enable_cache: bool = True
    cache_ttl: int = 3600  # seconds
    timeout_seconds: float = 5.0

    # Fallback settings
    enable_fallback: bool = True
    fallback_to_vector_scores: bool = True

    # Model-specific config
    model_config: Dict[str, Any] = field(default_factory=dict)

class IReranker(ABC):
    """Abstract base class for re-ranking models."""

    def __init__(self, config: RerankConfig):
        """Initialize re-ranker with configuration."""
        self.config = config

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]] = None
    ) -> List[Tuple[int, float]]:
        """
        Re-rank documents based on query relevance.

        Args:
            query: The search query
            documents: List of document texts to rank
            scores: Optional original scores from vector search

        Returns:
            List of (document_index, relevance_score) tuples, sorted by relevance
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the re-ranker model."""
        pass

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0.0-1.0 range."""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def validate_inputs(self, query: str, documents: List[str]) -> None:
        """Validate inputs to re-ranker."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not documents:
            raise ValueError("Documents list cannot be empty")

        if len(documents) > 500:
            raise ValueError(f"Too many documents: {len(documents)} (max: 500)")
```

### Re-ranker Service
```python
# rag_factory/strategies/reranking/reranker_service.py
from typing import List, Dict, Any, Optional
import time
import logging
from .base import IReranker, RerankConfig, RerankResponse, RerankResult, RerankerModel
from .cross_encoder_reranker import CrossEncoderReranker
from .cohere_reranker import CohereReranker
from .bge_reranker import BGEReranker
from .cache import RerankCache

logger = logging.getLogger(__name__)

@dataclass
class CandidateDocument:
    """Document candidate for re-ranking."""
    id: str
    text: str
    original_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class RerankerService:
    """
    Service for re-ranking retrieved documents using various models.

    Example:
        config = RerankConfig(
            model=RerankerModel.CROSS_ENCODER,
            initial_retrieval_size=100,
            top_k=10
        )
        service = RerankerService(config)
        response = service.rerank(query, candidates)
    """

    def __init__(self, config: RerankConfig):
        self.config = config
        self.reranker = self._init_reranker()
        self.cache = RerankCache(config) if config.enable_cache else None
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_documents_reranked": 0,
            "avg_execution_time_ms": 0.0
        }

    def _init_reranker(self) -> IReranker:
        """Initialize the re-ranker based on configuration."""
        reranker_map = {
            RerankerModel.CROSS_ENCODER: CrossEncoderReranker,
            RerankerModel.COHERE: CohereReranker,
            RerankerModel.BGE: BGEReranker
        }

        reranker_class = reranker_map.get(self.config.model)
        if not reranker_class:
            raise ValueError(f"Unknown re-ranker model: {self.config.model}")

        return reranker_class(self.config)

    def rerank(
        self,
        query: str,
        candidates: List[CandidateDocument]
    ) -> RerankResponse:
        """
        Re-rank candidate documents for a query.

        Args:
            query: The search query
            candidates: List of candidate documents with original scores

        Returns:
            RerankResponse with re-ranked results
        """
        start_time = time.time()
        self._stats["total_requests"] += 1

        # Validate inputs
        if not candidates:
            return self._empty_response(query)

        # Check cache
        cache_hit = False
        if self.cache:
            cache_key = self._compute_cache_key(query, candidates)
            cached_response = self.cache.get(cache_key)

            if cached_response:
                self._stats["cache_hits"] += 1
                cached_response.cache_hit = True
                return cached_response

        self._stats["cache_misses"] += 1

        # Extract document texts and scores
        documents = [c.text for c in candidates]
        original_scores = [c.original_score for c in candidates]

        try:
            # Perform re-ranking
            reranked_indices_scores = self.reranker.rerank(
                query,
                documents,
                original_scores
            )

            # Apply score threshold
            if self.config.score_threshold > 0:
                reranked_indices_scores = [
                    (idx, score) for idx, score in reranked_indices_scores
                    if score >= self.config.score_threshold
                ]

            # Limit to top-k
            reranked_indices_scores = reranked_indices_scores[:self.config.top_k]

            # Normalize scores if configured
            rerank_scores = [score for _, score in reranked_indices_scores]
            if self.config.normalize_scores:
                normalized = self.reranker.normalize_scores(rerank_scores)
            else:
                normalized = rerank_scores

            # Build results
            results = []
            for new_rank, ((orig_idx, rerank_score), norm_score) in enumerate(
                zip(reranked_indices_scores, normalized)
            ):
                candidate = candidates[orig_idx]
                result = RerankResult(
                    document_id=candidate.id,
                    original_rank=orig_idx,
                    reranked_rank=new_rank,
                    original_score=candidate.original_score,
                    rerank_score=rerank_score,
                    normalized_score=norm_score
                )
                results.append(result)

            execution_time_ms = (time.time() - start_time) * 1000

            response = RerankResponse(
                query=query,
                results=results,
                total_candidates=len(candidates),
                reranked_count=len(results),
                top_k=self.config.top_k,
                model_used=self.reranker.get_model_name(),
                execution_time_ms=execution_time_ms,
                cache_hit=cache_hit
            )

            # Cache the response
            if self.cache:
                self.cache.set(cache_key, response)

            # Update stats
            self._stats["total_documents_reranked"] += len(candidates)
            self._update_avg_execution_time(execution_time_ms)

            return response

        except Exception as e:
            logger.error(f"Re-ranking failed: {e}", exc_info=True)

            # Fallback to original ranking if configured
            if self.config.enable_fallback and self.config.fallback_to_vector_scores:
                logger.warning("Falling back to vector similarity scores")
                return self._fallback_ranking(query, candidates, start_time)
            else:
                raise

    def _fallback_ranking(
        self,
        query: str,
        candidates: List[CandidateDocument],
        start_time: float
    ) -> RerankResponse:
        """Fallback to vector similarity ranking."""
        # Sort by original scores
        sorted_candidates = sorted(
            enumerate(candidates),
            key=lambda x: x[1].original_score,
            reverse=True
        )[:self.config.top_k]

        results = []
        for new_rank, (orig_idx, candidate) in enumerate(sorted_candidates):
            result = RerankResult(
                document_id=candidate.id,
                original_rank=orig_idx,
                reranked_rank=new_rank,
                original_score=candidate.original_score,
                rerank_score=candidate.original_score,
                normalized_score=candidate.original_score
            )
            results.append(result)

        execution_time_ms = (time.time() - start_time) * 1000

        return RerankResponse(
            query=query,
            results=results,
            total_candidates=len(candidates),
            reranked_count=len(results),
            top_k=self.config.top_k,
            model_used="fallback_vector_similarity",
            execution_time_ms=execution_time_ms,
            cache_hit=False,
            metadata={"fallback": True}
        )

    def _empty_response(self, query: str) -> RerankResponse:
        """Return empty response for empty candidate list."""
        return RerankResponse(
            query=query,
            results=[],
            total_candidates=0,
            reranked_count=0,
            top_k=self.config.top_k,
            model_used=self.reranker.get_model_name(),
            execution_time_ms=0.0,
            cache_hit=False
        )

    def _compute_cache_key(self, query: str, candidates: List[CandidateDocument]) -> str:
        """Compute cache key from query and candidates."""
        import hashlib

        # Use query + document IDs + model name
        doc_ids = ",".join(sorted([c.id for c in candidates]))
        content = f"{query}:{doc_ids}:{self.reranker.get_model_name()}"

        return hashlib.sha256(content.encode()).hexdigest()

    def _update_avg_execution_time(self, new_time_ms: float):
        """Update average execution time."""
        total = self._stats["total_requests"]
        current_avg = self._stats["avg_execution_time_ms"]

        # Incremental average
        new_avg = ((current_avg * (total - 1)) + new_time_ms) / total
        self._stats["avg_execution_time_ms"] = new_avg

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        cache_hit_rate = 0.0
        if self._stats["total_requests"] > 0:
            cache_hit_rate = self._stats["cache_hits"] / self._stats["total_requests"]

        return {
            **self._stats,
            "cache_hit_rate": cache_hit_rate,
            "model": self.reranker.get_model_name()
        }

    def clear_cache(self):
        """Clear the re-ranking cache."""
        if self.cache:
            self.cache.clear()
```

### Cross-Encoder Implementation
```python
# rag_factory/strategies/reranking/cross_encoder_reranker.py
from typing import List, Tuple, Optional
from sentence_transformers import CrossEncoder
import torch
from .base import IReranker, RerankConfig

class CrossEncoderReranker(IReranker):
    """
    Re-ranker using cross-encoder models from sentence-transformers.
    Cross-encoders jointly encode query and document for better relevance scoring.
    """

    # Popular cross-encoder models
    MODELS = {
        "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-MiniLM-L-12-v2": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "ms-marco-TinyBERT-L-2-v2": "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    }

    def __init__(self, config: RerankConfig):
        super().__init__(config)

        # Get model name
        model_name = config.model_name
        if model_name in self.MODELS:
            model_name = self.MODELS[model_name]

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]] = None
    ) -> List[Tuple[int, float]]:
        """Re-rank documents using cross-encoder."""
        self.validate_inputs(query, documents)

        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]

        # Predict scores in batches
        batch_size = self.config.batch_size
        all_scores = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            batch_scores = self.model.predict(batch, show_progress_bar=False)
            all_scores.extend(batch_scores.tolist())

        # Create (index, score) pairs and sort by score descending
        indexed_scores = list(enumerate(all_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name
```

### Cohere Re-ranker Implementation
```python
# rag_factory/strategies/reranking/cohere_reranker.py
from typing import List, Tuple, Optional
import cohere
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import IReranker, RerankConfig

class CohereReranker(IReranker):
    """
    Re-ranker using Cohere's Rerank API.
    Provides state-of-the-art re-ranking with minimal setup.
    """

    MODELS = {
        "rerank-english-v2.0": "rerank-english-v2.0",
        "rerank-multilingual-v2.0": "rerank-multilingual-v2.0"
    }

    def __init__(self, config: RerankConfig):
        super().__init__(config)

        # Get API key from config
        api_key = config.model_config.get("api_key")
        if not api_key:
            raise ValueError("Cohere API key required in model_config")

        self.client = cohere.Client(api_key)
        self.model_name = config.model_name or "rerank-english-v2.0"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def rerank(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]] = None
    ) -> List[Tuple[int, float]]:
        """Re-rank documents using Cohere Rerank API."""
        self.validate_inputs(query, documents)

        # Call Cohere Rerank API
        response = self.client.rerank(
            model=self.model_name,
            query=query,
            documents=documents,
            top_n=len(documents),  # Return all, we'll filter later
            return_documents=False
        )

        # Extract results
        results = []
        for result in response.results:
            results.append((result.index, result.relevance_score))

        # Results are already sorted by relevance
        return results

    def get_model_name(self) -> str:
        """Get the model name."""
        return f"cohere:{self.model_name}"
```

---

## Unit Tests

### Test File Location
`tests/unit/strategies/reranking/test_reranker_service.py`
`tests/unit/strategies/reranking/test_cross_encoder_reranker.py`

### Test Cases

#### TC4.2.1: Reranker Service Tests
```python
import pytest
from unittest.mock import Mock, MagicMock
from rag_factory.strategies.reranking.reranker_service import RerankerService, CandidateDocument
from rag_factory.strategies.reranking.base import RerankConfig, RerankerModel

@pytest.fixture
def rerank_config():
    return RerankConfig(
        model=RerankerModel.CROSS_ENCODER,
        initial_retrieval_size=100,
        top_k=10,
        enable_cache=True
    )

@pytest.fixture
def mock_reranker():
    reranker = Mock()
    reranker.rerank.return_value = [
        (2, 0.95),  # doc 2 has highest score
        (0, 0.85),
        (1, 0.75)
    ]
    reranker.get_model_name.return_value = "mock-reranker"
    reranker.normalize_scores.return_value = [1.0, 0.5, 0.0]
    return reranker

def test_service_initialization(rerank_config):
    """Test service initializes correctly."""
    service = RerankerService(rerank_config)
    assert service.config == rerank_config
    assert service.reranker is not None

def test_rerank_basic(rerank_config, mock_reranker, monkeypatch):
    """Test basic re-ranking."""
    service = RerankerService(rerank_config)
    monkeypatch.setattr(service, "reranker", mock_reranker)

    candidates = [
        CandidateDocument(id="doc1", text="Document 1", original_score=0.9),
        CandidateDocument(id="doc2", text="Document 2", original_score=0.8),
        CandidateDocument(id="doc3", text="Document 3", original_score=0.7)
    ]

    response = service.rerank("test query", candidates)

    assert response.total_candidates == 3
    assert response.reranked_count == 3
    assert len(response.results) == 3

    # Check that doc2 is now ranked first (highest rerank score)
    assert response.results[0].document_id == "doc3"  # index 2
    assert response.results[0].reranked_rank == 0

def test_rerank_empty_candidates(rerank_config):
    """Test re-ranking with empty candidates list."""
    service = RerankerService(rerank_config)

    response = service.rerank("test query", [])

    assert response.total_candidates == 0
    assert response.reranked_count == 0
    assert len(response.results) == 0

def test_rerank_top_k_limit(rerank_config, mock_reranker, monkeypatch):
    """Test that top_k limits results."""
    config = RerankConfig(
        model=RerankerModel.CROSS_ENCODER,
        top_k=2  # Only return top 2
    )
    service = RerankerService(config)
    monkeypatch.setattr(service, "reranker", mock_reranker)

    candidates = [
        CandidateDocument(id=f"doc{i}", text=f"Document {i}", original_score=0.9 - i*0.1)
        for i in range(5)
    ]

    response = service.rerank("test query", candidates)

    assert response.total_candidates == 5
    assert response.reranked_count <= 2
    assert len(response.results) <= 2

def test_cache_hit(rerank_config, mock_reranker, monkeypatch):
    """Test that cache returns cached results."""
    service = RerankerService(rerank_config)
    monkeypatch.setattr(service, "reranker", mock_reranker)

    candidates = [
        CandidateDocument(id="doc1", text="Document 1", original_score=0.9)
    ]

    # First call - cache miss
    response1 = service.rerank("test query", candidates)
    assert response1.cache_hit == False

    # Second call - should hit cache
    response2 = service.rerank("test query", candidates)
    assert response2.cache_hit == True

    # Reranker should only be called once
    assert mock_reranker.rerank.call_count == 1

def test_cache_disabled(mock_reranker, monkeypatch):
    """Test service with cache disabled."""
    config = RerankConfig(
        model=RerankerModel.CROSS_ENCODER,
        enable_cache=False
    )
    service = RerankerService(config)
    monkeypatch.setattr(service, "reranker", mock_reranker)

    candidates = [
        CandidateDocument(id="doc1", text="Document 1", original_score=0.9)
    ]

    service.rerank("query", candidates)
    service.rerank("query", candidates)

    # Should call reranker twice (no caching)
    assert mock_reranker.rerank.call_count == 2

def test_score_threshold(rerank_config, monkeypatch):
    """Test score threshold filtering."""
    config = RerankConfig(
        model=RerankerModel.CROSS_ENCODER,
        score_threshold=0.8  # Filter scores below 0.8
    )
    service = RerankerService(config)

    mock_reranker = Mock()
    mock_reranker.rerank.return_value = [
        (0, 0.95),  # Above threshold
        (1, 0.85),  # Above threshold
        (2, 0.75)   # Below threshold - should be filtered
    ]
    mock_reranker.get_model_name.return_value = "mock"
    mock_reranker.normalize_scores.side_effect = lambda x: x

    monkeypatch.setattr(service, "reranker", mock_reranker)

    candidates = [
        CandidateDocument(id=f"doc{i}", text=f"Doc {i}", original_score=0.9)
        for i in range(3)
    ]

    response = service.rerank("query", candidates)

    # Should only return 2 results (above threshold)
    assert response.reranked_count == 2

def test_fallback_on_error(rerank_config, monkeypatch):
    """Test fallback to vector scores on error."""
    service = RerankerService(rerank_config)

    mock_reranker = Mock()
    mock_reranker.rerank.side_effect = Exception("Reranker failed")
    mock_reranker.get_model_name.return_value = "mock"

    monkeypatch.setattr(service, "reranker", mock_reranker)

    candidates = [
        CandidateDocument(id="doc1", text="Doc 1", original_score=0.9),
        CandidateDocument(id="doc2", text="Doc 2", original_score=0.8),
        CandidateDocument(id="doc3", text="Doc 3", original_score=0.7)
    ]

    response = service.rerank("query", candidates)

    # Should fallback successfully
    assert response.model_used == "fallback_vector_similarity"
    assert len(response.results) > 0
    assert response.metadata.get("fallback") == True

def test_get_stats(rerank_config, mock_reranker, monkeypatch):
    """Test statistics tracking."""
    service = RerankerService(rerank_config)
    monkeypatch.setattr(service, "reranker", mock_reranker)

    candidates = [
        CandidateDocument(id="doc1", text="Doc 1", original_score=0.9)
    ]

    service.rerank("query1", candidates)
    service.rerank("query1", candidates)  # Cache hit
    service.rerank("query2", candidates)  # Cache miss

    stats = service.get_stats()

    assert stats["total_requests"] == 3
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 2
    assert stats["cache_hit_rate"] == pytest.approx(0.333, 0.01)
```

#### TC4.2.2: Cross-Encoder Tests
```python
import pytest
from unittest.mock import Mock, patch
from rag_factory.strategies.reranking.cross_encoder_reranker import CrossEncoderReranker
from rag_factory.strategies.reranking.base import RerankConfig

@pytest.fixture
def rerank_config():
    return RerankConfig(
        model_name="ms-marco-MiniLM-L-6-v2",
        batch_size=32
    )

@patch('rag_factory.strategies.reranking.cross_encoder_reranker.CrossEncoder')
def test_cross_encoder_initialization(mock_cross_encoder, rerank_config):
    """Test cross-encoder initializes correctly."""
    reranker = CrossEncoderReranker(rerank_config)

    assert reranker.config == rerank_config
    mock_cross_encoder.assert_called_once()

def test_validate_inputs(rerank_config):
    """Test input validation."""
    with patch('rag_factory.strategies.reranking.cross_encoder_reranker.CrossEncoder'):
        reranker = CrossEncoderReranker(rerank_config)

        # Empty query should raise error
        with pytest.raises(ValueError, match="Query cannot be empty"):
            reranker.validate_inputs("", ["doc"])

        # Empty documents should raise error
        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            reranker.validate_inputs("query", [])

        # Too many documents should raise error
        with pytest.raises(ValueError, match="Too many documents"):
            reranker.validate_inputs("query", ["doc"] * 501)

@patch('rag_factory.strategies.reranking.cross_encoder_reranker.CrossEncoder')
def test_rerank_basic(mock_cross_encoder_class, rerank_config):
    """Test basic re-ranking."""
    # Mock the cross-encoder model
    mock_model = Mock()
    mock_model.predict.return_value = Mock()
    mock_model.predict.return_value.tolist.return_value = [0.95, 0.85, 0.75]
    mock_cross_encoder_class.return_value = mock_model

    reranker = CrossEncoderReranker(rerank_config)

    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of AI",
        "Python is a programming language",
        "ML uses algorithms to learn from data"
    ]

    results = reranker.rerank(query, documents)

    # Should return sorted results
    assert len(results) == 3

    # First result should have highest score
    assert results[0][1] > results[1][1]
    assert results[1][1] > results[2][1]

    # Model predict should be called
    mock_model.predict.assert_called_once()

def test_normalize_scores(rerank_config):
    """Test score normalization."""
    with patch('rag_factory.strategies.reranking.cross_encoder_reranker.CrossEncoder'):
        reranker = CrossEncoderReranker(rerank_config)

        scores = [0.5, 0.75, 1.0, 0.25]
        normalized = reranker.normalize_scores(scores)

        # Should be in 0-1 range
        assert all(0.0 <= s <= 1.0 for s in normalized)

        # Min should be 0, max should be 1
        assert min(normalized) == 0.0
        assert max(normalized) == 1.0

def test_normalize_scores_all_same(rerank_config):
    """Test normalization when all scores are the same."""
    with patch('rag_factory.strategies.reranking.cross_encoder_reranker.CrossEncoder'):
        reranker = CrossEncoderReranker(rerank_config)

        scores = [0.8, 0.8, 0.8]
        normalized = reranker.normalize_scores(scores)

        # All should be 1.0 when identical
        assert all(s == 1.0 for s in normalized)
```

---

## Integration Tests

### Test File Location
`tests/integration/strategies/test_reranking_integration.py`

### Test Scenarios

#### IS4.2.1: End-to-End Re-ranking
```python
import pytest
from rag_factory.strategies.reranking.reranker_service import RerankerService, CandidateDocument
from rag_factory.strategies.reranking.cross_encoder_reranker import CrossEncoderReranker
from rag_factory.strategies.reranking.base import RerankConfig, RerankerModel

@pytest.mark.integration
def test_cross_encoder_reranking_real_model():
    """Test re-ranking with real cross-encoder model."""
    config = RerankConfig(
        model=RerankerModel.CROSS_ENCODER,
        model_name="ms-marco-MiniLM-L-6-v2",
        top_k=3
    )

    service = RerankerService(config)

    query = "What is the capital of France?"

    candidates = [
        CandidateDocument(
            id="doc1",
            text="Paris is the capital and largest city of France.",
            original_score=0.7
        ),
        CandidateDocument(
            id="doc2",
            text="London is the capital of the United Kingdom.",
            original_score=0.9  # Higher vector score but less relevant
        ),
        CandidateDocument(
            id="doc3",
            text="Berlin is the capital of Germany.",
            original_score=0.8
        ),
        CandidateDocument(
            id="doc4",
            text="France is a country in Western Europe.",
            original_score=0.6
        )
    ]

    response = service.rerank(query, candidates)

    # Should re-rank correctly
    assert response.total_candidates == 4
    assert response.reranked_count <= 3  # top_k

    # doc1 (about Paris) should be ranked first despite lower vector score
    assert response.results[0].document_id == "doc1"

    print(f"\nRe-ranking results:")
    for result in response.results:
        print(f"  Rank {result.reranked_rank}: {result.document_id}")
        print(f"    Original score: {result.original_score:.3f}")
        print(f"    Rerank score: {result.rerank_score:.3f}")

@pytest.mark.integration
def test_reranking_improves_relevance():
    """Test that re-ranking improves relevance over vector search."""
    config = RerankConfig(
        model=RerankerModel.CROSS_ENCODER,
        top_k=5
    )

    service = RerankerService(config)

    query = "machine learning algorithms"

    candidates = [
        CandidateDocument(
            id="relevant1",
            text="Machine learning algorithms include decision trees, neural networks, and support vector machines.",
            original_score=0.6
        ),
        CandidateDocument(
            id="irrelevant1",
            text="The weather today is sunny and warm.",
            original_score=0.9  # High vector score but irrelevant
        ),
        CandidateDocument(
            id="relevant2",
            text="Supervised learning algorithms learn from labeled data.",
            original_score=0.7
        ),
        CandidateDocument(
            id="irrelevant2",
            text="Cooking recipes for beginners.",
            original_score=0.8
        ),
        CandidateDocument(
            id="relevant3",
            text="Deep learning is a subset of machine learning.",
            original_score=0.65
        )
    ]

    response = service.rerank(query, candidates)

    # Relevant documents should be ranked higher after re-ranking
    top_3_ids = [r.document_id for r in response.results[:3]]

    relevant_in_top_3 = sum(1 for doc_id in top_3_ids if "relevant" in doc_id)

    assert relevant_in_top_3 >= 2, "Re-ranking should promote relevant documents"

@pytest.mark.integration
def test_performance_benchmark():
    """Benchmark re-ranking performance."""
    import time

    config = RerankConfig(
        model=RerankerModel.CROSS_ENCODER,
        top_k=10
    )

    service = RerankerService(config)

    query = "artificial intelligence applications"

    # Create 100 candidates
    candidates = [
        CandidateDocument(
            id=f"doc{i}",
            text=f"Document {i} about various topics.",
            original_score=0.9 - (i * 0.001)
        )
        for i in range(100)
    ]

    start = time.time()
    response = service.rerank(query, candidates)
    duration = time.time() - start

    print(f"\nRe-ranked {response.total_candidates} candidates in {duration:.3f}s")

    # Should meet performance requirement (<2 seconds for 100 candidates)
    assert duration < 2.0, f"Re-ranking took {duration:.3f}s (expected <2s)"

    assert response.execution_time_ms < 2000
```

---

## Definition of Done

- [ ] Base re-ranker interface defined
- [ ] Re-ranker service implemented
- [ ] Cross-encoder re-ranker implemented
- [ ] Cohere re-ranker implemented
- [ ] BGE re-ranker implemented
- [ ] Two-step retrieval working
- [ ] Scoring and ranking system implemented
- [ ] Cache implementation complete
- [ ] Fallback strategies working
- [ ] All unit tests pass (>90% coverage)
- [ ] All integration tests pass
- [ ] Performance benchmarks meet <2s requirement
- [ ] Quality tests show improvement
- [ ] Metrics (NDCG, MRR) implemented
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] No linting errors

---

## Notes for Developers

1. **Model Selection**: Start with cross-encoder/ms-marco-MiniLM-L-6-v2 for balance of speed and quality.

2. **Retrieval Size**: Retrieve 50-100 candidates. More candidates = better recall but slower re-ranking.

3. **Batch Size**: Tune batch size based on GPU memory. Larger batches = faster but more memory.

4. **Caching**: Enable caching in production. Re-ranking is expensive.

5. **Cohere API**: Great quality but costs money. Monitor usage.

6. **Fallback**: Always enable fallback to handle errors gracefully.

7. **Testing**: Test with real queries to validate improvement over vector search.

8. **Metrics**: Track NDCG and MRR to measure re-ranking effectiveness.

9. **GPU**: Use GPU for cross-encoders when available. Much faster than CPU.

10. **A/B Testing**: Compare different re-ranker models to find best for your use case.
