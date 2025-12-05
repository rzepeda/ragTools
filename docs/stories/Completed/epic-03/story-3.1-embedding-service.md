# Story 3.1: Build Embedding Service

**Story ID:** 3.1
**Epic:** Epic 3 - Core Services Layer
**Story Points:** 8
**Priority:** Critical
**Dependencies:** Epic 2 (database for caching)

---

## User Story

**As a** system
**I want** a centralized embedding service
**So that** all strategies use consistent embeddings

---

## Detailed Requirements

### Functional Requirements

1. **Multi-Provider Support**
   - Support for OpenAI embedding models (text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002)
   - Support for Cohere embedding models (embed-english-v3.0, embed-multilingual-v3.0)
   - Support for local models via sentence-transformers (all-MiniLM-L6-v2, all-mpnet-base-v2)
   - Pluggable architecture for adding new providers
   - Provider-specific configuration

2. **Batch Processing**
   - Batch multiple texts in single API call for efficiency
   - Configurable batch sizes per provider
   - Automatic batch splitting for oversized requests
   - Parallel processing for local models
   - Progress tracking for large batches

3. **Caching Layer**
   - Cache embeddings by text hash to avoid redundant API calls
   - Support multiple cache backends (in-memory, Redis, database)
   - Configurable cache TTL (time-to-live)
   - Cache hit/miss metrics
   - Cache invalidation strategies

4. **Rate Limiting & Retries**
   - Rate limiting per provider (respects API limits)
   - Exponential backoff on failures
   - Configurable retry attempts
   - Error handling for API failures
   - Queue management for rate-limited requests

5. **Model Management**
   - Switch models via configuration without code changes
   - Model version tracking
   - Warm-up for local models (preload)
   - Model metadata (dimensions, max tokens, cost)

6. **Cost Tracking**
   - Track API usage per provider
   - Calculate costs based on token usage
   - Cost reporting and budgeting
   - Usage analytics

### Non-Functional Requirements

1. **Performance**
   - Single embedding generation <100ms (excluding API latency)
   - Batch processing >100 texts/second for local models
   - Cache lookup <10ms
   - Support concurrent requests (thread-safe)

2. **Reliability**
   - Handle API failures gracefully with retries
   - Fallback to alternative providers if configured
   - Automatic recovery from transient errors
   - Comprehensive error logging

3. **Scalability**
   - Support 1000+ concurrent embedding requests
   - Cache should handle millions of embeddings
   - Horizontal scaling support

4. **Maintainability**
   - Clear separation of concerns (provider adapters)
   - Comprehensive logging and monitoring
   - Configuration-driven behavior
   - Well-documented API

5. **Security**
   - Secure API key storage (environment variables)
   - No API keys in logs or errors
   - Encrypted cache for sensitive data (optional)

---

## Acceptance Criteria

### AC1: Provider Support
- [ ] OpenAI provider implemented with all models
- [ ] Cohere provider implemented
- [ ] Local sentence-transformers provider implemented
- [ ] Provider selection via configuration
- [ ] New providers can be added without modifying core code

### AC2: Batch Processing
- [ ] Batch processing supported for all providers
- [ ] Automatic batch splitting when exceeding provider limits
- [ ] Batch size configurable per provider
- [ ] Parallel processing for local models
- [ ] Progress callbacks for long-running batches

### AC3: Caching
- [ ] In-memory cache implementation
- [ ] Cache key generation from text content
- [ ] Cache hit rate >80% for repeated texts
- [ ] Cache TTL configurable
- [ ] Cache statistics available (hits, misses, size)

### AC4: Rate Limiting
- [ ] Rate limiter implemented per provider
- [ ] Configurable rate limits (requests per minute/second)
- [ ] Exponential backoff on rate limit errors
- [ ] Retry logic with configurable max attempts
- [ ] Queue for rate-limited requests

### AC5: Model Management
- [ ] Models configurable via config file
- [ ] Model metadata stored (dimensions, cost, etc.)
- [ ] Local models preloaded on initialization
- [ ] Model switching without restart (for non-local models)

### AC6: Error Handling
- [ ] API errors caught and wrapped in custom exceptions
- [ ] Meaningful error messages with context
- [ ] Retry logic for transient errors
- [ ] Fallback provider support (optional)
- [ ] All errors logged with details

### AC7: Cost Tracking
- [ ] Token usage tracked per request
- [ ] Cost calculated per provider
- [ ] Usage metrics available via API
- [ ] Cost alerts when exceeding budget (optional)

### AC8: Testing
- [ ] Unit tests for all providers with mocked APIs
- [ ] Integration tests with real providers (skippable)
- [ ] Cache tests (hit/miss scenarios)
- [ ] Rate limiting tests
- [ ] Performance benchmarks

---

## Technical Specifications

### File Structure
```
rag_factory/
├── services/
│   ├── __init__.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── base.py              # Base embedding provider interface
│   │   ├── service.py           # Main embedding service
│   │   ├── cache.py             # Caching implementation
│   │   ├── rate_limiter.py      # Rate limiting
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── openai.py        # OpenAI provider
│   │   │   ├── cohere.py        # Cohere provider
│   │   │   └── local.py         # Local models provider
│   │   └── config.py            # Embedding service config
│
tests/
├── unit/
│   └── services/
│       └── embedding/
│           ├── test_service.py
│           ├── test_cache.py
│           ├── test_rate_limiter.py
│           ├── test_openai_provider.py
│           ├── test_cohere_provider.py
│           └── test_local_provider.py
│
├── integration/
│   └── services/
│       └── test_embedding_integration.py
```

### Dependencies
```python
# requirements.txt additions
openai==1.12.0              # OpenAI embeddings
cohere==4.47                # Cohere embeddings
sentence-transformers==2.3.1 # Local embeddings
redis==5.0.1                # Optional cache backend
tiktoken==0.5.2             # Token counting
tenacity==8.2.3             # Retry logic
```

### Base Provider Interface
```python
# rag_factory/services/embedding/base.py
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    embeddings: List[List[float]]
    model: str
    dimensions: int
    token_count: int
    cost: float
    provider: str
    cached: List[bool]  # Which embeddings came from cache

class IEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initialize provider with configuration."""
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings for list of texts."""
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """Get embedding dimensions for this model."""
        pass

    @abstractmethod
    def get_max_batch_size(self) -> int:
        """Get maximum batch size for this provider."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name."""
        pass

    @abstractmethod
    def calculate_cost(self, token_count: int) -> float:
        """Calculate cost for given token count."""
        pass
```

### Embedding Service
```python
# rag_factory/services/embedding/service.py
from typing import List, Optional, Dict, Any
import hashlib
import logging
from .base import IEmbeddingProvider, EmbeddingResult
from .cache import EmbeddingCache
from .rate_limiter import RateLimiter
from .config import EmbeddingServiceConfig

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Centralized embedding service with caching and rate limiting.

    Example:
        config = EmbeddingServiceConfig(provider="openai", model="text-embedding-3-small")
        service = EmbeddingService(config)
        result = service.embed(["Hello world", "Another text"])
        embeddings = result.embeddings
    """

    def __init__(self, config: EmbeddingServiceConfig):
        self.config = config
        self.provider = self._init_provider()
        self.cache = EmbeddingCache(config.cache_config) if config.enable_cache else None
        self.rate_limiter = RateLimiter(config.rate_limit_config) if config.enable_rate_limiting else None
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        }

    def _init_provider(self) -> IEmbeddingProvider:
        """Initialize the embedding provider based on config."""
        provider_map = {
            "openai": OpenAIProvider,
            "cohere": CohereProvider,
            "local": LocalProvider
        }

        provider_class = provider_map.get(self.config.provider)
        if not provider_class:
            raise ValueError(f"Unknown provider: {self.config.provider}")

        return provider_class(self.config.provider_config)

    def embed(self, texts: List[str], use_cache: bool = True) -> EmbeddingResult:
        """
        Generate embeddings for list of texts.

        Args:
            texts: List of text strings to embed
            use_cache: Whether to use cache (default: True)

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not texts:
            raise ValueError("texts cannot be empty")

        self._stats["total_requests"] += 1

        # Check cache first
        embeddings = []
        texts_to_embed = []
        cached_flags = []

        if use_cache and self.cache:
            for text in texts:
                cache_key = self._compute_cache_key(text)
                cached_embedding = self.cache.get(cache_key)

                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                    cached_flags.append(True)
                    self._stats["cache_hits"] += 1
                else:
                    texts_to_embed.append(text)
                    cached_flags.append(False)
                    self._stats["cache_misses"] += 1
        else:
            texts_to_embed = texts
            cached_flags = [False] * len(texts)

        # Generate embeddings for uncached texts
        if texts_to_embed:
            # Apply rate limiting
            if self.rate_limiter:
                self.rate_limiter.wait_if_needed()

            # Split into batches if needed
            batches = self._create_batches(texts_to_embed)
            new_embeddings = []

            for batch in batches:
                try:
                    batch_result = self.provider.get_embeddings(batch)
                    new_embeddings.extend(batch_result.embeddings)

                    # Update stats
                    self._stats["total_tokens"] += batch_result.token_count
                    self._stats["total_cost"] += batch_result.cost

                    # Cache new embeddings
                    if self.cache:
                        for text, embedding in zip(batch, batch_result.embeddings):
                            cache_key = self._compute_cache_key(text)
                            self.cache.set(cache_key, embedding)

                except Exception as e:
                    logger.error(f"Error generating embeddings: {e}")
                    raise

            # Merge cached and new embeddings
            result_embeddings = []
            new_idx = 0
            cached_idx = 0

            for is_cached in cached_flags:
                if is_cached:
                    result_embeddings.append(embeddings[cached_idx])
                    cached_idx += 1
                else:
                    result_embeddings.append(new_embeddings[new_idx])
                    new_idx += 1

            embeddings = result_embeddings

        # Create result
        return EmbeddingResult(
            embeddings=embeddings,
            model=self.provider.get_model_name(),
            dimensions=self.provider.get_dimensions(),
            token_count=self._stats["total_tokens"],
            cost=self._stats["total_cost"],
            provider=self.config.provider,
            cached=cached_flags
        )

    def _compute_cache_key(self, text: str) -> str:
        """Compute cache key from text."""
        model_name = self.provider.get_model_name()
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _create_batches(self, texts: List[str]) -> List[List[str]]:
        """Split texts into batches based on provider limits."""
        max_batch_size = self.provider.get_max_batch_size()
        batches = []

        for i in range(0, len(texts), max_batch_size):
            batches.append(texts[i:i + max_batch_size])

        return batches

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        cache_hit_rate = 0.0
        if self._stats["total_requests"] > 0:
            cache_hit_rate = self._stats["cache_hits"] / (
                self._stats["cache_hits"] + self._stats["cache_misses"]
            )

        return {
            **self._stats,
            "cache_hit_rate": cache_hit_rate,
            "model": self.provider.get_model_name(),
            "provider": self.config.provider
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        if self.cache:
            self.cache.clear()
```

### OpenAI Provider
```python
# rag_factory/services/embedding/providers/openai.py
from typing import List, Dict, Any
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from ..base import IEmbeddingProvider, EmbeddingResult

class OpenAIProvider(IEmbeddingProvider):
    """OpenAI embedding provider."""

    MODELS = {
        "text-embedding-3-small": {"dimensions": 1536, "cost_per_1k": 0.00002},
        "text-embedding-3-large": {"dimensions": 3072, "cost_per_1k": 0.00013},
        "text-embedding-ada-002": {"dimensions": 1536, "cost_per_1k": 0.0001}
    }

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("api_key")
        self.model = config.get("model", "text-embedding-3-small")
        self.max_batch_size = config.get("max_batch_size", 100)

        if self.model not in self.MODELS:
            raise ValueError(f"Unknown OpenAI model: {self.model}")

        self.client = openai.OpenAI(api_key=self.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_embeddings(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings using OpenAI API."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )

        embeddings = [item.embedding for item in response.data]
        token_count = response.usage.total_tokens
        cost = self.calculate_cost(token_count)

        return EmbeddingResult(
            embeddings=embeddings,
            model=self.model,
            dimensions=self.get_dimensions(),
            token_count=token_count,
            cost=cost,
            provider="openai",
            cached=[False] * len(texts)
        )

    def get_dimensions(self) -> int:
        return self.MODELS[self.model]["dimensions"]

    def get_max_batch_size(self) -> int:
        return self.max_batch_size

    def get_model_name(self) -> str:
        return self.model

    def calculate_cost(self, token_count: int) -> float:
        cost_per_1k = self.MODELS[self.model]["cost_per_1k"]
        return (token_count / 1000.0) * cost_per_1k
```

### Cache Implementation
```python
# rag_factory/services/embedding/cache.py
from typing import Optional, List, Dict, Any
import time
from collections import OrderedDict
import threading

class EmbeddingCache:
    """
    In-memory LRU cache for embeddings.
    Thread-safe implementation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.max_size = config.get("max_size", 10000)
        self.ttl = config.get("ttl", 3600)  # seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        with self._lock:
            # Check if key exists and not expired
            if key in self._cache:
                timestamp = self._timestamps.get(key, 0)
                if time.time() - timestamp < self.ttl:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return self._cache[key]
                else:
                    # Expired, remove
                    del self._cache[key]
                    del self._timestamps[key]

            self._misses += 1
            return None

    def set(self, key: str, embedding: List[float]):
        """Set embedding in cache."""
        with self._lock:
            # Remove oldest if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]

            self._cache[key] = embedding
            self._timestamps[key] = time.time()
            self._cache.move_to_end(key)

    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate
            }
```

### Rate Limiter
```python
# rag_factory/services/embedding/rate_limiter.py
import time
import threading
from typing import Dict, Any

class RateLimiter:
    """
    Token bucket rate limiter.
    Thread-safe implementation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.requests_per_minute = config.get("requests_per_minute", 60)
        self.requests_per_second = config.get("requests_per_second", None)

        # Use most restrictive limit
        if self.requests_per_second:
            self.min_interval = 1.0 / self.requests_per_second
        else:
            self.min_interval = 60.0 / self.requests_per_minute

        self._last_request_time = 0.0
        self._lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        with self._lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time

            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)

            self._last_request_time = time.time()
```

---

## Unit Tests

### Test File Locations
- `tests/unit/services/embedding/test_service.py`
- `tests/unit/services/embedding/test_cache.py`
- `tests/unit/services/embedding/test_rate_limiter.py`
- `tests/unit/services/embedding/test_openai_provider.py`

### Test Cases

#### TC3.1.1: Embedding Service Tests
```python
import pytest
from unittest.mock import Mock, patch
from rag_factory.services.embedding.service import EmbeddingService
from rag_factory.services.embedding.config import EmbeddingServiceConfig

@pytest.fixture
def mock_config():
    return EmbeddingServiceConfig(
        provider="openai",
        model="text-embedding-3-small",
        enable_cache=True,
        enable_rate_limiting=False
    )

@pytest.fixture
def mock_provider():
    provider = Mock()
    provider.get_embeddings.return_value = Mock(
        embeddings=[[0.1, 0.2, 0.3]] * 2,
        model="test-model",
        dimensions=3,
        token_count=10,
        cost=0.0001,
        provider="mock",
        cached=[False, False]
    )
    provider.get_dimensions.return_value = 3
    provider.get_max_batch_size.return_value = 100
    provider.get_model_name.return_value = "test-model"
    return provider

def test_service_initialization(mock_config):
    """Test service initializes correctly."""
    service = EmbeddingService(mock_config)
    assert service.config == mock_config
    assert service.provider is not None
    assert service.cache is not None

def test_embed_single_text(mock_config, mock_provider, monkeypatch):
    """Test embedding single text."""
    service = EmbeddingService(mock_config)
    monkeypatch.setattr(service, "provider", mock_provider)

    result = service.embed(["Hello world"])

    assert len(result.embeddings) == 1
    assert result.model == "test-model"
    assert result.dimensions == 3

def test_embed_multiple_texts(mock_config, mock_provider, monkeypatch):
    """Test embedding multiple texts."""
    service = EmbeddingService(mock_config)
    monkeypatch.setattr(service, "provider", mock_provider)

    result = service.embed(["Text 1", "Text 2"])

    assert len(result.embeddings) == 2
    mock_provider.get_embeddings.assert_called_once()

def test_embed_empty_list_raises_error(mock_config):
    """Test embedding empty list raises error."""
    service = EmbeddingService(mock_config)

    with pytest.raises(ValueError, match="texts cannot be empty"):
        service.embed([])

def test_cache_hit(mock_config, mock_provider, monkeypatch):
    """Test cache returns cached embedding."""
    service = EmbeddingService(mock_config)
    monkeypatch.setattr(service, "provider", mock_provider)

    # First call - cache miss
    result1 = service.embed(["Hello"])

    # Second call - should hit cache
    result2 = service.embed(["Hello"])

    # Provider should only be called once
    assert mock_provider.get_embeddings.call_count == 1
    assert service._stats["cache_hits"] == 1
    assert service._stats["cache_misses"] == 1

def test_cache_disabled(mock_provider, monkeypatch):
    """Test service works with cache disabled."""
    config = EmbeddingServiceConfig(
        provider="openai",
        enable_cache=False
    )
    service = EmbeddingService(config)
    monkeypatch.setattr(service, "provider", mock_provider)

    service.embed(["Hello"])
    service.embed(["Hello"])

    # Provider called twice (no caching)
    assert mock_provider.get_embeddings.call_count == 2

def test_batch_splitting(mock_config, mock_provider, monkeypatch):
    """Test automatic batch splitting."""
    mock_provider.get_max_batch_size.return_value = 2
    service = EmbeddingService(mock_config)
    monkeypatch.setattr(service, "provider", mock_provider)

    # 5 texts with batch size of 2 should create 3 batches
    result = service.embed(["A", "B", "C", "D", "E"], use_cache=False)

    assert mock_provider.get_embeddings.call_count == 3

def test_get_stats(mock_config, mock_provider, monkeypatch):
    """Test statistics tracking."""
    service = EmbeddingService(mock_config)
    monkeypatch.setattr(service, "provider", mock_provider)

    service.embed(["Hello"])
    service.embed(["Hello"])  # Cache hit
    service.embed(["World"])  # Cache miss

    stats = service.get_stats()

    assert stats["total_requests"] == 3
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 2
    assert stats["cache_hit_rate"] == pytest.approx(0.333, 0.01)

def test_clear_cache(mock_config, mock_provider, monkeypatch):
    """Test cache clearing."""
    service = EmbeddingService(mock_config)
    monkeypatch.setattr(service, "provider", mock_provider)

    service.embed(["Hello"])
    service.clear_cache()
    service.embed(["Hello"])

    # Should call provider twice (cache cleared)
    assert mock_provider.get_embeddings.call_count == 2
```

#### TC3.1.2: Cache Tests
```python
import pytest
import time
from rag_factory.services.embedding.cache import EmbeddingCache

@pytest.fixture
def cache():
    return EmbeddingCache({"max_size": 3, "ttl": 1})

def test_cache_set_and_get(cache):
    """Test basic set and get operations."""
    embedding = [0.1, 0.2, 0.3]
    cache.set("key1", embedding)

    result = cache.get("key1")
    assert result == embedding

def test_cache_miss(cache):
    """Test cache miss returns None."""
    result = cache.get("nonexistent")
    assert result is None

def test_cache_expiration(cache):
    """Test cache entry expires after TTL."""
    embedding = [0.1, 0.2, 0.3]
    cache.set("key1", embedding)

    # Wait for expiration
    time.sleep(1.1)

    result = cache.get("key1")
    assert result is None

def test_cache_max_size(cache):
    """Test cache respects max size."""
    cache.set("key1", [0.1])
    cache.set("key2", [0.2])
    cache.set("key3", [0.3])
    cache.set("key4", [0.4])  # Should evict key1

    assert cache.get("key1") is None
    assert cache.get("key4") is not None

def test_cache_lru_eviction(cache):
    """Test LRU eviction policy."""
    cache.set("key1", [0.1])
    cache.set("key2", [0.2])
    cache.set("key3", [0.3])

    # Access key1 to make it most recent
    cache.get("key1")

    # Add new key, should evict key2 (least recently used)
    cache.set("key4", [0.4])

    assert cache.get("key2") is None
    assert cache.get("key1") is not None

def test_cache_clear(cache):
    """Test cache clearing."""
    cache.set("key1", [0.1])
    cache.set("key2", [0.2])

    cache.clear()

    assert cache.get("key1") is None
    assert cache.get("key2") is None

def test_cache_stats(cache):
    """Test cache statistics."""
    cache.set("key1", [0.1])

    cache.get("key1")  # Hit
    cache.get("key2")  # Miss
    cache.get("key1")  # Hit

    stats = cache.get_stats()

    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.666, 0.01)
    assert stats["size"] == 1

def test_cache_thread_safety():
    """Test cache is thread-safe."""
    import threading

    cache = EmbeddingCache({"max_size": 100, "ttl": 10})

    def worker(thread_id):
        for i in range(100):
            cache.set(f"key_{thread_id}_{i}", [float(i)])
            cache.get(f"key_{thread_id}_{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # Should not crash
    stats = cache.get_stats()
    assert stats["size"] <= 100
```

#### TC3.1.3: Rate Limiter Tests
```python
import pytest
import time
from rag_factory.services.embedding.rate_limiter import RateLimiter

def test_rate_limiter_initialization():
    """Test rate limiter initializes correctly."""
    limiter = RateLimiter({"requests_per_minute": 60})
    assert limiter.min_interval == 1.0

def test_rate_limiter_allows_first_request():
    """Test first request is not delayed."""
    limiter = RateLimiter({"requests_per_second": 10})

    start = time.time()
    limiter.wait_if_needed()
    duration = time.time() - start

    assert duration < 0.01  # Should be immediate

def test_rate_limiter_enforces_limit():
    """Test rate limiter enforces rate limit."""
    limiter = RateLimiter({"requests_per_second": 2})

    start = time.time()
    limiter.wait_if_needed()
    limiter.wait_if_needed()
    duration = time.time() - start

    # Second request should wait ~0.5 seconds
    assert duration >= 0.4

def test_rate_limiter_requests_per_minute():
    """Test rate limiter with requests per minute."""
    limiter = RateLimiter({"requests_per_minute": 120})

    # Should allow 2 requests per second
    start = time.time()
    limiter.wait_if_needed()
    limiter.wait_if_needed()
    duration = time.time() - start

    assert duration >= 0.4

def test_rate_limiter_thread_safety():
    """Test rate limiter is thread-safe."""
    import threading

    limiter = RateLimiter({"requests_per_second": 10})
    results = []

    def worker():
        start = time.time()
        limiter.wait_if_needed()
        results.append(time.time() - start)

    threads = [threading.Thread(target=worker) for _ in range(5)]

    start = time.time()
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    total_duration = time.time() - start

    # 5 requests at 10 req/sec should take ~0.4 seconds
    assert total_duration >= 0.3
```

#### TC3.1.4: OpenAI Provider Tests
```python
import pytest
from unittest.mock import Mock, patch
from rag_factory.services.embedding.providers.openai import OpenAIProvider

@pytest.fixture
def openai_config():
    return {
        "api_key": "test-key",
        "model": "text-embedding-3-small"
    }

def test_provider_initialization(openai_config):
    """Test provider initializes correctly."""
    provider = OpenAIProvider(openai_config)
    assert provider.model == "text-embedding-3-small"
    assert provider.get_dimensions() == 1536

def test_provider_invalid_model():
    """Test provider raises error for invalid model."""
    config = {"api_key": "test-key", "model": "invalid-model"}

    with pytest.raises(ValueError, match="Unknown OpenAI model"):
        OpenAIProvider(config)

@patch("openai.OpenAI")
def test_get_embeddings(mock_openai_class, openai_config):
    """Test getting embeddings."""
    # Mock OpenAI response
    mock_client = Mock()
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1, 0.2, 0.3]),
        Mock(embedding=[0.4, 0.5, 0.6])
    ]
    mock_response.usage.total_tokens = 10
    mock_client.embeddings.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    provider = OpenAIProvider(openai_config)
    result = provider.get_embeddings(["Hello", "World"])

    assert len(result.embeddings) == 2
    assert result.token_count == 10
    assert result.model == "text-embedding-3-small"
    assert result.provider == "openai"

def test_calculate_cost(openai_config):
    """Test cost calculation."""
    provider = OpenAIProvider(openai_config)

    # text-embedding-3-small costs $0.00002 per 1k tokens
    cost = provider.calculate_cost(1000)
    assert cost == pytest.approx(0.00002)

    cost = provider.calculate_cost(5000)
    assert cost == pytest.approx(0.0001)

def test_get_max_batch_size(openai_config):
    """Test max batch size."""
    provider = OpenAIProvider(openai_config)
    assert provider.get_max_batch_size() == 100

def test_get_model_name(openai_config):
    """Test getting model name."""
    provider = OpenAIProvider(openai_config)
    assert provider.get_model_name() == "text-embedding-3-small"
```

---

## Integration Tests

### Test File Location
`tests/integration/services/test_embedding_integration.py`

### Test Scenarios

#### IS3.1.1: End-to-End Embedding Workflow
```python
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="API key not set")
def test_full_embedding_workflow():
    """Test complete embedding workflow with real API."""
    config = EmbeddingServiceConfig(
        provider="openai",
        model="text-embedding-3-small",
        provider_config={
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        enable_cache=True,
        enable_rate_limiting=True
    )

    service = EmbeddingService(config)

    # Test single embedding
    result1 = service.embed(["Hello, world!"])
    assert len(result1.embeddings) == 1
    assert len(result1.embeddings[0]) == 1536
    assert result1.cached[0] == False

    # Test cache hit
    result2 = service.embed(["Hello, world!"])
    assert result2.cached[0] == True

    # Test batch embedding
    texts = [f"Test text {i}" for i in range(10)]
    result3 = service.embed(texts)
    assert len(result3.embeddings) == 10

    # Check stats
    stats = service.get_stats()
    assert stats["total_requests"] == 3
    assert stats["cache_hits"] >= 1

@pytest.mark.integration
def test_local_embedding_provider():
    """Test local sentence-transformers provider."""
    config = EmbeddingServiceConfig(
        provider="local",
        model="all-MiniLM-L6-v2",
        enable_cache=True
    )

    service = EmbeddingService(config)

    texts = ["This is a test", "Another test"]
    result = service.embed(texts)

    assert len(result.embeddings) == 2
    assert result.provider == "local"
    assert result.cost == 0.0  # Local models have no cost

@pytest.mark.integration
def test_multiple_providers_consistency():
    """Test that different providers give similar results."""
    import numpy as np
    from scipy.spatial.distance import cosine

    text = "Machine learning is fascinating"

    # Get embeddings from different providers
    providers_configs = [
        {"provider": "openai", "model": "text-embedding-3-small"},
        {"provider": "local", "model": "all-MiniLM-L6-v2"}
    ]

    embeddings = []
    for prov_config in providers_configs:
        config = EmbeddingServiceConfig(**prov_config)
        service = EmbeddingService(config)
        result = service.embed([text])
        embeddings.append(result.embeddings[0])

    # Embeddings should have some similarity (not exact due to different models)
    # Just verify they're valid embeddings
    for emb in embeddings:
        assert len(emb) > 0
        assert all(isinstance(x, float) for x in emb)

@pytest.mark.integration
def test_large_batch_processing():
    """Test processing large batch of texts."""
    config = EmbeddingServiceConfig(
        provider="local",
        model="all-MiniLM-L6-v2"
    )

    service = EmbeddingService(config)

    # Generate 1000 texts
    texts = [f"Document {i} with some content" for i in range(1000)]

    import time
    start = time.time()
    result = service.embed(texts, use_cache=False)
    duration = time.time() - start

    assert len(result.embeddings) == 1000

    # Should process at least 100 texts/second for local model
    throughput = 1000 / duration
    assert throughput > 100, f"Throughput {throughput:.0f} texts/sec is too low"

@pytest.mark.integration
def test_concurrent_embedding_requests():
    """Test concurrent requests to embedding service."""
    import concurrent.futures

    config = EmbeddingServiceConfig(
        provider="local",
        model="all-MiniLM-L6-v2",
        enable_cache=True
    )

    service = EmbeddingService(config)

    def embed_text(text_id):
        texts = [f"Concurrent text {text_id}"]
        result = service.embed(texts)
        return result.embeddings[0]

    # Run 50 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(embed_text, i) for i in range(50)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # All requests should succeed
    assert len(results) == 50
    assert all(len(emb) > 0 for emb in results)
```

---

## Performance Benchmarks

```python
# tests/benchmarks/test_embedding_performance.py

import pytest
import time
from rag_factory.services.embedding.service import EmbeddingService
from rag_factory.services.embedding.config import EmbeddingServiceConfig

@pytest.mark.benchmark
def test_single_embedding_performance():
    """Test single embedding generation is <100ms."""
    config = EmbeddingServiceConfig(
        provider="local",
        model="all-MiniLM-L6-v2",
        enable_cache=False
    )

    service = EmbeddingService(config)

    # Warm up
    service.embed(["warmup"])

    # Benchmark
    start = time.time()
    service.embed(["Performance test"])
    duration = (time.time() - start) * 1000

    assert duration < 100, f"Single embedding took {duration:.2f}ms (expected <100ms)"

@pytest.mark.benchmark
def test_batch_embedding_throughput():
    """Test batch embedding throughput >100 texts/second."""
    config = EmbeddingServiceConfig(
        provider="local",
        model="all-MiniLM-L6-v2",
        enable_cache=False
    )

    service = EmbeddingService(config)

    texts = [f"Text {i}" for i in range(500)]

    start = time.time()
    service.embed(texts)
    duration = time.time() - start

    throughput = 500 / duration
    assert throughput > 100, f"Throughput {throughput:.0f} texts/sec (expected >100)"

@pytest.mark.benchmark
def test_cache_lookup_performance():
    """Test cache lookup is <10ms."""
    config = EmbeddingServiceConfig(
        provider="local",
        model="all-MiniLM-L6-v2",
        enable_cache=True
    )

    service = EmbeddingService(config)

    # Prime cache
    service.embed(["Cached text"])

    # Benchmark cache lookup
    start = time.time()
    service.embed(["Cached text"])
    duration = (time.time() - start) * 1000

    assert duration < 10, f"Cache lookup took {duration:.2f}ms (expected <10ms)"

@pytest.mark.benchmark
def test_concurrent_request_handling():
    """Test handling 1000+ concurrent requests."""
    import concurrent.futures

    config = EmbeddingServiceConfig(
        provider="local",
        model="all-MiniLM-L6-v2",
        enable_cache=True
    )

    service = EmbeddingService(config)

    def embed_task(i):
        return service.embed([f"Text {i}"])

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(embed_task, i) for i in range(1000)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    duration = time.time() - start

    assert len(results) == 1000
    assert duration < 60, f"1000 concurrent requests took {duration:.2f}s (expected <60s)"
```

---

## Definition of Done

- [ ] Base embedding provider interface defined
- [ ] OpenAI provider fully implemented
- [ ] Cohere provider fully implemented
- [ ] Local sentence-transformers provider implemented
- [ ] Embedding service with caching implemented
- [ ] Rate limiting implemented
- [ ] All unit tests pass (>90% coverage)
- [ ] All integration tests pass
- [ ] Performance benchmarks meet requirements
- [ ] Configuration system working
- [ ] Cost tracking implemented
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] No linting errors

---

## Setup Instructions

### Installation

```bash
# Install dependencies
pip install openai cohere sentence-transformers tiktoken tenacity redis

# For local models, download model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Configuration

```python
# config.yaml
embedding_service:
  provider: "openai"  # openai, cohere, or local
  model: "text-embedding-3-small"

  provider_config:
    api_key: "${OPENAI_API_KEY}"
    max_batch_size: 100

  cache:
    enabled: true
    max_size: 10000
    ttl: 3600

  rate_limiting:
    enabled: true
    requests_per_minute: 3000
```

### Environment Variables

```bash
export OPENAI_API_KEY="your-key-here"
export COHERE_API_KEY="your-key-here"
```

### Usage Example

```python
from rag_factory.services.embedding import EmbeddingService, EmbeddingServiceConfig

# Initialize service
config = EmbeddingServiceConfig(
    provider="openai",
    model="text-embedding-3-small"
)
service = EmbeddingService(config)

# Generate embeddings
texts = ["Hello world", "Another text"]
result = service.embed(texts)

print(f"Generated {len(result.embeddings)} embeddings")
print(f"Dimensions: {result.dimensions}")
print(f"Cost: ${result.cost:.6f}")

# Check statistics
stats = service.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

---

## Notes for Developers

1. **Provider Selection**: Start with local models for development to avoid API costs.

2. **Caching**: Caching significantly improves performance and reduces costs. Enable it by default.

3. **Rate Limiting**: Configure rate limits based on your API plan to avoid hitting limits.

4. **Batch Processing**: Always use batch processing when embedding multiple texts.

5. **Error Handling**: The service includes automatic retries with exponential backoff.

6. **Testing**: Use mocked providers for unit tests to avoid API calls during testing.

7. **Cost Management**: Monitor costs using the built-in tracking. Set budget alerts if needed.

8. **Model Selection**: Choose model based on requirements:
   - `text-embedding-3-small`: Fast, cheap, good for most use cases
   - `text-embedding-3-large`: Better quality, more expensive
   - Local models: No cost, good for development

9. **Thread Safety**: All components are thread-safe and can handle concurrent requests.

10. **Monitoring**: Use the `get_stats()` method to monitor performance and cache effectiveness.
