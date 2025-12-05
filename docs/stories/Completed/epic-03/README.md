# Epic 3: Core Services Layer - Stories Summary

**Epic Goal:** Build the foundational services (Embedding Service and LLM Service) that all RAG strategies will depend on.

**Total Story Points:** 16
**Dependencies:** Epic 2 (database for caching)

---

## Stories Overview

### Story 3.1: Build Embedding Service
**Story Points:** 8
**File:** `story-3.1-embedding-service.md`
**Status:** Ready for Development

#### What's Included:
✅ **User Story & Requirements**
- Centralized embedding service for consistent embeddings
- Multi-provider support (OpenAI, Cohere, local models)
- Batch processing for efficiency
- Caching layer with multiple backends
- Rate limiting and retry logic
- Cost tracking

✅ **Acceptance Criteria (8 ACs)**
- AC1: Provider Support (OpenAI, Cohere, Local)
- AC2: Batch Processing
- AC3: Caching
- AC4: Rate Limiting
- AC5: Model Management
- AC6: Error Handling
- AC7: Cost Tracking
- AC8: Testing

✅ **Technical Specifications**
- Base embedding provider interface
- OpenAI provider implementation
- Cohere provider implementation
- Local sentence-transformers provider
- Embedding service with caching
- LRU cache with TTL
- Rate limiter (token bucket)
- ~600 lines of implementation code

✅ **Working Code Examples**
```python
# IEmbeddingProvider interface
# EmbeddingService (main service)
# OpenAIProvider (with retry logic)
# CohereProvider
# LocalProvider (sentence-transformers)
# EmbeddingCache (LRU with TTL)
# RateLimiter (thread-safe)
```

✅ **Unit Tests (40+ test cases)**
- TC3.1.1: Embedding Service Tests (10 tests)
  - Initialization, embedding single/multiple texts
  - Cache hit/miss scenarios
  - Batch splitting
  - Statistics tracking

- TC3.1.2: Cache Tests (8 tests)
  - Set/get operations
  - Expiration and LRU eviction
  - Thread safety
  - Statistics

- TC3.1.3: Rate Limiter Tests (5 tests)
  - Rate enforcement
  - Thread safety
  - Different rate configurations

- TC3.1.4: OpenAI Provider Tests (6 tests)
  - Initialization, embeddings generation
  - Cost calculation
  - Error handling

✅ **Integration Tests (5 scenarios)**
- End-to-end embedding workflow
- Local provider testing
- Multiple providers consistency
- Large batch processing (1000+ texts)
- Concurrent requests

✅ **Performance Benchmarks (4 tests)**
- Single embedding <100ms
- Batch throughput >100 texts/second
- Cache lookup <10ms
- 1000+ concurrent requests

---

### Story 3.2: Implement LLM Service Adapter
**Story Points:** 8
**File:** `story-3.2-llm-service-adapter.md`
**Status:** Ready for Development

#### What's Included:
✅ **User Story & Requirements**
- Unified interface for LLM calls
- Multi-provider support (Anthropic, OpenAI, Ollama)
- Prompt template system
- Token counting and cost tracking
- Rate limiting and retries
- Streaming support

✅ **Acceptance Criteria (8 ACs)**
- AC1: Provider Support (Anthropic, OpenAI, Ollama)
- AC2: Prompt Templates
- AC3: Response Handling
- AC4: Token Counting
- AC5: Cost Tracking
- AC6: Rate Limiting
- AC7: Streaming
- AC8: Testing

✅ **Technical Specifications**
- Base LLM provider interface
- Anthropic Claude provider (Sonnet, Opus, Haiku)
- OpenAI provider (GPT-4, GPT-3.5)
- Ollama local provider
- LLM service with rate limiting
- Prompt template system with variables
- Token counting utilities
- ~700 lines of implementation code

✅ **Working Code Examples**
```python
# ILLMProvider interface
# LLMService (main service)
# AnthropicProvider (with streaming)
# OpenAIProvider
# OllamaProvider
# PromptTemplate (with variable substitution)
# CommonTemplates (RAG QA, summarization, etc.)
```

✅ **Unit Tests (35+ test cases)**
- TC3.2.1: LLM Service Tests (12 tests)
  - Completion with single/multiple messages
  - Temperature and max_tokens parameters
  - Statistics tracking
  - Streaming with callbacks
  - Cost estimation

- TC3.2.2: Prompt Template Tests (8 tests)
  - System and user messages
  - Few-shot examples
  - Variable substitution
  - Template validation
  - Common templates

- TC3.2.3: Anthropic Provider Tests (9 tests)
  - Initialization, completion
  - System message handling
  - Token counting and cost calculation
  - Streaming

✅ **Integration Tests (5 scenarios)**
- End-to-end LLM workflow
- Streaming responses
- Local Ollama provider
- Prompt templates with real LLM
- Concurrent LLM requests

✅ **Performance Benchmarks (2 tests)**
- Request overhead <50ms
- 100+ concurrent requests

---

## Combined Statistics

### Documentation
- **Total Lines:** 2,917 lines
- **Total Size:** 88 KB
- **Story Points:** 16 (8 + 8)

### Code Provided
- **Implementation Code:** ~1,300 lines
- **Service Classes:** 2 main services (Embedding, LLM)
- **Provider Classes:** 6 providers (3 embedding + 3 LLM)
- **Utility Classes:** 4 (Cache, RateLimiter, PromptTemplate, TokenCounter)

### Tests Defined
- **Unit Tests:** 75+ functions
- **Integration Tests:** 10 scenarios
- **Performance Benchmarks:** 6 tests
- **Total Test Coverage:** 90+ test cases

### Requirements
- **Functional Requirements:** 12 total
- **Non-functional Requirements:** 10 total
- **Acceptance Criteria:** 16 ACs with 80+ specific criteria

---

## Developer Workflow

### Prerequisites
1. Complete Epic 2 (database for caching)
2. Python 3.11+ environment
3. API keys for OpenAI, Anthropic, Cohere (optional)
4. Ollama installed (optional for local models)

### Story 3.1: Embedding Service

**Implementation Steps:**
1. Define base embedding provider interface
2. Implement OpenAI provider
3. Implement Cohere provider
4. Implement local sentence-transformers provider
5. Build embedding service with caching
6. Implement LRU cache with TTL
7. Implement rate limiter
8. Write unit tests for all components
9. Write integration tests
10. Run performance benchmarks

**Testing:**
```bash
# Unit tests
pytest tests/unit/services/embedding/ -v

# Integration tests (requires API keys)
export OPENAI_API_KEY="..."
export COHERE_API_KEY="..."
pytest tests/integration/services/test_embedding_integration.py -v

# Performance benchmarks
pytest tests/benchmarks/test_embedding_performance.py -v
```

**Verification:**
- [ ] All providers support embedding generation
- [ ] Batch processing works correctly
- [ ] Cache reduces API calls by >80%
- [ ] Rate limiting enforced
- [ ] All unit tests pass (>90% coverage)
- [ ] Integration tests pass
- [ ] Performance benchmarks met

---

### Story 3.2: LLM Service Adapter

**Implementation Steps:**
1. Define base LLM provider interface
2. Implement Anthropic Claude provider
3. Implement OpenAI provider
4. Implement Ollama local provider
5. Build LLM service with rate limiting
6. Implement prompt template system
7. Implement token counting
8. Add streaming support
9. Write unit tests for all components
10. Write integration tests
11. Run performance benchmarks

**Testing:**
```bash
# Unit tests
pytest tests/unit/services/llm/ -v

# Integration tests (requires API keys)
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
pytest tests/integration/services/test_llm_integration.py -v

# Performance benchmarks
pytest tests/benchmarks/test_llm_performance.py -v
```

**Verification:**
- [ ] All providers support completion
- [ ] Streaming works for all providers
- [ ] Prompt templates with variable substitution
- [ ] Token counting accurate
- [ ] Cost tracking working
- [ ] All unit tests pass (>90% coverage)
- [ ] Integration tests pass
- [ ] Performance benchmarks met

---

## Testing Strategy

### Test Pyramid

```
         /\
        /  \    Integration Tests (15%)
       /----\   - Real API calls
      /      \  - Full workflows
     /--------\  - Multi-component
    /          \
   /------------\ Unit Tests (85%)
  /______________\ - Mocked APIs
                  - Individual components
                  - Fast execution
```

### Mocking Strategy

For unit tests, mock external APIs:

```python
@patch("openai.OpenAI")
def test_openai_provider(mock_openai):
    mock_client = Mock()
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1, 0.2])]
    mock_client.embeddings.create.return_value = mock_response
    mock_openai.return_value = mock_client

    provider = OpenAIProvider(config)
    result = provider.get_embeddings(["test"])
    assert len(result.embeddings) == 1
```

### Integration Test Flags

Use pytest markers for integration tests:

```bash
# Run only unit tests (fast)
pytest -v -m "not integration"

# Run only integration tests
pytest -v -m integration

# Skip tests requiring API keys
pytest -v -m "not integration or skipif"
```

---

## Configuration Examples

### Embedding Service Config

```yaml
# config.yaml
embedding_service:
  provider: "openai"
  model: "text-embedding-3-small"

  provider_config:
    api_key: "${OPENAI_API_KEY}"
    max_batch_size: 100

  cache:
    enabled: true
    max_size: 10000
    ttl: 3600  # 1 hour

  rate_limiting:
    enabled: true
    requests_per_minute: 3000
```

### LLM Service Config

```yaml
# config.yaml
llm_service:
  provider: "anthropic"
  model: "claude-sonnet-4.5"

  provider_config:
    api_key: "${ANTHROPIC_API_KEY}"

  rate_limiting:
    enabled: true
    requests_per_minute: 50
```

---

## Cost Management

### Embedding Costs

| Provider | Model | Cost per 1M tokens | Dimensions |
|----------|-------|-------------------|------------|
| OpenAI | text-embedding-3-small | $0.02 | 1536 |
| OpenAI | text-embedding-3-large | $0.13 | 3072 |
| Cohere | embed-english-v3.0 | $0.10 | 1024 |
| Local | all-MiniLM-L6-v2 | $0 | 384 |

**Cost Reduction Strategies:**
- Use caching (>80% hit rate = 80% cost reduction)
- Use local models for development
- Batch requests when possible
- Use smaller models when sufficient

### LLM Costs

| Provider | Model | Cost per 1M prompt tokens | Cost per 1M completion tokens |
|----------|-------|---------------------------|------------------------------|
| Anthropic | Claude Sonnet 4.5 | $3.00 | $15.00 |
| Anthropic | Claude Opus | $15.00 | $75.00 |
| Anthropic | Claude Haiku | $0.25 | $1.25 |
| OpenAI | GPT-4 Turbo | $10.00 | $30.00 |
| OpenAI | GPT-3.5 Turbo | $0.50 | $1.50 |
| Ollama | llama2 (local) | $0 | $0 |

**Cost Reduction Strategies:**
- Use Haiku for simple tasks
- Use local models for development
- Set max_tokens to limit completions
- Monitor costs with built-in tracking

---

## Performance Requirements

### Embedding Service

| Operation | Target | Notes |
|-----------|--------|-------|
| Single embedding | <100ms | Excluding API latency |
| Batch processing | >100 texts/sec | Local models |
| Cache lookup | <10ms | In-memory cache |
| Concurrent requests | 1000+ | Thread-safe |

### LLM Service

| Operation | Target | Notes |
|-----------|--------|-------|
| Request overhead | <50ms | Excluding API latency |
| Streaming latency | <100ms | To first token |
| Concurrent requests | 100+ | Thread-safe |
| Token counting | <10ms | Cached per model |

---

## Dependencies

### External Services
- OpenAI API (optional)
- Anthropic API (optional)
- Cohere API (optional)
- Ollama (optional for local models)

### Python Libraries
```
openai==1.12.0
anthropic==0.18.1
cohere==4.47
sentence-transformers==2.3.1
tiktoken==0.5.2
tenacity==8.2.3
httpx==0.26.0
redis==5.0.1  # Optional cache backend
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     RAG Strategies                      │
│         (Will use these services in future epics)       │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐       ┌───────▼────────┐
│ Embedding      │       │ LLM Service    │
│ Service        │       │                │
├────────────────┤       ├────────────────┤
│ • Caching      │       │ • Prompt       │
│ • Rate Limit   │       │   Templates    │
│ • Batch        │       │ • Streaming    │
│ • Multi-       │       │ • Cost Track   │
│   Provider     │       │ • Multi-       │
│                │       │   Provider     │
└────────┬───────┘       └───────┬────────┘
         │                       │
    ┌────┴────┐             ┌────┴────┐
    │ OpenAI  │             │Anthropic│
    │ Cohere  │             │ OpenAI  │
    │ Local   │             │ Ollama  │
    └─────────┘             └─────────┘
```

---

## Story Dependencies

```
Epic 2: Database & Storage
    ↓
Story 3.1: Embedding Service ──┐
                               ├──→ Future RAG Strategies
Story 3.2: LLM Service ────────┘
```

**Note:** Stories 3.1 and 3.2 can be developed in parallel.

---

## Usage Examples

### Embedding Service

```python
from rag_factory.services.embedding import EmbeddingService, EmbeddingServiceConfig

# Initialize
config = EmbeddingServiceConfig(
    provider="openai",
    model="text-embedding-3-small",
    enable_cache=True
)
service = EmbeddingService(config)

# Generate embeddings
texts = ["Hello world", "Another text"]
result = service.embed(texts)

print(f"Dimensions: {result.dimensions}")
print(f"Cost: ${result.cost:.6f}")
print(f"Cached: {result.cached}")

# Check stats
stats = service.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

### LLM Service

```python
from rag_factory.services.llm import LLMService, LLMServiceConfig
from rag_factory.services.llm.base import Message, MessageRole
from rag_factory.services.llm.prompt_template import CommonTemplates

# Initialize
config = LLMServiceConfig(
    provider="anthropic",
    model="claude-sonnet-4.5"
)
service = LLMService(config)

# Simple completion
messages = [Message(role=MessageRole.USER, content="What is AI?")]
response = service.complete(messages)
print(response.content)

# With template
messages = CommonTemplates.RAG_QA.format(
    context="AI is artificial intelligence...",
    question="What is AI?"
)
response = service.complete(messages)
print(response.content)

# Streaming
for chunk in service.stream(messages):
    if not chunk.is_final:
        print(chunk.content, end="")

# Check stats
stats = service.get_stats()
print(f"Total cost: ${stats['total_cost']:.4f}")
```

---

## Success Criteria for Epic 3

Epic 3 is complete when:

- [ ] **Story 3.1: Embedding Service (8 points)**
  - All providers implemented (OpenAI, Cohere, Local)
  - Caching working with >80% hit rate
  - Batch processing efficient
  - All tests passing

- [ ] **Story 3.2: LLM Service (8 points)**
  - All providers implemented (Anthropic, OpenAI, Ollama)
  - Streaming support working
  - Prompt templates functional
  - Cost tracking accurate
  - All tests passing

- [ ] **Integration**
  - Both services work together
  - Configuration system complete
  - Cost tracking consolidated
  - Documentation complete

---

## Next Steps

### For Product Owner
1. Review story completeness
2. Verify acceptance criteria
3. Confirm story points (8 + 8 = 16)
4. Add to Sprint 2 backlog
5. Assign to developers

### For Developers
1. Read story documentation thoroughly
2. Set up API keys and Ollama
3. Create feature branches
4. Implement following TDD approach
5. Run tests continuously
6. Complete Definition of Done
7. Submit PRs for review

---

## Notes for Future Epics

These services will be used by:
- **Epic 4:** Priority Strategies (Naive RAG, Parent-Child, etc.)
- **Epic 5:** Agentic & Advanced Strategies
- **Epic 6:** Multi-Query & Contextual Strategies
- **Epic 7:** Experimental Strategies

All strategies will:
- Use `EmbeddingService` for generating embeddings
- Use `LLMService` for completions
- Benefit from caching and cost tracking
- Support multiple providers without code changes

---

**Epic Total:** 16 Story Points
**Status:** ✅ READY FOR DEVELOPMENT
**Sprint:** Sprint 2
**Dependencies:** Epic 2 (database)
