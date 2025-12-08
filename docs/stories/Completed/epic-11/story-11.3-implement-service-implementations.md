# Story 11.3: Implement Service Implementations

**Story ID:** 11.3
**Epic:** Epic 11 - Dependency Injection & Service Interface Decoupling
**Story Points:** 13
**Priority:** High
**Dependencies:** Story 11.1

---

## User Story

**As a** developer
**I want** concrete implementations of all service interfaces
**So that** strategies can use real services

---

## Detailed Requirements

### Functional Requirements

1.  **ONNX Service Implementations (Local/Testing)**
    - `ONNXEmbeddingService`: Wraps ONNX Runtime for embeddings
    - `ONNXLLMService`: Wraps ONNX Runtime for text generation (GenAI)
    - Must implement `IEmbeddingService` and `ILLMService` respectively
    - Support loading models from local paths

2.  **API Service Implementations (Production)**
    - `AnthropicLLMService`: Wraps Anthropic API (Claude)
    - `OpenAILLMService`: Wraps OpenAI API (GPT)
    - `OpenAIEmbeddingService`: Wraps OpenAI Embeddings API
    - `CohereRerankingService`: Wraps Cohere Rerank API
    - Must handle API keys and configuration

3.  **Database Service Implementations**
    - `Neo4jGraphService`: Wraps Neo4j driver
    - `PostgresqlDatabaseService`: Wraps PostgreSQL with pgvector
    - `CosineRerankingService`: Local reranking using cosine similarity (numpy)

### Non-Functional Requirements

1.  **Code Quality**
    - All implementations must pass interface type checks
    - Error handling for API failures and connection issues
    - Proper resource management (closing connections/sessions)

2.  **Configuration**
    - Services should accept configuration via constructor (API keys, paths, etc.)
    - Support environment variables for secrets

---

## Acceptance Criteria

### AC1: ONNX Services
- [ ] `ONNXEmbeddingService` implements `IEmbeddingService`
- [ ] `ONNXLLMService` implements `ILLMService`
- [ ] Both work with local ONNX models

### AC2: LLM API Services
- [ ] `AnthropicLLMService` implements `ILLMService`
- [ ] `OpenAILLMService` implements `ILLMService`
- [ ] Support streaming and non-streaming

### AC3: Reranking Services
- [ ] `CohereRerankingService` implements `IRerankingService`
- [ ] `CosineRerankingService` implements `IRerankingService`

### AC4: Database Services
- [ ] `Neo4jGraphService` implements `IGraphService`
- [ ] `PostgresqlDatabaseService` implements `IDatabaseService`

### AC5: Testing
- [ ] Integration tests for each service (mocked external calls)
- [ ] Verify interface compliance

---

## Technical Specifications

### File Structure
```
rag_factory/
├── services/
│   ├── __init__.py
│   ├── onnx/
│   │   ├── __init__.py
│   │   ├── embedding.py
│   │   └── llm.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── anthropic.py
│   │   ├── openai.py
│   │   └── cohere.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── neo4j.py
│   │   └── postgres.py
│   └── local/
│       ├── __init__.py
│       └── reranker.py
```

### Implementation Snippets

**Anthropic LLM Service:**
```python
class AnthropicLLMService(ILLMService):
    """Anthropic Claude API service"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
    
    async def complete(self, prompt: str, **kwargs) -> str:
        response = await self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text
```

**ONNX Embedding Service:**
```python
class ONNXEmbeddingService(IEmbeddingService):
    """ONNX-based local embedding service"""
    
    def __init__(self, model_path: str, tokenizer_path: str):
        # Initialize ONNX session and tokenizer
        pass
    
    async def embed(self, text: str) -> List[float]:
        # Tokenize and run inference
        pass
```
