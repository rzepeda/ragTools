# Epic 3: Core Services Layer

**Epic Goal:** Build the foundational services (Embedding Service and LLM Service) that all RAG strategies will depend on.

**Epic Story Points Total:** 16

**Dependencies:** Epic 2 (database must be ready)

---

## Story 3.1: Build Embedding Service

**As a** system
**I want** a centralized embedding service
**So that** all strategies use consistent embeddings

**Acceptance Criteria:**
- Support multiple embedding models (OpenAI, Cohere, local models)
- Batch embedding for efficiency
- Caching layer for repeated texts
- Rate limiting and retry logic
- Model switching without code changes
- Unit tests with mock embeddings

**Story Points:** 8

---

## Story 3.2: Implement LLM Service Adapter

**As a** system
**I want** a unified interface for LLM calls
**So that** strategies can use different LLM providers

**Acceptance Criteria:**
- Support multiple providers (Anthropic, OpenAI, etc.)
- Consistent prompt templates
- Token counting and cost tracking
- Rate limiting and retries
- Streaming support
- Unit tests with mock LLM responses

**Story Points:** 8

---

## Sprint Planning

This epic is recommended for **Sprint 2**.

**Total Sprint 2:** 16 points (Epic 3)

---

## Technical Stack

**Embedding Models:**
- OpenAI text-embedding-3-small/large
- Cohere embed-multilingual-v3
- Local: sentence-transformers

**LLM Providers:**
- Anthropic Claude (Sonnet 4.5)
- OpenAI GPT-4
- Local models via Ollama

**Python Libraries:**
- openai
- anthropic
- cohere
- sentence-transformers

---

## Success Criteria

- [ ] Embedding service can generate embeddings from text
- [ ] Multiple embedding models supported
- [ ] Embedding caching works
- [ ] LLM service can make calls to at least one provider
- [ ] Multiple LLM providers supported
- [ ] Cost tracking implemented
- [ ] All service tests passing
- [ ] Services can be configured via config file
