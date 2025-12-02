# Epic 4: Priority RAG Strategies (High Impact)

**Epic Goal:** Implement the three highest-impact strategies recommended in the video: Re-ranking, Context-Aware Chunking, and Query Expansion.

**Epic Story Points Total:** 34

**Dependencies:** Epic 3 (requires Embedding and LLM services)

---

## Story 4.1: Implement Context-Aware Chunking Strategy

**As a** system
**I want** to split documents at natural boundaries
**So that** document structure is preserved

**Acceptance Criteria:**
- Use embedding model to find semantic boundaries
- Respect document structure (sections, paragraphs)
- Configurable chunk size ranges
- Maintain metadata about document structure
- Support multiple chunking approaches (hybrid via dockling)
- Integration tests with sample documents

**Technical Dependencies:**
- dockling library or similar
- Embedding model

**Story Points:** 13

---

## Story 4.2: Implement Re-ranking Strategy

**As a** system
**I want** a two-step retrieval with re-ranking
**So that** I can retrieve many chunks but return only the most relevant

**Acceptance Criteria:**
- Retrieve large number of chunks (configurable, e.g., 50-100)
- Integrate cross-encoder model for re-ranking
- Return top-k re-ranked results (configurable, e.g., 5-10)
- Measure and log re-ranking scores
- Support multiple re-ranker models (Cohere, BGE, etc.)
- Performance benchmarks

**Technical Dependencies:**
- Re-ranking model integration (e.g., sentence-transformers)

**Story Points:** 13

---

## Story 4.3: Implement Query Expansion Strategy

**As a** system
**I want** to expand user queries with more specific details
**So that** search precision improves

**Acceptance Criteria:**
- Use LLM to expand query with relevant details
- Configurable expansion instructions
- Return both original and expanded queries
- Use expanded query for search
- Log expansion results for debugging
- A/B testing capability

**Technical Dependencies:**
- LLM integration

**Story Points:** 8

---

## Sprint Planning

**Sprint 3:** Stories 4.1, 4.2 (26 points)
**Sprint 4:** Story 4.3 (8 points) + Epic 8 Story 8.1

---

## Priority Implementation Path (MVP)

These three strategies are the **highest priority** for MVP:
1. Context-Aware Chunking (4.1)
2. Re-ranking (4.2)
3. Query Expansion (4.3)

This is the recommended "High-Impact Trio" that provides the best results.

---

## Technical Stack

**Chunking:**
- dockling (hybrid chunking)
- Custom semantic boundary detection

**Re-ranking:**
- sentence-transformers (cross-encoders)
- Cohere rerank API
- BGE reranker models

**Query Expansion:**
- LLM service (Anthropic Claude, OpenAI GPT-4)

---

## Success Criteria

- [ ] Context-aware chunking produces semantically coherent chunks
- [ ] Re-ranking improves relevance over baseline retrieval
- [ ] Query expansion improves search precision
- [ ] All three strategies can be combined in a pipeline
- [ ] Performance metrics tracked for each strategy
- [ ] Integration tests passing
- [ ] Can demonstrate improvement over naive chunking + retrieval
