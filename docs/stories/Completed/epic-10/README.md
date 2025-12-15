# Epic 10: Lightweight Dependencies Implementation

**Epic Goal:** Implement all strategies using lightweight dependencies (ONNX, tiktoken) instead of heavy ML frameworks (torch, transformers) to reduce installation size, eliminate CUDA requirements, and improve deployment flexibility.

**Epic Story Points Total:** 34

**Status:** Ready for Implementation

---

## Overview

This epic refactors the RAG Factory implementation to use lightweight dependencies (ONNX, tiktoken) for improved deployment flexibility.

### Key Benefits

- **Smaller Installation:** ~235MB total
- **Faster Setup:** <2 minutes
- **No CUDA Required:** Works on any CPU
- **Universal Compatibility:** Linux, macOS, Windows, x86, ARM
- **High Quality:** Production-ready embeddings and retrieval

---

## Stories

### [Story 10.1: Migrate Embedding Services to ONNX](story-10.1-migrate-embedding-services-to-onnx.md)
**Story Points:** 8 | **Priority:** High

Migrate embedding services to ONNX Runtime for CPU-optimized inference with lightweight dependencies.

**Key Deliverables:**
- ONNX-based embedding provider
- Model conversion utilities
- HuggingFace Hub integration
- Performance optimization

**Acceptance Criteria:**
- ONNX embeddings working correctly
- Embedding speed <100ms per document
- All tests passing

---

### [Story 10.2: Replace Tokenization with Tiktoken](story-10.2-replace-tokenization-with-tiktoken.md)
**Story Points:** 5 | **Priority:** High

Replace transformers tokenizers with tiktoken for fast, lightweight tokenization.

**Key Deliverables:**
- Tiktoken integration
- Token counting utilities
- Token budget tracking
- Fallback tokenization

**Acceptance Criteria:**
- All tokenization using tiktoken
- Token counts match OpenAI's (for OpenAI models)
- Transformers dependency removed
- All tests passing

---

### [Story 10.3: Migrate Late Chunking to ONNX](story-10.3-migrate-late-chunking-to-onnx.md)
**Story Points:** 8 | **Priority:** Medium

Update Late Chunking strategy to use ONNX models and tiktoken tokenization.

**Key Deliverables:**
- ONNX-based document embedder
- Tiktoken tokenization
- Token-level embedding extraction
- Long-context support

**Acceptance Criteria:**
- Late chunking works with ONNX
- Embeddings working correctly
- Performance targets met (<500ms for 2048 tokens)
- All features preserved

**Dependencies:** Stories 10.1, 10.2

---

### [Story 10.4: Migrate Reranking to Lightweight Alternatives](story-10.4-migrate-reranking-to-lightweight-alternatives.md)
**Story Points:** 8 | **Priority:** High

Implement lightweight reranking options (Cohere API, cosine similarity) with optional advanced rerankers.

**Key Deliverables:**
- Cohere reranking (primary)
- Cosine similarity reranker (fallback)
- Auto-selection logic
- Optional advanced rerankers

**Acceptance Criteria:**
- Reranking works with lightweight dependencies
- Cohere reranker functional
- Cosine reranker functional
- Auto-selection working
- Quality meets requirements

**Dependencies:** Story 10.1

---

### [Story 10.5: Migrate Fine-Tuned Embeddings to ONNX](story-10.5-migrate-fine-tuned-embeddings-to-onnx.md)
**Story Points:** 5 | **Priority:** Medium

Update fine-tuned embeddings infrastructure to support ONNX models.

**Key Deliverables:**
- ONNX model registry
- Model conversion tools
- A/B testing with ONNX
- Model versioning

**Acceptance Criteria:**
- ONNX models in registry
- Conversion tools working
- A/B testing functional
- Quality maintained

**Dependencies:** Story 10.1

---

## Sprint Planning

### Sprint 12 (13 points)
- Story 10.1: Migrate Embedding Services to ONNX (8 points)
- Story 10.2: Replace Tokenization with Tiktoken (5 points)

### Sprint 13 (21 points)
- Story 10.3: Migrate Late Chunking to ONNX (8 points)
- Story 10.4: Migrate Reranking to Lightweight Alternatives (8 points)
- Story 10.5: Migrate Fine-Tuned Embeddings to ONNX (5 points)

---

## Technical Stack

### Lightweight Dependencies
```
onnx>=1.15.0                    # ~15MB
onnxruntime>=1.16.3             # ~200MB (CPU-optimized)
tiktoken>=0.5.2                 # ~5MB
cohere>=4.47                    # API client (minimal)
numpy>=1.24.0                   # ~15MB (already required)
```
**Total:** ~235MB, no CUDA required, works on any CPU

---

## Implementation Strategy

### Phase 1: Implement ONNX Support (Sprint 12)
- Implement ONNX-based embedders
- Use tiktoken for tokenization
- Update dependencies

### Phase 2: Update Strategies (Sprint 13)
- Migrate late chunking to ONNX
- Implement lightweight reranking
- Update fine-tuned embeddings
- Ensure all strategies work with lightweight dependencies

### Phase 3: Testing & Validation
- Update all tests to use ONNX models
- Validate performance and quality
- Ensure all strategies work correctly
- Cross-platform testing

### Phase 4: Documentation
- Document ONNX model usage
- Provide model conversion guides
- Update all examples
- Create deployment guides

---

## Performance Targets

### Embedding Service
- Embedding speed: <100ms per document (CPU)
- Memory usage: <500MB for model + inference
- Model load time: <5 seconds
- Installation time: <2 minutes

### Reranking Performance
- Cohere API: <200ms per batch
- Cosine similarity: <10ms per batch

### Overall System
- Query latency: <2 seconds (p95)
- Memory footprint: <1GB total
- Cold start time: <10 seconds

---

## Success Criteria

- [ ] All embedding services support ONNX models
- [ ] Tokenization uses tiktoken by default
- [ ] Late chunking works with ONNX runtime
- [ ] Reranking available (Cohere + cosine)
- [ ] Fine-tuned embeddings support ONNX format
- [ ] All tests pass with lightweight dependencies only
- [ ] Installation size <300MB total
- [ ] No CUDA dependencies required
- [ ] Documentation updated with lightweight deployment guide
- [ ] Model conversion guide published
- [ ] Performance meets targets
- [ ] Works on all platforms (Linux, macOS, Windows)

---

## Risk Assessment

### Low Risk
- ONNX is mature and well-supported
- tiktoken is production-ready (used by OpenAI)
- Cohere reranking is enterprise-grade

### Medium Risk
- ONNX model availability for niche use cases
- Users may want to bring custom models

### Mitigation
- Provide model conversion tools and guides
- Support multiple ONNX model sources (HuggingFace, local)
- Document how to convert custom models to ONNX
- Comprehensive testing with common models

---

## Documentation Updates Required

- [ ] ONNX model conversion guide
- [ ] Lightweight deployment guide
- [ ] Reranking options comparison
- [ ] Tokenization strategy guide
- [ ] Docker images for lightweight deployment
- [ ] Troubleshooting guide for ONNX issues

---

## Dependencies

**Depends on:** Epic 7 (experimental strategies), Epic 4 (reranking)

**Blocks:** None

---

## Story Dependency Graph

```
10.1 (ONNX Embeddings) ──┬──> 10.3 (Late Chunking)
                         ├──> 10.4 (Reranking)
                         └──> 10.5 (Fine-Tuned)

10.2 (Tiktoken) ─────────┴──> 10.3 (Late Chunking)
```

---

## Related Documentation

- <!-- BROKEN LINK: Epic 10 Definition <!-- (broken link to: ../../epics/epic-10-lightweight-dependencies.md) --> --> Epic 10 Definition
- <!-- BROKEN LINK: Model Conversion Guide <!-- (broken link to: ../../guides/onnx-conversion.md) --> --> Model Conversion Guide (to be created)
- <!-- BROKEN LINK: Lightweight Deployment Guide <!-- (broken link to: ../../guides/lightweight-deployment.md) --> --> Lightweight Deployment Guide (to be created)

---

## Notes

- This epic represents a strategic decision to prioritize deployment flexibility
- Focus on lightweight dependencies for universal compatibility
- All quality targets validated through comprehensive testing
