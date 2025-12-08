# Epic 10: Lightweight Dependencies Implementation

**Epic Goal:** Implement all strategies using lightweight dependencies (ONNX, tiktoken) instead of heavy ML frameworks (torch, transformers) to reduce installation size, eliminate CUDA requirements, and improve deployment flexibility.

**Epic Story Points Total:** 34

**Dependencies:** Epic 7 (experimental strategies), Epic 4 (reranking)

**Note:** This epic refactors the implementation approach for strategies that currently have torch/transformers imports. PyTorch was never fully implemented or tested - we're choosing the lightweight path from the start.

---

## Story 10.1: Migrate Embedding Services to ONNX

**As a** developer
**I want** embedding services to use ONNX runtime instead of PyTorch
**So that** I can deploy without heavy ML dependencies

**Acceptance Criteria:**
- Update `onnx_local.py` provider to be the primary local embedding option
- Remove torch dependency from embedding providers
- Support ONNX model loading from HuggingFace Hub
- Maintain embedding quality and performance
- Update documentation for ONNX model usage
- Provide model conversion guide (PyTorch → ONNX)

**Technical Dependencies:**
- `onnx>=1.15.0` (already in requirements)
- `onnxruntime>=1.16.3` (already in requirements)
- `optimum` for model conversion

**Story Points:** 8

---

## Story 10.2: Replace Tokenization with Tiktoken

**As a** developer
**I want** to use tiktoken for tokenization
**So that** I avoid the heavy transformers library dependency

**Acceptance Criteria:**
- Replace transformers tokenizers with tiktoken
- Support multiple tiktoken encodings (cl100k_base, p50k_base, etc.)
- Maintain token counting accuracy
- Update all strategies using tokenization
- Fallback to basic tokenization if tiktoken unavailable
- Update documentation with tokenization options

**Technical Dependencies:**
- `tiktoken>=0.5.2` (already in requirements)

**Story Points:** 5

---

## Story 10.3: Migrate Late Chunking to ONNX

**As a** developer
**I want** late chunking strategy to use ONNX models
**So that** this experimental feature doesn't require PyTorch

**Acceptance Criteria:**
- Update `DocumentEmbedder` to use ONNX runtime
- Replace transformers tokenizer with tiktoken
- Maintain token-level embedding extraction
- Support long-context ONNX models
- Update tests to use ONNX models
- Document ONNX model requirements for late chunking

**Technical Dependencies:**
- ONNX runtime with token-level output support
- Compatible ONNX embedding models

**Story Points:** 8

---

## Story 10.4: Migrate Reranking to Lightweight Alternatives

**As a** developer
**I want** reranking without PyTorch dependencies
**So that** I can use reranking in lightweight deployments

**Acceptance Criteria:**
- Implement Cohere reranking as primary option
- Implement cosine similarity reranker (no external deps)
- Make torch-based rerankers (BGE, Cross-Encoder) optional
- Add runtime checks with helpful error messages
- Update reranking strategy to auto-select available reranker
- Update documentation with reranking options

**Technical Dependencies:**
- `cohere>=4.47` (already in requirements)
- `numpy>=1.24.0` (already in requirements)

**Story Points:** 8

---

## Story 10.5: Migrate Fine-Tuned Embeddings to ONNX

**As a** developer
**I want** custom embedding models in ONNX format
**So that** fine-tuned models don't require PyTorch

**Acceptance Criteria:**
- Update `CustomModelLoader` to prioritize ONNX format
- Support ONNX model registry
- Update A/B testing framework for ONNX models
- Provide conversion tools (PyTorch → ONNX)
- Update model versioning for ONNX models
- Document ONNX model training and conversion workflow

**Technical Dependencies:**
- ONNX runtime
- Model conversion utilities

**Story Points:** 5

---

## Sprint Planning

**Sprint 12:** Stories 10.1, 10.2 (13 points)
**Sprint 13:** Stories 10.3, 10.4, 10.5 (21 points)

---

## Implementation Strategy

### Phase 1: Refactor to ONNX (Current)
- Replace torch/transformers imports with ONNX runtime
- Implement ONNX-based embedders and rerankers
- Use tiktoken for tokenization
- Remove torch from requirements.txt

### Phase 2: Testing & Validation
- Update all tests to use ONNX models
- Validate performance and quality
- Ensure all strategies work without torch

### Phase 3: Documentation
- Document ONNX model usage
- Provide model conversion guides (for users bringing their own models)
- Update all examples to use lightweight dependencies

---

## Dependency Strategy

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

## Technical Stack

**Embeddings:**
- ONNX Runtime for model inference
- Optimum for model conversion
- HuggingFace Hub for ONNX model distribution

**Tokenization:**
- tiktoken for OpenAI-compatible tokenization
- Fallback to basic regex tokenization

**Reranking:**
- Cohere API for cloud-based reranking
- Cosine similarity for local reranking
- Optional: ONNX-based reranking models

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
- [ ] Model conversion guide published (for custom models)
- [ ] Performance meets targets (see Performance Targets section)
- [ ] Works on all platforms (Linux, macOS, Windows)

---

## Performance Targets

**Embedding Service:**
- Embedding speed: <100ms per document (CPU)
- Memory usage: <500MB for model + inference
- Model load time: <5 seconds
- Installation time: <2 minutes

**Reranking Performance:**
- Cohere API: <200ms per batch
- Cosine similarity: <10ms per batch

**Overall System:**
- Query latency: <2 seconds (p95)
- Memory footprint: <1GB total
- Cold start time: <10 seconds

---

## Risk Assessment

**Low Risk:**
- ONNX is mature and well-supported
- tiktoken is production-ready (used by OpenAI)
- Cohere reranking is enterprise-grade

**Medium Risk:**
- ONNX model availability for niche use cases
- Users may want to bring custom PyTorch models

**Mitigation:**
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
