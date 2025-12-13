# Product Backlog

This document tracks future improvements and features that are planned but not yet scheduled for implementation.

---

## Epic: Embedding Service Architecture Improvements

### Story: Implement Embedding Provider Interface

**Priority:** Medium  
**Story Points:** 13  
**Status:** Backlog  
**Created:** 2025-12-12

#### User Story

**As a** developer  
**I want** to configure embeddings as an interface instead of using a single implementation  
**So that** I can easily switch between different embedding providers (ONNX, OpenAI, Cohere, etc.) without code changes

#### Background

Currently, the embedding service uses ONNX as the primary local implementation (Epic 10). While this works well for lightweight deployments, there are scenarios where different embedding providers might be needed:

- **Local ONNX**: Lightweight, no external dependencies (current default)
- **OpenAI Embeddings**: High quality, cloud-based
- **Cohere Embeddings**: Multilingual support
- **Custom Fine-tuned Models**: Domain-specific embeddings
- **Hybrid Approaches**: Combine multiple providers

The fine-tuned embeddings infrastructure (Epic 7) attempted to support multiple formats, but it's tightly coupled to specific implementations rather than using a clean interface pattern.

#### Proposed Solution

Create a clean embedding provider interface that allows:

1. **Interface Definition**
   - Define `IEmbeddingProvider` interface with standard methods
   - `embed_documents(texts: List[str]) -> List[List[float]]`
   - `embed_query(text: str) -> List[float]`
   - `get_dimension() -> int`
   - `get_metadata() -> Dict[str, Any]`

2. **Provider Implementations**
   - `ONNXEmbeddingProvider` (default, lightweight)
   - `OpenAIEmbeddingProvider` (cloud-based)
   - `CohereEmbeddingProvider` (cloud-based)
   - `FineTunedEmbeddingProvider` (custom models)
   - `HybridEmbeddingProvider` (combines multiple providers)

3. **Configuration-Based Selection**
   ```yaml
   embedding:
     provider: onnx  # or openai, cohere, fine-tuned, hybrid
     config:
       model_name: Xenova/all-mpnet-base-v2
       cache_dir: ./models
   ```

4. **Factory Pattern**
   - `EmbeddingProviderFactory` to instantiate providers based on config
   - Lazy loading to avoid importing unused dependencies
   - Graceful fallback if preferred provider unavailable

#### Benefits

- **Flexibility**: Easy to switch providers via configuration
- **Testability**: Mock providers for testing
- **Extensibility**: Add new providers without changing core code
- **Dependency Management**: Only load dependencies for active provider
- **Multi-Provider Support**: Use different providers for different use cases

#### Technical Considerations

- Maintain backward compatibility with current ONNX-only setup
- Keep ONNX as default to preserve lightweight architecture
- Use dependency injection for provider selection
- Implement lazy imports to avoid loading unused dependencies
- Ensure consistent embedding dimensions across providers

#### Acceptance Criteria

- [ ] `IEmbeddingProvider` interface defined
- [ ] At least 3 provider implementations (ONNX, OpenAI, Cohere)
- [ ] Configuration-based provider selection working
- [ ] Factory pattern implemented with lazy loading
- [ ] All existing tests pass with ONNX provider
- [ ] New tests for provider switching
- [ ] Documentation for adding custom providers
- [ ] Migration guide for existing code

#### Dependencies

- Epic 10 (ONNX migration) - ✅ Complete
- Epic 7 (Fine-tuned embeddings) - Partial (needs refactoring)

#### Related Issues

- Fine-tuned embeddings currently use `ModelFormat` enum instead of interface
- Some code still has sentence-transformers imports for legacy support
- Need to deprecate multi-format loader in favor of interface pattern

#### Future Enhancements

- **Embedding Caching**: Cache embeddings across providers
- **Provider Metrics**: Track performance per provider
- **Auto-Selection**: Automatically choose best provider based on criteria
- **Fallback Chain**: Try multiple providers in sequence if one fails
- **Cost Optimization**: Route to cheapest provider that meets quality threshold

---

## Other Backlog Items

### Story: Improve Test Isolation in Repository Tests

**Priority:** High  
**Story Points:** 5  
**Status:** Backlog

Fix test isolation issues in `test_repository_integration.py` causing duplicate entities and incorrect counts.

---

### Story: Implement Pytest Markers for Test Organization

**Priority:** Medium  
**Story Points:** 3  
**Status:** Backlog

Add markers like `@pytest.mark.requires_llm_api`, `@pytest.mark.requires_database` for better test organization and CI/CD.

---

### Story: Performance Test Separation

**Priority:** Low  
**Story Points:** 2  
**Status:** Backlog

Separate performance benchmarks from functional tests to avoid false failures on slower systems.

---

**Last Updated:** 2025-12-12
