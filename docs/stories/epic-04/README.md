# Epic 4: Priority RAG Strategies

## Overview

Epic 4 implements the three highest-impact RAG strategies recommended for production systems:
1. **Context-Aware Chunking** - Split documents intelligently
2. **Re-ranking** - Improve retrieval relevance
3. **Query Expansion** - Enhance search precision and recall

These three strategies form the "High-Impact Trio" that provides the best results when combined.

## Epic Goal

Implement the three highest-impact strategies to significantly improve RAG system quality and relevance.

## Epic Story Points

**Total: 34 points**
- Story 4.1: Context-Aware Chunking (13 points)
- Story 4.2: Re-ranking Strategy (13 points)
- Story 4.3: Query Expansion (8 points)

## Dependencies

- **Epic 2**: Vector Database (required for initial retrieval)
- **Epic 3**: Embedding Service (required for semantic chunking and vector search)
- **Epic 3**: LLM Service (required for query expansion)

## Stories

### Story 4.1: Context-Aware Chunking Strategy (13 points)

**Status**: Not Started

**Objective**: Split documents at natural boundaries preserving semantic coherence

**Key Features**:
- Semantic boundary detection using embeddings
- Document structure preservation (headers, paragraphs, code blocks)
- Hybrid chunking with dockling library
- Configurable chunk size ranges
- Metadata tracking
- Multiple chunking strategies (semantic, structural, hybrid, fixed-size)

**Technical Components**:
- Base chunker interface
- Semantic chunker (embedding-based)
- Structural chunker (document structure)
- Hybrid chunker (combines both)
- Dockling integration for advanced parsing
- Chunk quality metrics

**Acceptance Criteria**:
- ✅ Semantic boundary detection working
- ✅ Document structure preserved
- ✅ Dockling integration complete
- ✅ Multiple strategies implemented
- ✅ Performance: >100 chunks/second
- ✅ All tests passing

### Story 4.2: Re-ranking Strategy (13 points)

**Status**: Not Started

**Objective**: Two-step retrieval with re-ranking for improved relevance

**Key Features**:
- Broad initial retrieval (50-100 candidates)
- Cross-encoder re-ranking
- Multiple re-ranker models (sentence-transformers, Cohere, BGE)
- Score normalization
- Performance optimization (batching, caching)
- Ranking metrics (NDCG, MRR)

**Technical Components**:
- Base re-ranker interface
- Re-ranker service
- Cross-encoder implementation
- Cohere Rerank API integration
- BGE reranker support
- Re-ranking cache
- Fallback strategies

**Acceptance Criteria**:
- ✅ Two-step retrieval working
- ✅ Multiple models supported
- ✅ Score normalization implemented
- ✅ Performance: <2 seconds for 100 candidates
- ✅ Metrics tracking (NDCG, MRR)
- ✅ All tests passing

### Story 4.3: Query Expansion Strategy (8 points)

**Status**: Not Started

**Objective**: Expand user queries for improved search precision

**Key Features**:
- LLM-based query expansion
- Multiple expansion techniques (keywords, reformulation, HyDE)
- Configurable expansion prompts
- Expansion caching
- A/B testing support
- Original + expanded query tracking

**Technical Components**:
- Base query expander interface
- LLM-based expander
- Keyword expansion
- Query reformulation
- Multi-query generation
- HyDE (Hypothetical Document Expansion)
- Expansion cache
- A/B testing framework

**Acceptance Criteria**:
- ✅ LLM expansion working
- ✅ Multiple techniques implemented
- ✅ Configurable prompts
- ✅ Performance: <1 second
- ✅ A/B testing framework working
- ✅ All tests passing

## Sprint Planning

### Sprint 3 (26 points)
- Story 4.1: Context-Aware Chunking (13 points)
- Story 4.2: Re-ranking Strategy (13 points)

### Sprint 4 (8 points)
- Story 4.3: Query Expansion (8 points)
- Epic 8 Story 8.1 (if time permits)

## Technical Stack

### Context-Aware Chunking
- **dockling**: Hybrid chunking and document parsing
- **tiktoken**: Token counting
- **sentence-transformers**: Semantic similarity
- **beautifulsoup4**: HTML parsing
- **pypdf2**: PDF processing

### Re-ranking
- **sentence-transformers**: Cross-encoder models
  - `cross-encoder/ms-marco-MiniLM-L-6-v2` (recommended)
  - `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **cohere**: Cohere Rerank API
- **transformers**: BGE reranker models

### Query Expansion
- **LLM Service**: OpenAI GPT-3.5/4, Anthropic Claude
- **nltk**: NLP utilities
- **spacy**: Advanced NLP
- **scikit-learn**: Similarity metrics

## Success Criteria

### Quality Metrics
- [ ] Context-aware chunking produces coherent chunks (avg coherence >0.7)
- [ ] Re-ranking improves NDCG by >10% over baseline
- [ ] Query expansion improves search precision by >15%
- [ ] Combined strategies show >25% improvement over naive approach

### Performance Metrics
- [ ] Chunking: >100 chunks/second
- [ ] Re-ranking: <2 seconds for 100 candidates
- [ ] Query expansion: <1 second per query
- [ ] End-to-end latency: <3 seconds

### System Metrics
- [ ] All three strategies can be combined in a pipeline
- [ ] Performance metrics tracked for each strategy
- [ ] Integration tests passing
- [ ] Can demonstrate improvement over baseline

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                          │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Query Expansion     │
                    │   (Story 4.3)         │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Vector Search        │
                    │  (Retrieve 50-100)    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Re-ranking          │
                    │   (Story 4.2)         │
                    │   (Return top 5-10)   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   LLM Generation      │
                    │   (with context)      │
                    └───────────────────────┘

Note: Document chunks created using Context-Aware Chunking (Story 4.1)
      before being stored in vector database
```

## Implementation Order

1. **Story 4.1: Context-Aware Chunking** (FIRST)
   - Implement chunking strategies
   - This affects how documents are prepared
   - Needed before populating vector database

2. **Story 4.2: Re-ranking** (SECOND)
   - Implement re-ranking models
   - Improves retrieval quality
   - Works with existing vector search

3. **Story 4.3: Query Expansion** (THIRD)
   - Implement query expansion
   - Improves search precision
   - Can be added without changing other components

## Configuration Example

```yaml
# config.yaml for Epic 4 strategies

# Context-Aware Chunking
chunking:
  method: "hybrid"  # semantic | structural | hybrid | fixed_size
  min_chunk_size: 128
  max_chunk_size: 1024
  target_chunk_size: 512
  chunk_overlap: 50
  similarity_threshold: 0.7
  use_dockling: true
  compute_coherence_scores: true

# Re-ranking
reranking:
  enabled: true
  model: "cross_encoder"  # cross_encoder | cohere | bge
  model_name: "ms-marco-MiniLM-L-6-v2"
  initial_retrieval_size: 100
  top_k: 10
  enable_cache: true
  batch_size: 32

# Query Expansion
query_expansion:
  enabled: true
  strategy: "keyword"  # keyword | reformulation | multi_query | hyde
  llm_model: "gpt-3.5-turbo"
  max_additional_terms: 5
  enable_cache: true
  track_metrics: true
```

## Usage Example

```python
from rag_factory.strategies.chunking import HybridChunker, ChunkingConfig
from rag_factory.strategies.reranking import RerankerService, RerankConfig
from rag_factory.strategies.query_expansion import QueryExpanderService, ExpansionConfig
from rag_factory.services.embedding import EmbeddingService
from rag_factory.services.llm import LLMService

# 1. Setup services
embedding_service = EmbeddingService(embed_config)
llm_service = LLMService(llm_config)

# 2. Create chunker (Story 4.1)
chunk_config = ChunkingConfig(method="hybrid", target_chunk_size=512)
chunker = HybridChunker(chunk_config, embedding_service)

# Chunk documents
documents = load_documents()
chunks = chunker.chunk_documents(documents)

# Store chunks in vector database
vector_db.add_chunks(chunks)

# 3. Create query expander (Story 4.3)
expansion_config = ExpansionConfig(strategy="keyword", max_additional_terms=5)
expander = QueryExpanderService(expansion_config, llm_service)

# 4. Create re-ranker (Story 4.2)
rerank_config = RerankConfig(
    model="cross_encoder",
    initial_retrieval_size=100,
    top_k=10
)
reranker = RerankerService(rerank_config)

# 5. Full RAG pipeline
def rag_search(user_query: str) -> str:
    # Expand query
    expansion = expander.expand(user_query)
    expanded_query = expansion.primary_expansion.expanded_query

    # Retrieve candidates
    candidates = vector_db.search(expanded_query, top_k=100)

    # Re-rank candidates
    reranked = reranker.rerank(user_query, candidates)
    top_chunks = reranked.results[:10]

    # Generate answer with LLM
    context = "\n\n".join([chunk.text for chunk in top_chunks])
    answer = llm_service.generate(
        prompt=f"Question: {user_query}\n\nContext: {context}\n\nAnswer:",
        max_tokens=500
    )

    return answer.text

# Use the pipeline
answer = rag_search("What is machine learning?")
print(answer)
```

## Testing Strategy

### Unit Tests
- Test each strategy independently
- Mock external dependencies (LLM, embeddings, vector DB)
- Test edge cases and error handling
- Aim for >90% code coverage

### Integration Tests
- Test with real models and services
- Validate quality improvements
- Test strategy combinations
- Measure performance benchmarks

### Quality Tests
- Compare against baseline (naive chunking + vector search)
- Measure NDCG, MRR, precision, recall
- A/B testing framework for query expansion
- Human evaluation for sample queries

### Performance Tests
- Benchmark chunking throughput
- Benchmark re-ranking latency
- Benchmark query expansion latency
- End-to-end pipeline performance

## Monitoring and Metrics

### Per-Strategy Metrics

**Chunking**:
- Average chunk size
- Chunk size distribution
- Coherence scores
- Chunking throughput

**Re-ranking**:
- Re-ranking latency
- NDCG improvement
- MRR improvement
- Cache hit rate

**Query Expansion**:
- Expansion latency
- Cache hit rate
- Number of added terms
- Expansion success rate

### Combined Metrics
- End-to-end latency
- Overall quality improvement
- User satisfaction scores
- Cost per query (LLM + API costs)

## Cost Considerations

### LLM Costs (Query Expansion)
- Every query uses LLM tokens
- Estimated cost: $0.0001-0.001 per query
- Mitigation: Aggressive caching

### Reranking Costs
- **Cross-encoder (local)**: No API cost, GPU recommended
- **Cohere Rerank**: ~$2 per 1000 searches
- Mitigation: Cache, use local models when possible

### Embedding Costs (Chunking)
- One-time cost per document
- Semantic chunking uses embeddings
- Use local models for development

## Risk Management

### Risks and Mitigations

1. **Performance Risk**: Strategies add latency
   - **Mitigation**: Caching, parallel processing, GPU acceleration

2. **Quality Risk**: Expansion may drift from intent
   - **Mitigation**: Careful prompt engineering, validation, A/B testing

3. **Cost Risk**: LLM and API costs may be high
   - **Mitigation**: Caching, local models, usage limits

4. **Complexity Risk**: Three strategies add system complexity
   - **Mitigation**: Modular design, comprehensive testing, monitoring

5. **Dependency Risk**: External APIs (Cohere, OpenAI) may fail
   - **Mitigation**: Fallback strategies, timeouts, error handling

## Future Enhancements

### Post-MVP Improvements
- [ ] Fine-tuned re-ranking models for specific domains
- [ ] Multi-stage re-ranking (coarse-to-fine)
- [ ] Learned query expansion (vs LLM-based)
- [ ] Adaptive chunk sizing based on content
- [ ] Real-time A/B testing dashboard
- [ ] Cost optimization strategies
- [ ] Multi-language support

## References

### Academic Papers
- "HyDE: Precise Zero-Shot Dense Retrieval" (Gao et al., 2022)
- "Lost in the Middle: How Language Models Use Long Contexts" (Liu et al., 2023)
- "Improving Retrieval with Query Expansion" (Nogueira et al., 2019)

### Libraries and Tools
- [Dockling](https://github.com/DS4SD/dockling) - Document parsing
- [Sentence Transformers](https://www.sbert.net/) - Cross-encoders
- [Cohere Rerank](https://docs.cohere.com/reference/rerank) - Rerank API

## Getting Help

For questions or issues:
1. Check story documentation in this directory
2. Review code examples in tests
3. See Epic 4 technical specifications
4. Consult team members for architecture decisions
