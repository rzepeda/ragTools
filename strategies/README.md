# Strategy Pair Configurations

This directory contains pre-built configuration files for RAG strategy pairs. Each YAML file defines a compatible pair of Indexing and Retrieval strategies, along with their configuration and service requirements.

## Available Strategy Pairs

### 1. **semantic-local-pair.yaml** ✅
**Local ONNX embeddings (no API keys required)**
- Indexer: `VectorEmbeddingIndexer`
- Retriever: `SemanticRetriever`
- Services: Local ONNX (`all-MiniLM-L6-v2`), PostgreSQL
- Use Cases: Development, testing, privacy-sensitive deployments
- Cost: Free (local compute only)

### 2. **semantic-api-pair.yaml**
**OpenAI/Cohere API embeddings**
- Indexer: `VectorEmbeddingIndexer`
- Retriever: `SemanticRetriever`
- Services: OpenAI/Cohere API, PostgreSQL
- Use Cases: Production deployments, high-quality embeddings, multi-language
- Cost: ~$0.02-0.10 per 1k documents

### 3. **keyword-pair.yaml**
**BM25 keyword search (no embeddings)**
- Indexer: `KeywordIndexer`
- Retriever: `KeywordRetriever`
- Services: PostgreSQL only
- Use Cases: Exact term matching, legal/compliance, low-latency
- Cost: Free (no embeddings)

### 4. **hybrid-search-pair.yaml** ✅
**Semantic + Keyword fusion**
- Indexer: `HybridIndexer`
- Retriever: `HybridRetriever`
- Services: Embedding service, PostgreSQL
- Use Cases: Best of both worlds, production-ready
- Cost: Embedding costs only

### 5. **reranking-pair.yaml**
**Two-stage retrieval with cross-encoder reranking**
- Indexer: `VectorEmbeddingIndexer`
- Retriever: `RerankingRetriever`
- Services: Embedding, Reranker, PostgreSQL
- Use Cases: High-precision retrieval, quality over speed
- Cost: Embedding + reranking compute

### 6. **query-expansion-pair.yaml**
**LLM-based query enhancement (HyDE)**
- Indexer: `VectorEmbeddingIndexer`
- Retriever: `QueryExpansionRetriever`
- Services: Embedding, LLM, PostgreSQL
- Use Cases: Complex queries, improved recall
- Cost: Embedding + LLM costs

### 7. **context-aware-chunking-pair.yaml**
**Semantic boundary-aware chunking**
- Indexer: `ContextAwareChunker`
- Retriever: `SemanticRetriever`
- Services: Embedding, PostgreSQL
- Use Cases: Natural content divisions, better chunk quality
- Cost: Embedding costs

### 8. **agentic-rag-pair.yaml**
**Agent-based tool selection**
- Indexer: `VectorEmbeddingIndexer`
- Retriever: `AgenticRAGStrategy`
- Services: Embedding, LLM, PostgreSQL
- Use Cases: Complex reasoning, multi-step retrieval
- Cost: Embedding + LLM costs (higher)

### 9. **hierarchical-rag-pair.yaml**
**Parent-child chunk hierarchy**
- Indexer: `HierarchicalRAGStrategy`
- Retriever: `HierarchicalRAGStrategy`
- Services: Embedding, PostgreSQL
- Use Cases: Context-aware retrieval, document structure preservation
- Cost: Embedding costs

### 10. **self-reflective-pair.yaml**
**Self-correcting retrieval with quality assessment**
- Indexer: `VectorEmbeddingIndexer`
- Retriever: `SelfReflectiveRAGStrategy`
- Services: Embedding, LLM, PostgreSQL
- Use Cases: High-quality results, iterative refinement
- Cost: Embedding + LLM costs (higher)

### 11. **multi-query-pair.yaml**
**Multiple query variants for improved recall**
- Indexer: `VectorEmbeddingIndexer`
- Retriever: `MultiQueryRAGStrategy`
- Services: Embedding, LLM, PostgreSQL
- Use Cases: Ambiguous queries, comprehensive retrieval
- Cost: Embedding + LLM costs

### 12. **contextual-retrieval-pair.yaml**
**LLM-enriched chunks with contextual information**
- Indexer: `ContextualRetrievalStrategy`
- Retriever: `SemanticRetriever`
- Services: Embedding, LLM, PostgreSQL
- Use Cases: High-quality indexing, context preservation
- Cost: Embedding + LLM costs (indexing time)

### 13. **knowledge-graph-pair.yaml**
**Graph + vector hybrid search**
- Indexer: `KnowledgeGraphRAGStrategy`
- Retriever: `KnowledgeGraphRAGStrategy`
- Services: Embedding, LLM, PostgreSQL, Neo4j
- Use Cases: Entity relationships, complex queries
- Cost: Embedding + LLM + graph DB costs

### 14. **late-chunking-pair.yaml**
**Embed-then-chunk for context preservation**
- Indexer: `LateChunkingStrategy`
- Retriever: `SemanticRetriever`
- Services: Embedding, PostgreSQL
- Use Cases: Context-preserving embeddings, long documents
- Cost: Embedding costs (higher per document)

### 15. **fine-tuned-embeddings-pair.yaml**
**Custom fine-tuned embedding models**
- Indexer: `VectorEmbeddingIndexer`
- Retriever: `SemanticRetriever`
- Services: Custom embedding service, PostgreSQL
- Use Cases: Domain-specific retrieval, specialized applications
- Cost: Custom model hosting/compute

## Strategy Compatibility Matrix

### Can Be Combined
| Base Strategy | Compatible Add-ons |
|---------------|-------------------|
| semantic-local-pair | + reranking, + query-expansion, + hierarchical |
| semantic-api-pair | + reranking, + query-expansion, + hierarchical |
| keyword-pair | + reranking |
| semantic + keyword | → hybrid-search-pair |
| any semantic | + contextual-retrieval (at indexing time) |

### Require Different Tables (Isolated)
- semantic-local vs semantic-api (different embedding dimensions)
- keyword vs semantic (different index structures)
- graph vs vector (different storage backends)

## Migration Dependencies

1. **Base schema** (Epic 2) - Required for all
2. **Vector tables** - Required for semantic pairs
3. **Keyword tables** - Required for keyword/hybrid pairs
4. **Graph tables** - Required for knowledge graph pair
5. **Hierarchy tables** - Required for hierarchical pair

## Quick Start

```python
from rag_factory.config.strategy_pair_manager import StrategyPairManager
from rag_factory.registry.service_registry import ServiceRegistry

# Initialize services
registry = ServiceRegistry()

# Load a strategy pair
manager = StrategyPairManager(registry, "strategies")
indexing, retrieval = manager.load_pair("semantic-local-pair")

# Use the strategies
await indexing.process(documents, context)
results = await retrieval.retrieve(query, context)
```

## Configuration Structure

Each strategy pair YAML file contains:
- `strategy_name`: Unique identifier
- `version`: Semantic version
- `description`: Human-readable description
- `indexer`: Indexing strategy configuration
  - `strategy`: Strategy class name
  - `services`: Required service references
  - `db_config`: Database table/field mappings
  - `config`: Strategy-specific parameters
- `retriever`: Retrieval strategy configuration
- `migrations`: Required Alembic revisions
- `expected_schema`: Database schema requirements
- `tags`: Categorization tags

## See Also

- [Strategy Pair Matrix](../docs/strategies/strategy-pair-matrix.md)
- [Migration Dependencies](../docs/strategies/migration-dependencies.md)
- [Service Registry Configuration](../rag_factory/config/examples/services.yaml)
