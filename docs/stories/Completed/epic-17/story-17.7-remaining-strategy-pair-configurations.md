# Story 17.7: Create Remaining Strategy Pair Configurations

**As a** user  
**I want** pre-built configurations for all RAG strategies from previous epics  
**So that** I can quickly deploy any RAG approach without writing YAML

## Acceptance Criteria
- Create strategy pair configurations for ALL strategies from Epics 4-7, 12-13:
  1. **semantic-api-pair.yaml** - OpenAI/Cohere API embeddings
  2. **reranking-pair.yaml** - Two-stage retrieval with reranking (Epic 4)
  3. **query-expansion-pair.yaml** - LLM-based query enhancement (Epic 4)
  4. **context-aware-chunking-pair.yaml** - Semantic boundary chunking (Epic 4)
  5. **agentic-rag-pair.yaml** - Agent-based tool selection (Epic 5)
  6. **hierarchical-rag-pair.yaml** - Parent-child chunks (Epic 5)
  7. **self-reflective-pair.yaml** - Self-correcting retrieval (Epic 5)
  8. **multi-query-pair.yaml** - Multiple query variants (Epic 6)
  9. **contextual-retrieval-pair.yaml** - LLM-enriched chunks (Epic 6)
  10. **keyword-pair.yaml** - BM25 keyword search (Epic 12/13)
  11. **hybrid-search-pair.yaml** - Semantic + keyword fusion (Epic 12/13)
  12. **knowledge-graph-pair.yaml** - Graph + vector search (Epic 7)
  13. **late-chunking-pair.yaml** - Embed-then-chunk (Epic 7)
  14. **fine-tuned-embeddings-pair.yaml** - Custom models (Epic 7)
- Each configuration includes:
  - Complete services.yaml entries (can reference or extend base)
  - Required Alembic migrations documented
  - db_config with table/field mappings
  - Example usage code
  - Performance characteristics
  - Cost estimates (if using APIs)
  - Recommended use cases
- All configurations tested with actual strategies
- Documentation matrix showing which pairs can be combined
- Migration dependencies documented (which must run first)

## Deliverables

```
strategies/
├── semantic-local-pair.yaml          # From Story 17.6 ✅
├── semantic-api-pair.yaml            # OpenAI/Cohere
├── reranking-pair.yaml               # Epic 4
├── query-expansion-pair.yaml         # Epic 4
├── context-aware-chunking-pair.yaml  # Epic 4
├── agentic-rag-pair.yaml            # Epic 5
├── hierarchical-rag-pair.yaml       # Epic 5
├── self-reflective-pair.yaml        # Epic 5
├── multi-query-pair.yaml            # Epic 6
├── contextual-retrieval-pair.yaml   # Epic 6
├── keyword-pair.yaml                # Epic 12/13
├── hybrid-search-pair.yaml          # Epic 12/13
├── knowledge-graph-pair.yaml        # Epic 7
├── late-chunking-pair.yaml          # Epic 7
├── fine-tuned-embeddings-pair.yaml  # Epic 7
└── README.md                         # Index of all pairs

docs/strategies/
├── strategy-pair-matrix.md           # Compatibility matrix
├── migration-dependencies.md         # Which migrations needed for what
└── [individual-pair-guides].md       # One per pair
```

## Strategy Pair Matrix (Deliverable)
```markdown
# Strategy Pair Compatibility Matrix

## Can Be Combined
| Base Strategy | Compatible Add-ons |
|---------------|-------------------|
| semantic-local-pair | + reranking, + query-expansion, + hierarchical |
| keyword-pair | + reranking |
| semantic + keyword | → hybrid-search-pair |
| any semantic | + contextual-retrieval (at indexing time) |

## Require Different Tables (Isolated)
- semantic-local vs semantic-api (different embedding dimensions)
- keyword vs semantic (different index structures)
- graph vs vector (different storage backends)

## Migration Dependencies
1. Base schema (Epic 2)
2. Vector tables (semantic pairs)
3. Keyword tables (keyword pair)
4. Graph tables (knowledge graph pair)
5. Hierarchy tables (hierarchical pair)
```

## Story Points
8
