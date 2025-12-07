# Epic 5: Agentic & Advanced Retrieval Strategies

This epic implements advanced, intelligent retrieval strategies that go beyond basic semantic search.

---

## Overview

**Epic Goal:** Implement advanced strategies that add intelligence and flexibility to retrieval: Agentic RAG, Self-Reflective RAG, and Hierarchical RAG.

**Total Story Points:** 34

**Dependencies:** Epic 3 (Embedding Service, LLM Service), Epic 4 (Basic RAG Strategies)

---

## Stories

### Story 5.1: Implement Agentic RAG Strategy (13 points)
**Status:** Planned
**File:** [story-5.1-agentic-rag-strategy.md](./story-5.1-agentic-rag-strategy.md)

**Description:** Implement an agent-based retrieval system where an AI agent dynamically selects the most appropriate search tools based on the query type.

**Key Features:**
- Multi-tool search architecture (semantic, document read, metadata, keyword)
- LLM-based intelligent tool selection
- Agent framework integration (LangGraph, Anthropic tool use)
- Multi-step retrieval workflows
- Query analysis and tool orchestration

**Acceptance Criteria:**
- Agents can select appropriate tools for different query types
- Tool selection includes reasoning/explanation
- Multi-tool workflows supported (sequential and parallel)
- Execution traced for observability
- Performance: <3s for typical queries

---

### Story 5.2: Implement Hierarchical RAG Strategy (13 points)
**Status:** Planned
**File:** [story-5.2-hierarchical-rag-strategy.md](./story-5.2-hierarchical-rag-strategy.md)

**Description:** Implement hierarchical chunk relationships where small chunks are indexed for precise search but parent chunks are returned for complete context.

**Key Features:**
- Parent-child chunk relationships in database
- Small chunk indexing (paragraphs, sentences)
- Parent context retrieval strategies
- Multiple expansion strategies (immediate parent, full section, window, document)
- Automatic hierarchy detection during ingestion

**Acceptance Criteria:**
- Hierarchy levels tracked (document ’ section ’ paragraph)
- Small chunks searchable, parent chunks retrievable
- At least 3 expansion strategies implemented
- Database schema supports efficient parent lookups
- Deduplication working across hierarchy levels

---

### Story 5.3: Implement Self-Reflective RAG Strategy (8 points)
**Status:** Planned
**File:** [story-5.3-self-reflective-rag-strategy.md](./story-5.3-self-reflective-rag-strategy.md)

**Description:** Implement a self-correcting retrieval loop that grades retrieved results and automatically refines the query and retries if quality is below threshold.

**Key Features:**
- LLM-based result quality grading (1-5 scale)
- Automatic query refinement based on gaps
- Configurable retry logic with max attempts
- Multiple refinement strategies
- Result aggregation across attempts

**Acceptance Criteria:**
- Results graded for relevance and completeness
- Retry triggered when grade < threshold (default: 4.0)
- Query refined using LLM based on initial results
- Max retries enforced (default: 2)
- Results aggregated and deduplicated across attempts
- Performance: <10s total including retries

---

## Sprint Planning

**Sprint 5:** Stories 5.1, 5.3 (21 points)
**Sprint 6:** Story 5.2 (13 points)

---

## Technical Stack

### Agentic (Story 5.1)
- **LangGraph** or similar agent framework
- **Anthropic tool use** for Claude function calling
- Custom tool definitions and registry
- Agent state management

### Hierarchical (Story 5.2)
- **Database schema extensions** for parent-child relationships
- Recursive SQL queries for hierarchy navigation
- Metadata management for hierarchy levels

### Self-Reflective (Story 5.3)
- **LLM service** for grading and query refinement
- Retry logic framework
- Result aggregation and deduplication

---

## Success Criteria

Epic 5 is complete when:

- [ ] **Agentic RAG** can intelligently select and orchestrate multiple search tools
- [ ] **Hierarchical RAG** retrieves correct parent chunks with proper context expansion
- [ ] **Self-Reflective RAG** successfully improves low-quality results through iteration
- [ ] All strategies integrate with the main RAG pipeline
- [ ] Performance metrics tracked and meeting requirements
- [ ] All integration tests passing
- [ ] Documentation complete with usage examples

---

## Getting Started

### Prerequisites

Before starting Epic 5, ensure:

1. **Epic 3 Complete:** Embedding Service and LLM Service working
2. **Epic 4 Complete:** Basic chunking and retrieval strategies implemented
3. **Database:** PostgreSQL or compatible with support for recursive queries
4. **LLM Access:** Anthropic API key or compatible LLM provider

### Installation

```bash
# Install Epic 5 dependencies
pip install langgraph anthropic langchain pydantic networkx

# Verify installations
python -c "import langgraph; print('LangGraph OK')"
python -c "import anthropic; print('Anthropic OK')"

# Apply database migrations (for Story 5.2)
psql -d rag_database -f database/migrations/add_hierarchy_support.sql
```

### Configuration

```yaml
# config.yaml
strategies:
  # Story 5.1: Agentic RAG
  agentic:
    enabled: true
    use_llm_selection: true
    max_iterations: 3
    fallback_to_rules: true
    llm:
      provider: anthropic
      model: claude-3-haiku-20240307  # Fast model for tool selection

  # Story 5.2: Hierarchical RAG
  hierarchical:
    enabled: true
    expansion_strategy: "immediate_parent"  # or full_section, window, full_document
    search_small_chunks: true
    small_chunk_size: 256
    large_chunk_size: 1024

  # Story 5.3: Self-Reflective RAG
  self_reflective:
    enabled: true
    grade_threshold: 4.0
    max_retries: 2
    timeout_seconds: 10
    llm:
      provider: anthropic
      model: claude-3-haiku-20240307  # Fast, cheap for grading
```

---

## Implementation Order

**Recommended order:**

1. **Story 5.3: Self-Reflective RAG** (8 points)
   - Simpler to implement
   - No new database schema changes
   - Can wrap existing strategies
   - Provides immediate quality improvements

2. **Story 5.2: Hierarchical RAG** (13 points)
   - Requires database migration
   - Foundation for better context retrieval
   - Used by other strategies

3. **Story 5.1: Agentic RAG** (13 points)
   - Most complex
   - Benefits from hierarchical and self-reflective being available
   - Can orchestrate all other strategies

---

## Testing Strategy

### Unit Tests
Each story includes comprehensive unit tests with >85% coverage:
- Tool/component isolation
- Mock external dependencies (LLM, database, vector store)
- Edge case handling
- Performance benchmarks

### Integration Tests
End-to-end tests with real services:
- Full retrieval workflows
- Real LLM calls (with API keys)
- Real database operations
- Performance validation

### Performance Benchmarks
- **Agentic RAG:** <3s for typical queries
- **Hierarchical RAG:** <50ms parent retrieval overhead
- **Self-Reflective RAG:** <10s including retries

---

## Cost Considerations

### LLM Usage

**Story 5.1 (Agentic):**
- Tool selection: 1-2 LLM calls per query
- Use Haiku for cost efficiency (~$0.001 per query)

**Story 5.2 (Hierarchical):**
- No LLM calls (database-only operations)
- Zero additional LLM cost

**Story 5.3 (Self-Reflective):**
- Grading: 1 LLM call per query
- Refinement: 1 LLM call per retry
- Average: 2-4 LLM calls per query
- Use Haiku (~$0.002 per query)

**Total estimated cost:** ~$0.003-0.005 per query with all strategies enabled

---

## Monitoring and Observability

All strategies include comprehensive logging:

- **Tool selection** decisions and reasoning (Agentic)
- **Hierarchy navigation** paths (Hierarchical)
- **Grade scores** and query refinements (Self-Reflective)
- **Performance metrics** (latency, token usage)
- **Success/failure rates** for each strategy

Example logs:
```
[Agentic] Selected tools: semantic_search, metadata_filter (reasoning: query requests recent documents)
[Hierarchical] Expanded chunk_123 to parent section_45 (strategy: full_section)
[Self-Reflective] Grade: 2.5/5, retrying with refined query (attempt 2/2)
```

---

## Common Patterns

### Strategy Composition

You can compose strategies for powerful combinations:

```python
# Example: Hierarchical + Self-Reflective
hierarchical = HierarchicalRAGStrategy(vector_store, database)
self_reflective = SelfReflectiveRAGStrategy(
    base_retrieval_strategy=hierarchical,
    llm_service=llm
)

# Search small chunks, return parent context, retry if poor quality
results = self_reflective.retrieve("complex query")
```

### Conditional Strategy Selection

```python
def get_strategy(query_type):
    if query_type == "specific_document":
        return AgenticRAGStrategy(...)
    elif query_type == "needs_context":
        return HierarchicalRAGStrategy(...)
    elif query_type == "complex":
        return SelfReflectiveRAGStrategy(...)
    else:
        return SemanticSearchStrategy(...)
```

---

## Troubleshooting

### Agentic RAG Issues

**Problem:** Agent selects wrong tools
**Solution:** Improve tool descriptions, use examples in prompts

**Problem:** Tool selection too slow
**Solution:** Use faster model (Haiku), cache common selections

### Hierarchical RAG Issues

**Problem:** Parent lookups slow
**Solution:** Ensure parent_chunk_id is indexed

**Problem:** Too much context returned
**Solution:** Use more restrictive expansion strategy (immediate_parent vs full_section)

### Self-Reflective RAG Issues

**Problem:** Too many retries, high cost
**Solution:** Lower grade_threshold or reduce max_retries

**Problem:** Refinements don't improve results
**Solution:** Improve refinement prompts, add more context

---

## Documentation

Each story includes:
-  Detailed requirements (functional and non-functional)
-  Acceptance criteria
-  Technical specifications with code examples
-  Unit test examples
-  Integration test scenarios
-  Setup instructions
-  Usage examples
-  Developer notes and best practices

---

## Related Epics

- **Epic 3:** Embedding & LLM Services (prerequisite)
- **Epic 4:** Priority RAG Strategies (prerequisite)
- **Epic 6:** Advanced Features (uses Epic 5 strategies)

---

## References

### Academic Papers
- **Agentic RAG:** "ReAct: Synergizing Reasoning and Acting in Language Models"
- **Hierarchical RAG:** "Lost in the Middle: How Language Models Use Long Contexts"
- **Self-Reflective RAG:** "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"

### Documentation
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Anthropic Tool Use Guide](https://docs.anthropic.com/claude/docs/tool-use)
- [PostgreSQL Recursive Queries](https://www.postgresql.org/docs/current/queries-with.html)

---

## Support

For questions or issues with Epic 5:

1. Check story-specific documentation
2. Review test examples
3. Check troubleshooting section above
4. Consult related epics for dependencies

---

## License

This project and all documentation is licensed under MIT License.
