# RAG Factory System - Project Plan

## Project Overview

Build a modular RAG (Retrieval Augmented Generation) factory system with a unified interface that supports multiple RAG strategies. Each strategy will be implemented as a separate class that adheres to a common interface, allowing for flexible composition and selection of RAG approaches based on use case requirements.

---

## Library Design Principles

**This is a Python library/module designed to be imported and used programmatically, NOT a standalone application.**

### Core Design Goals:

1. **Importable Module** - All functionality accessible via `from rag_factory import ...`
2. **Programmatic Configuration** - Strategies configured via code or passed objects, not just config files
3. **No Required Server** - Core library has zero server/network dependencies
4. **Clean Abstractions** - Interface-based design allows easy extension and testing
5. **Framework Agnostic** - Can be used with FastAPI, Flask, Django, or no framework at all

### Usage Example:
```python
from rag_factory import RAGFactory, StrategyPipeline
from rag_factory.strategies import ReRankingStrategy, QueryExpansionStrategy

# Initialize factory
factory = RAGFactory(db_config=db_config, embedding_config=embedding_config)

# Create strategy pipeline
pipeline = StrategyPipeline([
    factory.create_strategy("context_aware_chunking"),
    factory.create_strategy("reranking", top_k=10),
    factory.create_strategy("query_expansion")
])

# Use it
results = pipeline.retrieve(query="What are the action items?")
```

### Development Tools:

The project includes optional CLI and dev server tools for testing and POCs:
- **CLI** - Command-line interface for quick strategy testing
- **Dev Server** - Lightweight HTTP server for demos and POCs
- **Not Production Tools** - These are development aids, not the primary interface

---

## Epic 1: Core Infrastructure & Factory Pattern

**Epic Goal:** Establish the foundational architecture with interface definitions, factory pattern, and configuration management that will support all RAG strategies.

**Epic Story Points Total:** 31

**Dependencies:** None (foundational)

---

### Story 1.1: Design RAG Strategy Interface

**As a** developer  
**I want** a unified interface for all RAG strategies  
**So that** different strategies can be used interchangeably and combined

**Acceptance Criteria:**
- Define `IRAGStrategy` interface with methods: `prepare_data()`, `retrieve()`, `process_query()`
- Include configuration parameters (chunk_size, top_k, etc.)
- Support async operations
- Define common return types for chunks and results
- Include metadata structure for tracking which strategy was used

**Technical Notes:**
- Consider using Abstract Base Class (Python) or Interface (TypeScript/Java)
- Should support both sync and async implementations

**Story Points:** 5

---

### Story 1.2: Implement RAG Factory

**As a** developer  
**I want** a factory class to instantiate RAG strategies  
**So that** I can dynamically create and compose strategies

**Acceptance Criteria:**
- Factory can create any registered strategy by name
- Support strategy registration mechanism
- Allow configuration injection
- Support strategy chaining/composition
- Validate strategy configurations

**Technical Notes:**
- Use Factory pattern with strategy registry
- Consider dependency injection for external services (DB, LLM clients)

**Story Points:** 8

---

### Story 1.3: Build Strategy Composition Engine

**As a** developer  
**I want** to combine multiple RAG strategies in a pipeline  
**So that** I can leverage 3-5 strategies together for optimal results

**Acceptance Criteria:**
- Define pipeline configuration format
- Execute strategies in specified order
- Pass results between strategies
- Handle strategy failures gracefully
- Log performance metrics for each stage

**Story Points:** 8

---

### Story 1.4: Create Configuration Management System

**As a** developer  
**I want** centralized configuration for all RAG strategies  
**So that** I can easily tune and experiment with different settings

**Acceptance Criteria:**
- YAML/JSON config file support
- Environment-specific configurations
- Configuration validation
- Hot-reload capability for development
- Default configurations for each strategy

**Story Points:** 5

---

### Story 1.5: Setup Package Structure & Distribution

**As a** developer  
**I want** proper package structure with installability  
**So that** the library can be easily imported and distributed

**Acceptance Criteria:**
- Create proper package structure with `__init__.py` files
- Setup `pyproject.toml` or `setup.py` for pip installation
- Define clear import paths (e.g., `from rag_factory.strategies import ReRankingStrategy`)
- Create requirements.txt and optional dependencies
- Setup versioning (semantic versioning)
- Add package to PyPI test server
- Include basic smoke test that imports all public APIs

**Package Structure:**
```
rag_factory/
├── __init__.py              # Main exports
├── strategies/
│   ├── __init__.py
│   ├── base.py              # IRAGStrategy interface
│   ├── reranking.py
│   └── ...
├── services/
│   ├── __init__.py
│   ├── embedding.py
│   └── llm.py
├── repositories/
│   ├── __init__.py
│   └── ...
├── factory.py
├── pipeline.py
└── config.py
```

**Story Points:** 5

---

## Epic 2: Database & Storage Infrastructure

**Epic Goal:** Set up the database layer with PostgreSQL + pgvector for vector storage and implement repository patterns for data access.

**Epic Story Points Total:** 13

**Dependencies:** Epic 1 (Story 1.1 for interface definitions)

---

### Story 2.1: Set Up Vector Database with PG Vector

**As a** system  
**I want** PostgreSQL with pgvector extension  
**So that** I can store and search vector embeddings efficiently

**Acceptance Criteria:**
- PostgreSQL database setup with pgvector
- Chunks table with vector column
- Documents metadata table
- Indexes for fast similarity search
- Connection pooling
- Database migration scripts

**Technical Dependencies:**
- PostgreSQL 15+ with pgvector
- Consider Neon for managed solution

**Story Points:** 5

---

### Story 2.2: Implement Database Repository Pattern

**As a** developer  
**I want** repository classes for database operations  
**So that** database logic is abstracted from strategies

**Acceptance Criteria:**
- ChunkRepository with CRUD operations
- DocumentRepository with CRUD operations
- Support for batch operations
- Transaction management
- Unit tests for repositories

**Story Points:** 8

---

## Epic 3: Core Services Layer

**Epic Goal:** Build the foundational services (Embedding Service and LLM Service) that all RAG strategies will depend on.

**Epic Story Points Total:** 16

**Dependencies:** Epic 2 (database must be ready)

---

### Story 3.1: Build Embedding Service

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

### Story 3.2: Implement LLM Service Adapter

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

## Epic 4: Priority RAG Strategies (High Impact)

**Epic Goal:** Implement the three highest-impact strategies recommended in the video: Re-ranking, Context-Aware Chunking, and Query Expansion.

**Epic Story Points Total:** 34

**Dependencies:** Epic 3 (requires Embedding and LLM services)

---

### Story 4.1: Implement Context-Aware Chunking Strategy

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

### Story 4.2: Implement Re-ranking Strategy

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

### Story 4.3: Implement Query Expansion Strategy

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

## Epic 5: Agentic & Advanced Retrieval Strategies

**Epic Goal:** Implement advanced strategies that add intelligence and flexibility to retrieval: Agentic RAG, Self-Reflective RAG, and Hierarchical RAG.

**Epic Story Points Total:** 34

**Dependencies:** Epic 4 (basic strategies should be working first)

---

### Story 5.1: Implement Agentic RAG Strategy

**As a** system  
**I want** agents to choose how to search the knowledge base  
**So that** retrieval is flexible based on query type

**Acceptance Criteria:**
- Define multiple search tools (semantic search, full document read, metadata search)
- Implement tool selection logic (LLM-based or rule-based)
- Support both chunks table and documents table
- Log which tools were selected and why
- Handle cases where multiple tools are needed
- Integration tests with various query types

**Technical Dependencies:**
- Agent framework integration (LangGraph, Anthropic Computer Use, etc.)

**Story Points:** 13

---

### Story 5.2: Implement Hierarchical RAG Strategy

**As a** system  
**I want** parent-child chunk relationships  
**So that** I can search small but return large context

**Acceptance Criteria:**
- Store parent-child relationships in metadata
- Search at small chunk level (paragraphs)
- Retrieve parent context (sections/full doc)
- Configurable hierarchy levels
- Support multiple expansion strategies
- Database schema updates for relationships

**Technical Dependencies:**
- Metadata schema for relationships

**Story Points:** 13

---

### Story 5.3: Implement Self-Reflective RAG Strategy

**As a** system  
**I want** a self-correcting search loop  
**So that** poor results trigger refined searches

**Acceptance Criteria:**
- Grade retrieved chunks using LLM (1-5 scale)
- Retry with refined query if grade < threshold
- Configurable retry attempts (e.g., max 2 retries)
- Log grade and retry information
- Support custom grading prompts
- Performance monitoring for retry patterns

**Technical Dependencies:**
- LLM for grading

**Story Points:** 8

---

## Epic 6: Multi-Query & Contextual Strategies

**Epic Goal:** Implement strategies that enhance retrieval through multiple perspectives and enriched context.

**Epic Story Points Total:** 26

**Dependencies:** Epic 4 (basic retrieval working)

---

### Story 6.1: Implement Multi-Query RAG Strategy

**As a** system  
**I want** to generate multiple query variants  
**So that** retrieval has broader coverage

**Acceptance Criteria:**
- Generate 3-5 query variants using LLM
- Execute searches in parallel
- Deduplicate results across queries
- Merge and rank combined results
- Configurable number of variants
- Performance optimization for parallel execution

**Technical Dependencies:**
- Async execution framework

**Story Points:** 13

---

### Story 6.2: Implement Contextual Retrieval Strategy

**As a** system  
**I want** chunks enriched with document context  
**So that** embeddings have more contextual information

**Acceptance Criteria:**
- Generate context descriptions for each chunk using LLM
- Prepend context to chunk before embedding
- Store original chunk separately from contextualized version
- Configurable context generation prompt
- Batch processing for efficiency
- Cost tracking for context generation

**Technical Dependencies:**
- LLM for context generation (Claude, GPT-4, etc.)

**Story Points:** 13

---

## Epic 7: Advanced & Experimental Strategies

**Epic Goal:** Implement the most complex strategies: Knowledge Graphs, Late Chunking, and Fine-Tuned Embeddings.

**Epic Story Points Total:** 63

**Dependencies:** Epic 5 (requires mature system understanding)

---

### Story 7.1: Implement Knowledge Graph Strategy

**As a** system  
**I want** to combine vector search with graph relationships  
**So that** I can leverage entity connections in retrieval

**Acceptance Criteria:**
- Entity extraction from documents using LLM
- Store entities and relationships in graph database
- Hybrid search: vector + graph traversal
- Support relationship queries (e.g., "connected to", "causes")
- Visualize graph structure (optional)
- Performance benchmarks for hybrid search

**Technical Dependencies:**
- Graph database (Neo4j, graffiti, etc.)
- Entity extraction LLM prompts

**Story Points:** 21

---

### Story 7.2: Implement Late Chunking Strategy

**As a** system  
**I want** to apply embeddings before chunking  
**So that** chunks maintain full document context

**Acceptance Criteria:**
- Embed full document first
- Split token embeddings into chunks
- Maintain context relationships
- Support long-context embedding models
- Document the complexity and use cases
- Comparative analysis vs traditional chunking

**Technical Dependencies:**
- Long-context embedding model
- Custom chunking logic for token embeddings

**Story Points:** 21

---

### Story 7.3: Implement Fine-Tuned Embeddings Strategy

**As a** system  
**I want** to use domain-specific embedding models  
**So that** accuracy improves for specialized use cases

**Acceptance Criteria:**
- Support loading custom embedding models
- Training pipeline for fine-tuning (separate epic potential)
- A/B testing framework for comparing models
- Model versioning and rollback
- Performance metrics tracking
- Documentation for training custom models

**Technical Dependencies:**
- Model training infrastructure
- Model registry

**Story Points:** 21

---

## Epic 8: Observability & Quality Assurance

**Epic Goal:** Build comprehensive monitoring, logging, and evaluation frameworks to ensure system quality and enable continuous improvement.

**Epic Story Points Total:** 21

**Dependencies:** Epic 4 (need working strategies to evaluate)

---

### Story 8.1: Build Monitoring & Logging System

**As a** developer  
**I want** comprehensive logging and monitoring  
**So that** I can debug and optimize RAG performance

**Acceptance Criteria:**
- Log all strategy executions with timestamps
- Track performance metrics (latency, cost, token usage)
- Error tracking with stack traces
- Query analytics and aggregation
- Export logs to files or external systems
- Dashboard for real-time monitoring

**Story Points:** 8

---

### Story 8.2: Create Evaluation Framework

**As a** developer  
**I want** to evaluate and compare RAG strategies  
**So that** I can choose the best approach for my use case

**Acceptance Criteria:**
- Define evaluation metrics (accuracy, latency, cost)
- Test dataset management
- Benchmarking suite for strategy comparison
- Results visualization dashboard
- Export results to CSV/JSON
- Statistical significance testing

**Story Points:** 13

---

## Epic 8.5: Development Tools (CLI & Dev Server)

**Epic Goal:** Create lightweight development tools for testing, debugging, and POC demonstrations. These are NOT production-ready clients but development aids.

**Epic Story Points Total:** 16

**Dependencies:** Epic 4 (need working strategies to test)

---

### Story 8.5.1: Build CLI for Strategy Testing

**As a** developer  
**I want** a command-line interface for testing strategies  
**So that** I can quickly experiment without writing code

**Acceptance Criteria:**
- Commands for indexing documents: `rag-factory index --path ./docs --strategy context_aware_chunking`
- Commands for querying: `rag-factory query "your question" --strategies reranking,query_expansion`
- List available strategies: `rag-factory strategies --list`
- Configuration validation: `rag-factory config --validate config.yaml`
- Benchmark mode: `rag-factory benchmark --dataset test.json`
- Interactive REPL mode for exploration
- Uses the library exactly as a client would (validates public API)
- Colorized output and progress bars
- NOT intended for production use

**Technical Notes:**
- Use Click or Typer for CLI framework
- All logic delegates to the library

**Story Points:** 8

---

### Story 8.5.2: Create Lightweight Dev Server for POCs

**As a** developer  
**I want** a simple HTTP server for demos and POCs  
**So that** I can test integrations without building a full client

**Acceptance Criteria:**
- Simple HTTP endpoints (POST /query, POST /index, GET /strategies)
- JSON request/response format
- Strategy selection via request parameters
- Configuration hot-reloading for rapid iteration
- Simple HTML UI for manual testing (single page)
- Request/response logging
- Health check endpoint
- NOT production-ready (no auth, rate limiting, or scaling)
- Clear documentation that this is for development only
- Uses the library exactly as a client would

**Technical Notes:**
- Use FastAPI for simplicity
- Single-file implementation (~200 lines)
- Include Swagger/OpenAPI docs

**Example Usage:**
```bash
# Start dev server
rag-factory serve --config dev_config.yaml --port 8000

# Query via HTTP
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the action items?", "strategies": ["reranking", "query_expansion"]}'
```

**Story Points:** 8

---

## Epic 9: Documentation & Developer Experience

**Epic Goal:** Create comprehensive documentation and examples to enable developers to effectively use and extend the RAG factory system.

**Epic Story Points Total:** 26

**Dependencies:** All previous epics (need complete system)

---

### Story 9.1: Write Developer Documentation

**As a** developer  
**I want** comprehensive documentation  
**So that** I can understand and extend the RAG factory

**Acceptance Criteria:**
- Architecture overview with diagrams
- Strategy selection guide with decision tree
- Code examples for each strategy
- Configuration reference with all options
- Contribution guidelines
- API reference documentation
- Troubleshooting guide

**Story Points:** 13

---

### Story 9.2: Create Example Implementations

**As a** developer  
**I want** working examples showing how to import and use the library  
**So that** I can quickly get started

**Acceptance Criteria:**
- **Simple example** - Single strategy usage showing basic imports
  ```python
  from rag_factory import RAGFactory
  factory = RAGFactory(config)
  strategy = factory.create_strategy("reranking")
  results = strategy.retrieve(query)
  ```
- **Medium example** - 3 strategies pipeline (recommended combination)
  ```python
  from rag_factory import StrategyPipeline
  pipeline = StrategyPipeline([...])
  ```
- **Complex example** - 5+ strategies with custom configuration
- **Domain-specific examples** - Legal, medical, customer support use cases
- **Integration examples** - Using with FastAPI, Flask, LangChain
- Docker compose setup for local development (database + optional services)
- Jupyter notebooks for experimentation and visualization
- Each example emphasizes importing and programmatic usage
- Video walkthrough or tutorial

**Story Points:** 13

---

## Sprint Planning Recommendation

### Sprint 1: Foundation (44 points)
- Epic 1: All stories (1.1 - 1.5) = 31 points
- Epic 2: All stories (2.1 - 2.2) = 13 points

### Sprint 2: Core Services (16 points)
- Epic 3: All stories (3.1 - 3.2) = 16 points

### Sprint 3: Priority Strategies Part 1 (26 points)
- Epic 4: Stories 4.1, 4.2 = 26 points (13 + 13)

### Sprint 4: Priority Strategies Part 2 & Observability (16 points)
- Epic 4: Story 4.3 = 8 points
- Epic 8: Story 8.1 = 8 points

### Sprint 5: Advanced Retrieval Part 1 (21 points)
- Epic 5: Stories 5.1, 5.3 = 21 points (13 + 8)

### Sprint 6: Advanced Retrieval Part 2 (13 points)
- Epic 5: Story 5.2 = 13 points

### Sprint 7: Multi-Query & Context (26 points)
- Epic 6: All stories (6.1 - 6.2) = 26 points (13 + 13)

### Sprint 8: Evaluation & Dev Tools (29 points)
- Epic 8: Story 8.2 = 13 points
- Epic 8.5: All stories (8.5.1 - 8.5.2) = 16 points (8 + 8)

### Sprint 9: Knowledge Graph (21 points)
- Epic 7: Story 7.1 = 21 points

### Sprint 10: Experimental Strategies (42 points)
- Epic 7: Stories 7.2, 7.3 = 42 points (21 + 21)

### Sprint 11: Documentation (26 points)
- Epic 9: All stories (9.1 - 9.2) = 26 points (13 + 13)

---

## Total Project Story Points

**Total Story Points Across All Epics:** 280 points

**Epic Breakdown:**
- Epic 1: Core Infrastructure = 31 points
- Epic 2: Database & Storage = 13 points
- Epic 3: Core Services = 16 points
- Epic 4: Priority Strategies = 34 points
- Epic 5: Agentic & Advanced = 34 points
- Epic 6: Multi-Query & Contextual = 26 points
- Epic 7: Advanced & Experimental = 63 points
- Epic 8: Observability = 21 points
- Epic 8.5: Development Tools = 16 points
- Epic 9: Documentation = 26 points

**Estimated Duration:** 11 sprints (assuming 25-30 point velocity per sprint)

---

## Priority Implementation Path (MVP)

For quickest time to value, implement in this order:

1. **Foundation** (Epic 1, 2, 3) - Required infrastructure including package structure
2. **High-Impact Trio** (Stories 4.1, 4.2, 4.3) - Re-ranking + Context-Aware Chunking + Query Expansion
3. **Observability** (Story 8.1) - Logging to understand performance
4. **Dev Tools** (Story 8.5.1 - CLI) - Enable quick testing without writing code
5. **Documentation** (Story 9.2 - Simple example) - Enable library usage

This gives you an importable library with the three recommended strategies, basic CLI for testing, and examples showing how to use it programmatically.

---

## Technical Stack Recommendations

**Programming Language:** Python 3.11+

**Database:**
- PostgreSQL 15+ with pgvector extension
- Recommended: Neon (managed PostgreSQL)

**Embedding Models:**
- OpenAI text-embedding-3-small/large
- Cohere embed-multilingual-v3
- Local: sentence-transformers

**LLM Providers:**
- Anthropic Claude (Sonnet 4.5)
- OpenAI GPT-4
- Local models via Ollama

**Key Libraries:**
- dockling (hybrid chunking)
- sentence-transformers (re-ranking)
- graffiti (knowledge graphs)
- langchain/langgraph (agentic orchestration)
- psycopg2/asyncpg (database)
- pydantic (validation)

**Development Tools:**
- Docker & Docker Compose
- pytest (testing)
- black/ruff (formatting)
- mypy (type checking)

---

## Success Metrics

**Performance Metrics:**
- Retrieval accuracy > 85% (measured via evaluation framework)
- Query latency < 2 seconds (p95)
- Cost per query < $0.02

**Code Quality Metrics:**
- Test coverage > 80%
- Type coverage > 90%
- Zero critical security vulnerabilities

**Developer Experience:**
- Documentation completeness score > 90%
- Time to first working example < 30 minutes
- Strategy addition time < 4 hours

---

## Risk Management

**High Risks:**
1. **Late Chunking Complexity** - Extremely complex, may not be worth implementing
   - Mitigation: Mark as optional/experimental, implement last
   
2. **Knowledge Graph Performance** - Can be slow with large datasets
   - Mitigation: Implement caching, limit graph depth, use indexes
   
3. **Cost Overruns** - Multiple LLM calls can get expensive
   - Mitigation: Implement caching, batch operations, cost tracking

**Medium Risks:**
1. **Integration Complexity** - Many external services to integrate
   - Mitigation: Use adapter pattern, mock extensively in tests
   
2. **Configuration Explosion** - Too many knobs to tune
   - Mitigation: Provide sensible defaults, configuration presets

---

## Notes

- Story points use Fibonacci sequence (1, 2, 3, 5, 8, 13, 21)
- Each sprint assumes ~25-30 points capacity
- Stories can be broken down further during sprint planning
- Technical spikes may be needed for Epic 7 stories
- Consider pairing on complex stories (13+ points)
