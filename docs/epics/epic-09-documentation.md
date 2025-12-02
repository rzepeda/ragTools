# Epic 9: Documentation & Developer Experience

**Epic Goal:** Create comprehensive documentation and examples to enable developers to effectively use and extend the RAG factory system.

**Epic Story Points Total:** 26

**Dependencies:** All previous epics (need complete system)

---

## Story 9.1: Write Developer Documentation

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

## Story 9.2: Create Example Implementations

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

## Sprint Planning

**Sprint 11:** All stories (9.1 - 9.2) = 26 points

---

## Documentation Structure

```
docs/
├── README.md                      # Quick start
├── architecture/
│   ├── overview.md               # System architecture
│   ├── design-patterns.md        # Factory, Strategy patterns
│   └── diagrams/                 # Architecture diagrams
├── guides/
│   ├── getting-started.md        # Installation & first steps
│   ├── strategy-selection.md    # How to choose strategies
│   ├── configuration.md          # Configuration reference
│   └── troubleshooting.md        # Common issues
├── strategies/
│   ├── reranking.md             # Each strategy documented
│   ├── context-aware-chunking.md
│   └── ...
├── examples/
│   ├── simple/                   # Simple examples
│   ├── medium/                   # Medium complexity
│   ├── advanced/                 # Advanced examples
│   ├── domain-specific/          # Domain examples
│   └── integrations/             # Framework integrations
├── api-reference/                # Auto-generated API docs
└── contributing.md               # How to contribute
```

---

## Example Types

**1. Simple Example (Getting Started)**
```python
from rag_factory import RAGFactory

# Initialize factory
factory = RAGFactory(
    db_config={"host": "localhost", "database": "ragdb"},
    embedding_config={"model": "text-embedding-3-small"}
)

# Create a strategy
strategy = factory.create_strategy("reranking", top_k=5)

# Query
results = strategy.retrieve("What are the action items?")
for result in results:
    print(f"Score: {result.score}, Text: {result.text}")
```

**2. Medium Example (Pipeline)**
```python
from rag_factory import RAGFactory, StrategyPipeline

factory = RAGFactory(config)

# Create pipeline with 3 strategies
pipeline = StrategyPipeline([
    factory.create_strategy("context_aware_chunking"),
    factory.create_strategy("query_expansion"),
    factory.create_strategy("reranking", top_k=10)
])

results = pipeline.retrieve("Find critical bugs")
```

**3. Integration Example (FastAPI)**
```python
from fastapi import FastAPI
from rag_factory import RAGFactory, StrategyPipeline

app = FastAPI()
factory = RAGFactory(config)
pipeline = StrategyPipeline([...])

@app.post("/query")
async def query(request: QueryRequest):
    results = await pipeline.retrieve_async(request.query)
    return {"results": results}
```

---

## Developer Experience Goals

**Time to First Working Example:** < 30 minutes
- Install package
- Configure database
- Run first query

**Time to Add Custom Strategy:** < 4 hours
- Understand interface
- Implement strategy
- Write tests
- Register with factory

---

## Technical Stack

**Documentation:**
- MkDocs or Sphinx
- API documentation via docstrings
- Mermaid for diagrams

**Examples:**
- Jupyter notebooks
- Docker Compose for easy setup
- Sample datasets included

---

## Success Criteria

- [ ] All strategies documented with examples
- [ ] Architecture diagrams created
- [ ] Strategy selection guide completed
- [ ] Simple, medium, and complex examples working
- [ ] Integration examples for popular frameworks
- [ ] Docker Compose setup for quick start
- [ ] Jupyter notebooks for exploration
- [ ] API reference auto-generated
- [ ] Contributing guide written
- [ ] Troubleshooting guide covers common issues
- [ ] Documentation hosted (ReadTheDocs, GitHub Pages, etc.)

---

## Documentation Metrics

- Documentation completeness score > 90%
- All public APIs documented
- All strategies have usage examples
- Time to first working example < 30 minutes (user tested)
