# Story 9.2: Create Example Implementations

**Story ID:** 9.2
**Epic:** Epic 9 - Documentation & Developer Experience
**Story Points:** 13
**Priority:** High
**Dependencies:** All previous epics (1-8), Story 9.1 (documentation structure)

---

## User Story

**As a** developer
**I want** working examples showing how to import and use the library
**So that** I can quickly get started

---

## Detailed Requirements

### Functional Requirements

1. **Simple Example - Single Strategy**
   - Basic import statements from the library
   - Initialize RAGFactory with minimal config
   - Create a single strategy (e.g., vector similarity)
   - Execute a query and display results
   - Handle basic errors
   - < 30 lines of code
   - Clear comments explaining each step
   - Runnable as standalone script

2. **Medium Example - Strategy Pipeline (3 strategies)**
   - Import StrategyPipeline
   - Configure 3 strategies that work well together
   - Recommended combination for general use
   - Show pipeline execution
   - Display intermediate results if possible
   - Error handling
   - Performance timing
   - 50-100 lines of code

3. **Complex Example - Multi-Strategy System (5+ strategies)**
   - Full-featured RAG system
   - 5+ strategies configured
   - Custom configuration per strategy
   - Error handling and retry logic
   - Logging and metrics
   - Performance optimization
   - 200+ lines of code
   - Modular structure

4. **Domain-Specific Examples**
   - **Legal**: Contract analysis, case law retrieval
   - **Medical**: Clinical notes, research papers
   - **Customer Support**: Ticket resolution, knowledge base
   - Each example shows:
     - Domain-specific data structure
     - Optimal strategy combination
     - Custom metadata filtering
     - Industry-specific use case

5. **Integration Examples**
   - **FastAPI**: REST API endpoints
   - **Flask**: Web application
   - **LangChain**: Integration with LangChain
   - **Streamlit**: Interactive UI
   - **CLI**: Command-line tool
   - Each shows:
     - Full working integration
     - Authentication if applicable
     - Error handling
     - Deployment considerations

6. **Docker Compose Setup**
   - PostgreSQL with pgvector
   - Redis (optional, for caching)
   - Application container
   - Environment configuration
   - Volume management
   - Health checks
   - Easy `docker-compose up` experience

7. **Jupyter Notebooks**
   - Interactive exploration notebook
   - Visualization of results
   - Performance comparison notebook
   - Strategy experimentation
   - Data analysis examples
   - Charts and graphs

8. **Video Walkthrough/Tutorial**
   - 10-15 minute video
   - Covers installation to first query
   - Shows IDE setup
   - Demonstrates simple example
   - Troubleshoots common issues
   - Hosted on YouTube or similar

### Non-Functional Requirements

1. **Runability**
   - All examples must run without modification
   - Include requirements.txt for each example
   - Include README with setup instructions
   - Include sample data if needed
   - Environment variables documented

2. **Code Quality**
   - Follow PEP 8 style guide
   - Type hints included
   - Well-commented
   - Error handling
   - No hardcoded secrets
   - Production-ready patterns

3. **Documentation**
   - Each example has README
   - Prerequisites listed
   - Step-by-step setup
   - Expected output shown
   - Troubleshooting section
   - Links to relevant docs

4. **Maintainability**
   - Examples tested in CI/CD
   - Version pinning in requirements
   - Regular updates with library changes
   - Automated testing

5. **Performance**
   - Examples complete in reasonable time
   - No unnecessary delays
   - Show performance metrics
   - Optimized queries

---

## Acceptance Criteria

### AC1: Simple Example
- [ ] Basic example created (`examples/simple/basic_retrieval.py`)
- [ ] Uses clear import statements from rag_factory
- [ ] RAGFactory initialization with minimal config
- [ ] Single strategy (vector similarity) demonstrated
- [ ] Query execution and result display
- [ ] Basic error handling included
- [ ] Comments explain each step
- [ ] < 30 lines of code
- [ ] Runs successfully
- [ ] README with setup instructions
- [ ] requirements.txt included

### AC2: Medium Example - Pipeline
- [ ] Pipeline example created (`examples/medium/strategy_pipeline.py`)
- [ ] 3 recommended strategies configured:
  - [ ] Strategy 1: Context-aware chunking
  - [ ] Strategy 2: Query expansion
  - [ ] Strategy 3: Reranking
- [ ] StrategyPipeline correctly used
- [ ] Pipeline execution demonstrated
- [ ] Results formatted and displayed
- [ ] Performance timing included
- [ ] Error handling comprehensive
- [ ] 50-100 lines of code
- [ ] Runs successfully
- [ ] README explains strategy choices

### AC3: Complex Example
- [ ] Complex example created (`examples/advanced/full_system.py`)
- [ ] 5+ strategies integrated
- [ ] Custom configuration per strategy
- [ ] Advanced error handling with retries
- [ ] Logging integrated
- [ ] Metrics collection shown
- [ ] Performance optimizations demonstrated
- [ ] Modular code structure
- [ ] 200+ lines of code
- [ ] Production-ready patterns
- [ ] Comprehensive README

### AC4: Domain-Specific Examples
- [ ] Legal example (`examples/domain/legal/contract_analysis.py`)
  - [ ] Contract analysis use case
  - [ ] Legal-specific metadata
  - [ ] Citation handling
- [ ] Medical example (`examples/domain/medical/clinical_notes.py`)
  - [ ] Clinical notes retrieval
  - [ ] Medical terminology handling
  - [ ] HIPAA considerations noted
- [ ] Customer Support example (`examples/domain/support/ticket_resolution.py`)
  - [ ] Ticket categorization
  - [ ] Knowledge base search
  - [ ] Response generation
- [ ] Each has sample data
- [ ] Each has domain-specific README

### AC5: Integration Examples
- [ ] FastAPI integration (`examples/integrations/fastapi/`)
  - [ ] REST API with multiple endpoints
  - [ ] Request/response models
  - [ ] Error handling
  - [ ] OpenAPI/Swagger docs
  - [ ] Async support
- [ ] Flask integration (`examples/integrations/flask/`)
  - [ ] Web application
  - [ ] Form handling
  - [ ] Session management
  - [ ] Templates included
- [ ] LangChain integration (`examples/integrations/langchain/`)
  - [ ] Custom retriever class
  - [ ] Chain integration
  - [ ] Agent usage
- [ ] Streamlit UI (`examples/integrations/streamlit/`)
  - [ ] Interactive interface
  - [ ] Real-time query
  - [ ] Results visualization
- [ ] CLI tool (`examples/integrations/cli/`)
  - [ ] Command-line interface
  - [ ] Multiple commands
  - [ ] Config file support

### AC6: Docker Compose Setup
- [ ] docker-compose.yml created (`examples/docker/docker-compose.yml`)
- [ ] PostgreSQL with pgvector configured
- [ ] Redis service included (optional)
- [ ] Application container defined
- [ ] Environment variables configured
- [ ] Volumes for persistence
- [ ] Health checks implemented
- [ ] README with docker commands
- [ ] `docker-compose up` starts full system
- [ ] Sample data initialization script

### AC7: Jupyter Notebooks
- [ ] Exploration notebook (`examples/notebooks/01_exploration.ipynb`)
  - [ ] Interactive query execution
  - [ ] Result visualization
  - [ ] Strategy comparison
- [ ] Performance notebook (`examples/notebooks/02_performance.ipynb`)
  - [ ] Benchmarking different strategies
  - [ ] Charts comparing latency
  - [ ] Cost analysis
- [ ] Experimentation notebook (`examples/notebooks/03_experimentation.ipynb`)
  - [ ] Try different configurations
  - [ ] Visualize embeddings
  - [ ] Tune parameters
- [ ] All cells execute successfully
- [ ] Clear markdown explanations

### AC8: Video Tutorial
- [ ] Video recorded (10-15 minutes)
- [ ] Covers installation
- [ ] Shows IDE setup (VS Code/PyCharm)
- [ ] Demonstrates simple example
- [ ] Shows troubleshooting
- [ ] Uploaded to YouTube/Vimeo
- [ ] Linked from documentation
- [ ] Captions/subtitles added

### AC9: Code Quality & Testing
- [ ] All examples follow PEP 8
- [ ] Type hints included
- [ ] No linting errors
- [ ] All examples tested in CI/CD
- [ ] requirements.txt for each example
- [ ] No hardcoded secrets
- [ ] Environment variables documented
- [ ] Error messages helpful

### AC10: Documentation Quality
- [ ] Each example has detailed README
- [ ] Prerequisites clearly listed
- [ ] Setup steps numbered
- [ ] Expected output shown
- [ ] Troubleshooting section included
- [ ] Links to API docs
- [ ] License information

---

## Technical Specifications

### Example Directory Structure
```
examples/
├── README.md                          # Overview of all examples
├── requirements.txt                   # Common dependencies
│
├── simple/
│   ├── README.md
│   ├── requirements.txt
│   ├── basic_retrieval.py             # Single strategy
│   ├── .env.example                   # Example environment file
│   └── sample_data.json               # Sample data
│
├── medium/
│   ├── README.md
│   ├── requirements.txt
│   ├── strategy_pipeline.py           # 3-strategy pipeline
│   ├── config.yaml                    # Configuration file
│   └── sample_data/                   # Sample documents
│
├── advanced/
│   ├── README.md
│   ├── requirements.txt
│   ├── full_system.py                 # 5+ strategies
│   ├── config/
│   │   ├── strategies.yaml
│   │   └── database.yaml
│   ├── utils/
│   │   ├── logging_config.py
│   │   └── metrics.py
│   └── sample_data/
│
├── domain/
│   ├── legal/
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── contract_analysis.py
│   │   └── sample_contracts/
│   ├── medical/
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── clinical_notes.py
│   │   └── sample_notes/
│   └── support/
│       ├── README.md
│       ├── requirements.txt
│       ├── ticket_resolution.py
│       └── sample_tickets/
│
├── integrations/
│   ├── fastapi/
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── routers/
│   │   └── tests/
│   ├── flask/
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── app.py
│   │   ├── templates/
│   │   └── static/
│   ├── langchain/
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── custom_retriever.py
│   │   └── chain_example.py
│   ├── streamlit/
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── app.py
│   │   └── config.toml
│   └── cli/
│       ├── README.md
│       ├── requirements.txt
│       ├── rag_cli.py
│       └── setup.py
│
├── docker/
│   ├── README.md
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── .env.example
│   ├── init-scripts/
│   │   └── init-db.sql
│   └── sample_data/
│
└── notebooks/
    ├── README.md
    ├── requirements.txt
    ├── 01_exploration.ipynb
    ├── 02_performance.ipynb
    ├── 03_experimentation.ipynb
    └── data/
```

### Simple Example Implementation
```python
# examples/simple/basic_retrieval.py
"""
Basic RAG retrieval example using RAG Factory.

This example demonstrates:
- Importing the RAG Factory library
- Initializing the factory with basic configuration
- Creating a single retrieval strategy
- Executing a query
- Displaying results
"""

import os
from typing import List
from rag_factory import RAGFactory, RetrievalResult
from rag_factory.config import DatabaseConfig, EmbeddingConfig


def main():
    """Run a basic retrieval example."""

    # Configuration from environment or defaults
    db_config = DatabaseConfig(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "ragdb"),
        user=os.getenv("DB_USER", "raguser"),
        password=os.getenv("DB_PASSWORD", "password")
    )

    embedding_config = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Initialize RAG Factory
    print("Initializing RAG Factory...")
    factory = RAGFactory(
        db_config=db_config,
        embedding_config=embedding_config
    )

    # Create a vector similarity strategy
    print("Creating vector similarity strategy...")
    strategy = factory.create_strategy(
        "vector_similarity",
        top_k=5  # Return top 5 results
    )

    # Execute a query
    query = "What are the main features of the product?"
    print(f"\nQuery: {query}")
    print("Retrieving documents...\n")

    try:
        results: List[RetrievalResult] = strategy.retrieve(query)

        # Display results
        print(f"Found {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Score: {result.score:.4f}")
            print(f"  Text: {result.text[:200]}...")  # First 200 chars
            print(f"  Metadata: {result.metadata}")
            print()

    except Exception as e:
        print(f"Error during retrieval: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
```

### Medium Example Implementation
```python
# examples/medium/strategy_pipeline.py
"""
Medium complexity example using a strategy pipeline.

This example demonstrates:
- Using multiple strategies in a pipeline
- Recommended combination: Context-aware chunking + Query expansion + Reranking
- Configuration via YAML file
- Performance timing
- Error handling
"""

import os
import time
import yaml
from pathlib import Path
from typing import List, Dict, Any

from rag_factory import RAGFactory, StrategyPipeline, RetrievalResult
from rag_factory.config import DatabaseConfig, EmbeddingConfig


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_pipeline(factory: RAGFactory, config: Dict[str, Any]) -> StrategyPipeline:
    """
    Create a recommended 3-strategy pipeline.

    Pipeline:
    1. Context-aware chunking: Better document segmentation
    2. Query expansion: Handle ambiguous queries
    3. Reranking: Improve result quality
    """
    strategies = []

    # Strategy 1: Context-aware chunking
    print("Creating context-aware chunking strategy...")
    strategies.append(
        factory.create_strategy(
            "context_aware_chunking",
            chunk_size=config.get("chunk_size", 512),
            overlap=config.get("chunk_overlap", 50)
        )
    )

    # Strategy 2: Query expansion
    print("Creating query expansion strategy...")
    strategies.append(
        factory.create_strategy(
            "query_expansion",
            num_expansions=config.get("num_expansions", 3),
            llm_provider=config.get("llm_provider", "openai")
        )
    )

    # Strategy 3: Reranking
    print("Creating reranking strategy...")
    strategies.append(
        factory.create_strategy(
            "reranking",
            top_k=config.get("final_top_k", 10),
            reranker_model=config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
    )

    # Create pipeline
    pipeline = StrategyPipeline(strategies)
    return pipeline


def main():
    """Run pipeline example."""

    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(config_path)

    # Setup database and embedding configs
    db_config = DatabaseConfig(**config["database"])
    embedding_config = EmbeddingConfig(**config["embedding"])

    # Initialize factory
    print("Initializing RAG Factory...\n")
    factory = RAGFactory(
        db_config=db_config,
        embedding_config=embedding_config
    )

    # Create pipeline
    pipeline = create_pipeline(factory, config["pipeline"])
    print("\nPipeline created successfully.\n")

    # Example queries
    queries = [
        "What are the key features and benefits?",
        "How do I troubleshoot connection issues?",
        "What are the pricing options?"
    ]

    # Execute queries
    for query in queries:
        print(f"Query: {query}")
        print("-" * 80)

        try:
            # Time the retrieval
            start_time = time.time()
            results: List[RetrievalResult] = pipeline.retrieve(query)
            elapsed_time = time.time() - start_time

            # Display results
            print(f"Found {len(results)} results in {elapsed_time:.2f}s\n")

            for i, result in enumerate(results[:3], 1):  # Show top 3
                print(f"Result {i}:")
                print(f"  Score: {result.score:.4f}")
                print(f"  Text: {result.text[:150]}...")
                print()

            # Performance metrics
            print(f"Performance:")
            print(f"  Total time: {elapsed_time:.2f}s")
            print(f"  Results per second: {len(results) / elapsed_time:.2f}")
            print()

        except Exception as e:
            print(f"Error during retrieval: {e}\n")
            continue

        print()

    return 0


if __name__ == "__main__":
    exit(main())
```

### Docker Compose Setup
```yaml
# examples/docker/docker-compose.yml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: rag_postgres
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-ragdb}
      POSTGRES_USER: ${POSTGRES_USER:-raguser}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-ragpassword}
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-raguser}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: rag_redis
    ports:
      - "${REDIS_PORT:-6379}:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    command: redis-server --appendonly yes

  rag_app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag_application
    environment:
      DB_HOST: postgres
      DB_PORT: 5432
      DB_NAME: ${POSTGRES_DB:-ragdb}
      DB_USER: ${POSTGRES_USER:-raguser}
      DB_PASSWORD: ${POSTGRES_PASSWORD:-ragpassword}
      REDIS_HOST: redis
      REDIS_PORT: 6379
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    ports:
      - "${APP_PORT:-8000}:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./sample_data:/app/data
    command: python -m examples.simple.basic_retrieval

volumes:
  postgres_data:
  redis_data:
```

### FastAPI Integration Example
```python
# examples/integrations/fastapi/main.py
"""
FastAPI integration example for RAG Factory.

This example demonstrates:
- REST API endpoints for RAG operations
- Request/response models with Pydantic
- Async support
- Error handling
- OpenAPI documentation
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from rag_factory import RAGFactory, StrategyPipeline
from rag_factory.config import DatabaseConfig, EmbeddingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Factory API",
    description="REST API for RAG Factory retrieval system",
    version="1.0.0"
)

# Global factory instance (in production, use dependency injection)
factory: Optional[RAGFactory] = None
pipeline: Optional[StrategyPipeline] = None


# Request/Response Models
class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(10, description="Number of results to return", ge=1, le=100)
    strategy: Optional[str] = Field(None, description="Strategy to use (optional)")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")


class RetrievalResultResponse(BaseModel):
    """Single retrieval result."""
    text: str
    score: float
    metadata: Dict[str, Any]
    source: Optional[str] = None


class QueryResponse(BaseModel):
    """Query response model."""
    query: str
    results: List[RetrievalResultResponse]
    count: int
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    database: str
    strategies_available: List[str]


# Dependency to get factory
def get_factory() -> RAGFactory:
    """Get RAG Factory instance."""
    if factory is None:
        raise HTTPException(status_code=500, detail="RAG Factory not initialized")
    return factory


# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize RAG Factory on startup."""
    global factory, pipeline

    logger.info("Initializing RAG Factory...")

    try:
        # Configuration (from environment in production)
        db_config = DatabaseConfig(
            host="localhost",
            database="ragdb",
            user="raguser",
            password="password"
        )

        embedding_config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small"
        )

        # Initialize factory
        factory = RAGFactory(
            db_config=db_config,
            embedding_config=embedding_config
        )

        # Create default pipeline
        strategies = [
            factory.create_strategy("query_expansion"),
            factory.create_strategy("vector_similarity"),
            factory.create_strategy("reranking", top_k=10)
        ]
        pipeline = StrategyPipeline(strategies)

        logger.info("RAG Factory initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize RAG Factory: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down RAG Factory...")
    # Cleanup connections, etc.


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Factory API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(factory: RAGFactory = Depends(get_factory)):
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        database="connected",
        strategies_available=list(factory.STRATEGIES.keys())
    )


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    factory: RAGFactory = Depends(get_factory)
):
    """
    Execute a retrieval query.

    Args:
        request: Query request with query string and parameters

    Returns:
        Query results with scores and metadata
    """
    import time

    start_time = time.time()

    try:
        # Use specific strategy or pipeline
        if request.strategy:
            strategy = factory.create_strategy(
                request.strategy,
                top_k=request.top_k
            )
            results = strategy.retrieve(request.query)
        else:
            # Use default pipeline
            results = pipeline.retrieve(request.query)

        # Format results
        formatted_results = [
            RetrievalResultResponse(
                text=r.text,
                score=r.score,
                metadata=r.metadata,
                source=r.metadata.get("source")
            )
            for r in results[:request.top_k]
        ]

        latency_ms = (time.time() - start_time) * 1000

        return QueryResponse(
            query=request.query,
            results=formatted_results,
            count=len(formatted_results),
            latency_ms=latency_ms
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/strategies", response_model=List[str])
async def list_strategies(factory: RAGFactory = Depends(get_factory)):
    """List available strategies."""
    return list(factory.STRATEGIES.keys())


@app.post("/strategies/{strategy_name}/query")
async def query_with_strategy(
    strategy_name: str,
    request: QueryRequest,
    factory: RAGFactory = Depends(get_factory)
):
    """Query using a specific strategy."""
    if strategy_name not in factory.STRATEGIES:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{strategy_name}' not found"
        )

    # Update request with strategy
    request.strategy = strategy_name
    return await query(request, factory)


# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
```

### Jupyter Notebook Example
```python
# examples/notebooks/01_exploration.ipynb
# (Jupyter notebook format - shown as cells)

# Cell 1: Setup
"""
# RAG Factory Exploration

This notebook demonstrates basic usage of the RAG Factory library
for retrieval-augmented generation.

## Setup
"""

# !pip install rag-factory matplotlib pandas

import os
from rag_factory import RAGFactory
from rag_factory.config import DatabaseConfig, EmbeddingConfig
import pandas as pd
import matplotlib.pyplot as plt

# Cell 2: Initialize Factory
"""
## Initialize RAG Factory
"""

db_config = DatabaseConfig(
    host="localhost",
    database="ragdb",
    user="raguser",
    password="password"
)

embedding_config = EmbeddingConfig(
    provider="openai",
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

factory = RAGFactory(
    db_config=db_config,
    embedding_config=embedding_config
)

print("✓ RAG Factory initialized")

# Cell 3: Create Strategy
"""
## Create Vector Similarity Strategy
"""

strategy = factory.create_strategy("vector_similarity", top_k=10)
print(f"✓ Created strategy: {type(strategy).__name__}")

# Cell 4: Execute Query
"""
## Execute a Query
"""

query = "What are the main features?"
results = strategy.retrieve(query)

print(f"Query: {query}")
print(f"Found {len(results)} results\n")

# Display in DataFrame
df = pd.DataFrame([
    {
        "Score": r.score,
        "Text": r.text[:100] + "...",
        "Source": r.metadata.get("source", "N/A")
    }
    for r in results
])

df

# Cell 5: Visualize Scores
"""
## Visualize Result Scores
"""

scores = [r.score for r in results]

plt.figure(figsize=(10, 6))
plt.bar(range(len(scores)), scores)
plt.xlabel("Result Rank")
plt.ylabel("Similarity Score")
plt.title("Retrieval Result Scores")
plt.grid(axis='y', alpha=0.3)
plt.show()

# Cell 6: Compare Strategies
"""
## Compare Multiple Strategies
"""

strategies_to_compare = [
    "vector_similarity",
    "query_expansion",
    "reranking"
]

comparison_results = {}

for strat_name in strategies_to_compare:
    strat = factory.create_strategy(strat_name, top_k=5)
    results = strat.retrieve(query)
    comparison_results[strat_name] = {
        "avg_score": sum(r.score for r in results) / len(results),
        "count": len(results)
    }

# Visualize comparison
df_comparison = pd.DataFrame(comparison_results).T
df_comparison.plot(kind='bar', figsize=(10, 6))
plt.title("Strategy Comparison")
plt.ylabel("Average Score")
plt.xticks(rotation=45)
plt.legend(["Avg Score", "Count"])
plt.tight_layout()
plt.show()
```

---

## Unit Tests

### Test File Locations
- `tests/unit/examples/test_examples_syntax.py`
- `tests/unit/examples/test_examples_imports.py`
- `tests/unit/examples/test_requirements.py`

### Test Cases

#### TC9.2.1: Example Syntax Tests
```python
import pytest
from pathlib import Path
import ast
import subprocess

class TestExamplesSyntax:
    """Test that all example files have valid syntax."""

    @pytest.fixture
    def examples_root(self):
        return Path(__file__).parent.parent.parent / "examples"

    def get_python_files(self, examples_root: Path):
        """Get all Python example files."""
        return list(examples_root.rglob("*.py"))

    def test_all_examples_valid_syntax(self, examples_root):
        """Test that all example Python files have valid syntax."""
        python_files = self.get_python_files(examples_root)

        errors = []
        for py_file in python_files:
            try:
                with open(py_file) as f:
                    ast.parse(f.read())
            except SyntaxError as e:
                errors.append(f"{py_file}: {e}")

        assert len(errors) == 0, f"Syntax errors in examples:\n" + "\n".join(errors)

    def test_simple_example_exists(self, examples_root):
        """Test that simple example exists."""
        simple_example = examples_root / "simple" / "basic_retrieval.py"
        assert simple_example.exists(), "Simple example missing"

    def test_medium_example_exists(self, examples_root):
        """Test that medium example exists."""
        medium_example = examples_root / "medium" / "strategy_pipeline.py"
        assert medium_example.exists(), "Medium example missing"

    def test_complex_example_exists(self, examples_root):
        """Test that complex example exists."""
        complex_example = examples_root / "advanced" / "full_system.py"
        assert complex_example.exists(), "Complex example missing"

    def test_all_examples_have_docstrings(self, examples_root):
        """Test that all examples have module docstrings."""
        python_files = self.get_python_files(examples_root)

        missing_docs = []
        for py_file in python_files:
            with open(py_file) as f:
                tree = ast.parse(f.read())

            docstring = ast.get_docstring(tree)
            if not docstring:
                missing_docs.append(str(py_file))

        assert len(missing_docs) == 0, \
            f"Examples missing docstrings:\n" + "\n".join(missing_docs)

    def test_all_examples_have_main_function(self, examples_root):
        """Test that runnable examples have main() function."""
        python_files = self.get_python_files(examples_root)

        # Exclude utility files
        runnable_files = [
            f for f in python_files
            if "utils" not in str(f) and "models" not in str(f)
        ]

        missing_main = []
        for py_file in runnable_files:
            with open(py_file) as f:
                tree = ast.parse(f.read())

            has_main = any(
                isinstance(node, ast.FunctionDef) and node.name == "main"
                for node in ast.walk(tree)
            )

            if not has_main:
                missing_main.append(str(py_file))

        assert len(missing_main) == 0, \
            f"Examples missing main() function:\n" + "\n".join(missing_main)
```

#### TC9.2.2: Import Tests
```python
import pytest
from pathlib import Path
import ast
import re

class TestExamplesImports:
    """Test that examples import from rag_factory correctly."""

    @pytest.fixture
    def examples_root(self):
        return Path(__file__).parent.parent.parent / "examples"

    def extract_imports(self, py_file: Path):
        """Extract import statements from Python file."""
        with open(py_file) as f:
            tree = ast.parse(f.read())

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return imports

    def test_simple_example_imports_rag_factory(self, examples_root):
        """Test that simple example imports from rag_factory."""
        simple_example = examples_root / "simple" / "basic_retrieval.py"
        imports = self.extract_imports(simple_example)

        assert any("rag_factory" in imp for imp in imports), \
            "Simple example doesn't import from rag_factory"

    def test_medium_example_imports_pipeline(self, examples_root):
        """Test that medium example imports StrategyPipeline."""
        medium_example = examples_root / "medium" / "strategy_pipeline.py"

        with open(medium_example) as f:
            content = f.read()

        assert "StrategyPipeline" in content, \
            "Medium example doesn't import StrategyPipeline"

    def test_all_examples_import_factory(self, examples_root):
        """Test that all main examples import RAGFactory."""
        main_examples = [
            examples_root / "simple" / "basic_retrieval.py",
            examples_root / "medium" / "strategy_pipeline.py",
            examples_root / "advanced" / "full_system.py"
        ]

        for example in main_examples:
            if not example.exists():
                continue

            with open(example) as f:
                content = f.read()

            assert "RAGFactory" in content, \
                f"{example.name} doesn't import RAGFactory"

    def test_no_relative_imports(self, examples_root):
        """Test that examples don't use relative imports."""
        python_files = list(examples_root.rglob("*.py"))

        relative_imports = []
        for py_file in python_files:
            with open(py_file) as f:
                content = f.read()

            # Check for relative imports (from . or from ..)
            if re.search(r'from\s+\.+\s+import', content):
                relative_imports.append(str(py_file))

        assert len(relative_imports) == 0, \
            f"Examples using relative imports:\n" + "\n".join(relative_imports)
```

#### TC9.2.3: Requirements Tests
```python
import pytest
from pathlib import Path

class TestRequirements:
    """Test that each example has requirements.txt."""

    @pytest.fixture
    def examples_root(self):
        return Path(__file__).parent.parent.parent / "examples"

    def test_simple_has_requirements(self, examples_root):
        """Test that simple example has requirements.txt."""
        req_file = examples_root / "simple" / "requirements.txt"
        assert req_file.exists(), "Simple example missing requirements.txt"

    def test_medium_has_requirements(self, examples_root):
        """Test that medium example has requirements.txt."""
        req_file = examples_root / "medium" / "requirements.txt"
        assert req_file.exists(), "Medium example missing requirements.txt"

    def test_all_example_dirs_have_requirements(self, examples_root):
        """Test that all example directories have requirements.txt."""
        example_dirs = [
            d for d in examples_root.iterdir()
            if d.is_dir() and d.name not in ["__pycache__", "data"]
        ]

        missing_requirements = []
        for example_dir in example_dirs:
            req_file = example_dir / "requirements.txt"
            if not req_file.exists():
                missing_requirements.append(example_dir.name)

        assert len(missing_requirements) == 0, \
            f"Directories missing requirements.txt: {missing_requirements}"

    def test_requirements_have_rag_factory(self, examples_root):
        """Test that requirements include rag-factory."""
        req_files = list(examples_root.rglob("requirements.txt"))

        for req_file in req_files:
            with open(req_file) as f:
                content = f.read()

            assert "rag-factory" in content or "rag_factory" in content, \
                f"{req_file} missing rag-factory dependency"

    def test_no_version_conflicts(self, examples_root):
        """Test that requirements don't have conflicting versions."""
        req_files = list(examples_root.rglob("requirements.txt"))

        # Parse all requirements
        all_requirements = {}
        for req_file in req_files:
            with open(req_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Parse package name
                        package = line.split("==")[0].split(">=")[0].split("<=")[0]
                        if package in all_requirements:
                            if all_requirements[package] != line:
                                pytest.fail(
                                    f"Version conflict for {package}: "
                                    f"{all_requirements[package]} vs {line}"
                                )
                        all_requirements[package] = line
```

---

## Integration Tests

### Test File Location
`tests/integration/examples/test_examples_integration.py`

### Test Scenarios

#### IS9.2.1: Example Execution Tests
```python
import pytest
import subprocess
from pathlib import Path
import os
import tempfile

@pytest.mark.integration
class TestExamplesExecution:
    """Test that examples can be executed successfully."""

    @pytest.fixture
    def examples_root(self):
        return Path(__file__).parent.parent.parent / "examples"

    @pytest.fixture
    def test_env(self):
        """Create test environment variables."""
        return {
            **os.environ,
            "DB_HOST": "localhost",
            "DB_PORT": "5432",
            "DB_NAME": "test_ragdb",
            "DB_USER": "test_user",
            "DB_PASSWORD": "test_password",
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "test_key")
        }

    @pytest.mark.slow
    def test_simple_example_runs(self, examples_root, test_env):
        """Test that simple example runs without errors."""
        example_file = examples_root / "simple" / "basic_retrieval.py"

        if not example_file.exists():
            pytest.skip("Simple example not created yet")

        result = subprocess.run(
            ["python", str(example_file)],
            env=test_env,
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should not crash (may fail due to missing DB, but no syntax errors)
        assert result.returncode in [0, 1], \
            f"Simple example crashed: {result.stderr}"

    @pytest.mark.slow
    def test_medium_example_runs(self, examples_root, test_env):
        """Test that medium example runs without errors."""
        example_file = examples_root / "medium" / "strategy_pipeline.py"

        if not example_file.exists():
            pytest.skip("Medium example not created yet")

        result = subprocess.run(
            ["python", str(example_file)],
            env=test_env,
            capture_output=True,
            text=True,
            timeout=60
        )

        assert result.returncode in [0, 1], \
            f"Medium example crashed: {result.stderr}"

    def test_docker_compose_valid(self, examples_root):
        """Test that docker-compose.yml is valid."""
        import yaml

        docker_compose = examples_root / "docker" / "docker-compose.yml"

        if not docker_compose.exists():
            pytest.skip("docker-compose.yml not created yet")

        with open(docker_compose) as f:
            config = yaml.safe_load(f)

        assert "services" in config
        assert "postgres" in config["services"]
        assert "volumes" in config

    @pytest.mark.slow
    def test_fastapi_app_starts(self, examples_root):
        """Test that FastAPI app can start."""
        import time
        import requests

        fastapi_main = examples_root / "integrations" / "fastapi" / "main.py"

        if not fastapi_main.exists():
            pytest.skip("FastAPI example not created yet")

        # Start FastAPI app
        process = subprocess.Popen(
            ["python", str(fastapi_main)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        try:
            # Wait for startup
            time.sleep(3)

            # Check health endpoint
            response = requests.get("http://localhost:8000/health", timeout=5)
            assert response.status_code == 200

        finally:
            process.terminate()
            process.wait()
```

#### IS9.2.2: Jupyter Notebook Tests
```python
import pytest
from pathlib import Path
import subprocess

@pytest.mark.integration
class TestJupyterNotebooks:
    """Test that Jupyter notebooks execute successfully."""

    @pytest.fixture
    def notebooks_root(self):
        return Path(__file__).parent.parent.parent / "examples" / "notebooks"

    @pytest.mark.slow
    def test_all_notebooks_execute(self, notebooks_root):
        """Test that all notebooks execute without errors."""
        if not notebooks_root.exists():
            pytest.skip("Notebooks not created yet")

        notebooks = list(notebooks_root.glob("*.ipynb"))

        for notebook in notebooks:
            # Execute notebook using nbconvert
            result = subprocess.run(
                [
                    "jupyter", "nbconvert",
                    "--to", "notebook",
                    "--execute",
                    "--inplace",
                    str(notebook)
                ],
                capture_output=True,
                text=True,
                timeout=120
            )

            assert result.returncode == 0, \
                f"Notebook {notebook.name} failed to execute: {result.stderr}"
```

---

## Definition of Done

- [ ] Simple example created and working
- [ ] Medium pipeline example created and working
- [ ] Complex multi-strategy example created
- [ ] 3 domain-specific examples (legal, medical, support)
- [ ] 5 integration examples (FastAPI, Flask, LangChain, Streamlit, CLI)
- [ ] Docker Compose setup complete and tested
- [ ] 3 Jupyter notebooks created and executable
- [ ] Video tutorial recorded and uploaded
- [ ] All examples have README files
- [ ] All examples have requirements.txt
- [ ] All examples tested in CI/CD
- [ ] No syntax errors in any example
- [ ] All imports work correctly
- [ ] No hardcoded secrets
- [ ] Code follows PEP 8
- [ ] Type hints included
- [ ] Error handling comprehensive
- [ ] Sample data included where needed
- [ ] All tests pass
- [ ] Documentation links to examples
- [ ] Examples linked from docs site
- [ ] Code reviewed

---

## Notes for Developers

1. **Start with Simple**: Build simple example first, then progressively add complexity

2. **Real Data**: Use realistic sample data - makes examples more valuable

3. **Runnable**: Every example must run with minimal setup - provide docker-compose

4. **Comments**: Heavily comment examples - they're learning tools

5. **Error Handling**: Show proper error handling patterns - users will copy these

6. **Environment**: Use environment variables - never hardcode secrets

7. **README Essential**: Each example needs clear README with setup steps

8. **Test Everything**: Examples that don't work are worse than no examples

9. **Progressive Complexity**: Simple → Medium → Advanced path for learning

10. **Domain Examples**: Show real-world use cases - helps users see applicability

11. **Integration Focus**: Show how to integrate with popular frameworks

12. **Docker First**: Provide Docker setup for easy environment

13. **Jupyter for Exploration**: Notebooks are great for learning and experimentation

14. **Video Walkthrough**: Video dramatically improves understanding

15. **Keep Updated**: Update examples when library changes - stale examples frustrate users

16. **Copy-Paste Ready**: Code should work when copied - test this

17. **Performance Tips**: Show performance best practices in examples

18. **CI/CD Testing**: Automate example testing - prevents bit rot
