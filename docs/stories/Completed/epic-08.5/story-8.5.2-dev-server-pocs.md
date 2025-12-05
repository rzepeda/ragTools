# Story 8.5.2: Create Lightweight Dev Server for POCs

**Epic:** 8.5 - Development Tools (CLI & Dev Server)
**Story Points:** 8
**Priority:** Medium
**Dependencies:** Epic 4 (RAG Strategies)

---

## User Story

**As a** developer
**I want** a simple HTTP server for demos and POCs
**So that** I can test integrations without building a full client

---

## Business Value

- Enables rapid prototyping of RAG applications
- Provides HTTP interface for testing frontend integrations
- Facilitates demonstrations to stakeholders
- Validates library integration patterns
- Reduces time to create POC applications
- Educational tool for learning RAG system integration

---

## Requirements

### Functional Requirements

#### FR-1: Query Endpoint
- **Requirement:** Server must provide POST endpoint for querying indexed documents
- **Endpoint:** `POST /query`
- **Input:** JSON payload with query string, strategies list, optional parameters
- **Output:** JSON response with results, metadata, timing information
- **Request Format:**
  ```json
  {
    "query": "What are the action items?",
    "strategies": ["reranking", "query_expansion"],
    "top_k": 5,
    "config": {}
  }
  ```
- **Response Format:**
  ```json
  {
    "results": [
      {
        "text": "Document content...",
        "score": 0.95,
        "metadata": {...}
      }
    ],
    "timing": {
      "total_ms": 150,
      "strategy_breakdown": {...}
    },
    "query": "What are the action items?",
    "strategies_used": ["reranking", "query_expansion"]
  }
  ```

#### FR-2: Indexing Endpoint
- **Requirement:** Server must provide POST endpoint for indexing documents
- **Endpoint:** `POST /index`
- **Input:** JSON payload with documents or file paths, strategy, configuration
- **Output:** JSON response with indexing statistics
- **Request Format:**
  ```json
  {
    "path": "./docs",
    "strategy": "context_aware_chunking",
    "config": {}
  }
  ```
  Or:
  ```json
  {
    "documents": [
      {"text": "Document content...", "metadata": {...}},
      {"text": "Another document...", "metadata": {...}}
    ],
    "strategy": "semantic_chunking"
  }
  ```
- **Response Format:**
  ```json
  {
    "status": "success",
    "documents_indexed": 42,
    "chunks_created": 156,
    "time_ms": 3420,
    "strategy_used": "context_aware_chunking"
  }
  ```

#### FR-3: Strategy Listing Endpoint
- **Requirement:** Server must provide GET endpoint to list available strategies
- **Endpoint:** `GET /strategies`
- **Query Parameters:** Optional type filter
- **Output:** JSON list of available strategies with metadata
- **Response Format:**
  ```json
  {
    "strategies": [
      {
        "name": "reranking",
        "type": "retrieval",
        "description": "Reranks retrieved documents...",
        "parameters": {...}
      }
    ]
  }
  ```

#### FR-4: Health Check Endpoint
- **Requirement:** Server must provide health check endpoint
- **Endpoint:** `GET /health`
- **Output:** Server status and basic statistics
- **Response Format:**
  ```json
  {
    "status": "healthy",
    "uptime_seconds": 3600,
    "indexed_documents": 42,
    "version": "0.1.0"
  }
  ```

#### FR-5: Configuration Hot-Reloading
- **Requirement:** Server must reload configuration without restart
- **Endpoint:** `POST /config/reload`
- **Input:** Optional path to new config file
- **Output:** Success/failure status
- **Trigger:** File system watcher or manual endpoint call

#### FR-6: Simple HTML UI
- **Requirement:** Server must serve a single-page HTML UI for manual testing
- **Endpoint:** `GET /` (root)
- **Features:**
  - Form for submitting queries
  - Strategy selection checkboxes
  - Results display area
  - Index management interface
  - Request/response logging view
- **Technology:** Plain HTML/JavaScript/CSS (no build step)

#### FR-7: Request/Response Logging
- **Requirement:** Server must log all requests and responses
- **Endpoint:** `GET /logs` (returns recent logs)
- **Output:** JSON array of log entries
- **Storage:** In-memory (last 100 requests)
- **Information:** Timestamp, endpoint, request body, response status, timing

#### FR-8: OpenAPI Documentation
- **Requirement:** Server must provide auto-generated API documentation
- **Endpoint:** `GET /docs` (Swagger UI)
- **Endpoint:** `GET /redoc` (ReDoc UI)
- **Endpoint:** `GET /openapi.json` (OpenAPI spec)

### Non-Functional Requirements

#### NFR-1: Performance
- Response time < 5 seconds for typical queries
- Support for async request handling
- Streaming responses for large result sets
- Efficient memory usage

#### NFR-2: Usability
- Clear error messages with suggestions
- CORS enabled for frontend development
- Pretty-printed JSON responses (optional)
- Comprehensive API documentation

#### NFR-3: Development-Only Focus
- **EXPLICIT WARNINGS** that this is not production-ready
- No authentication/authorization
- No rate limiting
- No horizontal scaling support
- Minimal error handling
- Single-process operation
- File-based storage only

#### NFR-4: Compatibility
- Python 3.8+ support
- Cross-platform (Linux, macOS, Windows)
- Works with all strategies from Epic 4
- Standard HTTP/REST conventions

#### NFR-5: Architecture
- Single-file implementation (~200 lines core logic)
- Uses library's public API only
- FastAPI framework for simplicity
- Uvicorn ASGI server

---

## Acceptance Criteria

### AC-1: Query Endpoint Functionality
- **Given** indexed documents exist in the system
- **When** I POST to `/query` with:
  ```json
  {
    "query": "What are the action items?",
    "strategies": ["reranking", "query_expansion"]
  }
  ```
- **Then** the server should:
  - Return 200 OK status
  - Include relevant results with scores
  - Provide timing information
  - Apply specified strategies in order
  - Handle missing strategies with 400 error
  - Handle empty index with 404 error

### AC-2: Index Endpoint Functionality
- **Given** documents to index
- **When** I POST to `/index` with path or document array
- **Then** the server should:
  - Process all documents using specified strategy
  - Return indexing statistics
  - Handle invalid paths with 400 error
  - Handle invalid strategy with 400 error
  - Support both file paths and inline documents

### AC-3: Strategy Listing
- **Given** the library has strategies installed
- **When** I GET `/strategies`
- **Then** the server should:
  - Return list of all available strategies
  - Include strategy metadata (name, type, description)
  - Support filtering by type parameter
  - Return 200 OK with valid JSON

### AC-4: Health Check
- **Given** the server is running
- **When** I GET `/health`
- **Then** the server should:
  - Return 200 OK
  - Include uptime information
  - Include system statistics
  - Respond within 100ms

### AC-5: Configuration Hot-Reload
- **Given** the server is running with a config file
- **When** I modify the config file or POST to `/config/reload`
- **Then** the server should:
  - Reload configuration without restart
  - Apply new configuration to subsequent requests
  - Return success/failure status
  - Log configuration changes

### AC-6: HTML UI Functionality
- **Given** the server is running
- **When** I navigate to `http://localhost:8000/`
- **Then** I should see:
  - Query input form with strategy selection
  - Submit button that sends requests
  - Results display area showing formatted results
  - Index management section
  - Request/response log viewer
  - No external dependencies (works offline)

### AC-7: Request Logging
- **Given** the server has processed requests
- **When** I GET `/logs`
- **Then** the server should:
  - Return array of recent request logs (last 100)
  - Include timestamp, endpoint, status, timing
  - Format logs as JSON
  - Support filtering by endpoint or status

### AC-8: OpenAPI Documentation
- **Given** the server is running
- **When** I navigate to `/docs`
- **Then** I should see:
  - Auto-generated Swagger UI
  - All endpoints documented
  - Request/response schemas
  - Interactive API testing interface

### AC-9: Error Handling
- **Given** invalid requests
- **When** I send malformed JSON or missing fields
- **Then** the server should:
  - Return appropriate HTTP status codes (400, 404, 500)
  - Include clear error messages
  - Suggest corrections when possible
  - Not crash or show stack traces (unless debug mode)

### AC-10: Development-Only Warnings
- **Given** the server is started
- **When** The server starts
- **Then** it should:
  - Display prominent warning in console
  - Include warning in HTML UI
  - Document limitations in API docs
  - Log warning: "⚠️ DEVELOPMENT SERVER - NOT FOR PRODUCTION USE"

---

## Technical Implementation

### Technology Stack

- **Web Framework:** FastAPI - async support, automatic OpenAPI docs
- **ASGI Server:** Uvicorn - fast, lightweight server
- **Configuration:** Pydantic Settings - type-safe configuration
- **HTML/JS:** Vanilla JavaScript - no build step required
- **Logging:** Python logging module + structlog
- **Testing:** Pytest + HTTPX (async HTTP client)

### Architecture

```
rag_factory_server/
├── __init__.py
├── main.py              # FastAPI app, all endpoints (~200 lines)
├── models.py            # Pydantic request/response models
├── config.py            # Configuration management
├── static/
│   └── index.html       # Single-page UI
└── utils/
    ├── __init__.py
    ├── logging.py       # Request logging utilities
    └── library.py       # Library integration wrapper
```

### Module Specifications

#### main.py
```python
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(
    title="RAG Factory Dev Server",
    description="⚠️ DEVELOPMENT SERVER - NOT FOR PRODUCTION USE",
    version="0.1.0"
)

# Endpoints
@app.post("/query")
async def query_documents(request: QueryRequest) -> QueryResponse:
    """Query indexed documents with specified strategies"""
    pass

@app.post("/index")
async def index_documents(request: IndexRequest) -> IndexResponse:
    """Index documents using specified strategy"""
    pass

@app.get("/strategies")
async def list_strategies(type: Optional[str] = None) -> StrategiesResponse:
    """List available strategies"""
    pass

@app.get("/health")
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    pass

@app.post("/config/reload")
async def reload_config(config_path: Optional[str] = None) -> ConfigResponse:
    """Reload configuration"""
    pass

@app.get("/logs")
async def get_logs(limit: int = 100) -> LogsResponse:
    """Get recent request logs"""
    pass

@app.get("/")
async def serve_ui() -> HTMLResponse:
    """Serve simple HTML UI"""
    pass

def start_server(host: str = "0.0.0.0", port: int = 8000, config: str = None):
    """Start the dev server with warnings"""
    print("⚠️" * 50)
    print("⚠️  DEVELOPMENT SERVER - NOT FOR PRODUCTION USE  ⚠️")
    print("⚠️  No authentication, rate limiting, or security  ⚠️")
    print("⚠️" * 50)
    uvicorn.run(app, host=host, port=port)
```

#### models.py
```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., description="Query string")
    strategies: List[str] = Field(default=[], description="Strategies to apply")
    top_k: int = Field(default=5, ge=1, le=100)
    config: Dict[str, Any] = Field(default={})

class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    timing: Dict[str, float]
    query: str
    strategies_used: List[str]

class IndexRequest(BaseModel):
    path: Optional[str] = None
    documents: Optional[List[Dict[str, Any]]] = None
    strategy: str
    config: Dict[str, Any] = Field(default={})

class IndexResponse(BaseModel):
    status: str
    documents_indexed: int
    chunks_created: int
    time_ms: float
    strategy_used: str

# Additional models for other endpoints...
```

#### static/index.html
```html
<!DOCTYPE html>
<html>
<head>
    <title>RAG Factory Dev Server</title>
    <style>
        /* Simple, clean CSS styling */
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .warning { background: #fff3cd; border: 2px solid #ff9800; padding: 15px; margin-bottom: 20px; }
        .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        textarea { width: 100%; min-height: 100px; }
        .results { background: #f8f9fa; padding: 15px; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="warning">
        <strong>⚠️ DEVELOPMENT SERVER - NOT FOR PRODUCTION USE</strong>
        <p>This server has no authentication, rate limiting, or security features.</p>
    </div>

    <div class="section">
        <h2>Query Documents</h2>
        <form id="queryForm">
            <label>Query:</label><br>
            <input type="text" id="query" style="width: 100%;" placeholder="Enter your question..."><br><br>

            <label>Strategies:</label><br>
            <div id="strategies"></div><br>

            <label>Top K:</label>
            <input type="number" id="topK" value="5" min="1" max="100"><br><br>

            <button type="submit">Submit Query</button>
        </form>
        <div id="queryResults" class="results" style="display:none;"></div>
    </div>

    <div class="section">
        <h2>Index Documents</h2>
        <form id="indexForm">
            <label>Path:</label>
            <input type="text" id="indexPath" style="width: 100%;" placeholder="./docs"><br><br>

            <label>Strategy:</label>
            <select id="indexStrategy"></select><br><br>

            <button type="submit">Start Indexing</button>
        </form>
        <div id="indexResults" class="results" style="display:none;"></div>
    </div>

    <div class="section">
        <h2>Request Logs</h2>
        <button onclick="loadLogs()">Refresh Logs</button>
        <div id="logs" class="results"></div>
    </div>

    <script>
        // Load strategies on page load
        async function loadStrategies() {
            const response = await fetch('/strategies');
            const data = await response.json();

            // Populate strategy checkboxes
            const strategiesDiv = document.getElementById('strategies');
            data.strategies.forEach(strategy => {
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = strategy.name;
                checkbox.value = strategy.name;

                const label = document.createElement('label');
                label.htmlFor = strategy.name;
                label.textContent = ` ${strategy.name} (${strategy.type})`;

                strategiesDiv.appendChild(checkbox);
                strategiesDiv.appendChild(label);
                strategiesDiv.appendChild(document.createElement('br'));
            });

            // Populate strategy dropdown for indexing
            const select = document.getElementById('indexStrategy');
            data.strategies.forEach(strategy => {
                const option = document.createElement('option');
                option.value = strategy.name;
                option.textContent = `${strategy.name} (${strategy.type})`;
                select.appendChild(option);
            });
        }

        // Handle query form submission
        document.getElementById('queryForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const query = document.getElementById('query').value;
            const checkboxes = document.querySelectorAll('#strategies input:checked');
            const strategies = Array.from(checkboxes).map(cb => cb.value);
            const topK = parseInt(document.getElementById('topK').value);

            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, strategies, top_k: topK })
            });

            const data = await response.json();
            const resultsDiv = document.getElementById('queryResults');
            resultsDiv.style.display = 'block';
            resultsDiv.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
        });

        // Handle index form submission
        document.getElementById('indexForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const path = document.getElementById('indexPath').value;
            const strategy = document.getElementById('indexStrategy').value;

            const response = await fetch('/index', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path, strategy })
            });

            const data = await response.json();
            const resultsDiv = document.getElementById('indexResults');
            resultsDiv.style.display = 'block';
            resultsDiv.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
        });

        // Load logs
        async function loadLogs() {
            const response = await fetch('/logs');
            const data = await response.json();
            const logsDiv = document.getElementById('logs');
            logsDiv.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
        }

        // Initialize on page load
        loadStrategies();
    </script>
</body>
</html>
```

---

## Testing Requirements

### Unit Tests

#### UT-1: Request Model Validation Tests
**Test File:** `tests/unit/server/test_models.py`

```python
from rag_factory_server.models import QueryRequest, IndexRequest

def test_query_request_validation():
    """Test QueryRequest model validates correctly"""
    # Given: Valid query data
    data = {
        "query": "test query",
        "strategies": ["reranking"],
        "top_k": 5
    }
    # When: Model is instantiated
    request = QueryRequest(**data)
    # Then: All fields are set correctly
    assert request.query == "test query"
    assert request.strategies == ["reranking"]
    assert request.top_k == 5

def test_query_request_defaults():
    """Test QueryRequest defaults are applied"""
    # Given: Minimal query data
    data = {"query": "test query"}
    # When: Model is instantiated
    request = QueryRequest(**data)
    # Then: Defaults are applied
    assert request.strategies == []
    assert request.top_k == 5

def test_query_request_top_k_validation():
    """Test top_k validation constraints"""
    # Given: Invalid top_k values
    # When: Model is instantiated with top_k < 1
    # Then: ValidationError is raised
    with pytest.raises(ValidationError):
        QueryRequest(query="test", top_k=0)

    # When: Model is instantiated with top_k > 100
    # Then: ValidationError is raised
    with pytest.raises(ValidationError):
        QueryRequest(query="test", top_k=101)

def test_index_request_validation():
    """Test IndexRequest requires either path or documents"""
    # Given: IndexRequest with neither path nor documents
    data = {"strategy": "semantic_chunking"}
    # When: Model is instantiated
    request = IndexRequest(**data)
    # Then: Both path and documents are None (validation in endpoint)
    assert request.path is None
    assert request.documents is None

def test_index_request_with_documents():
    """Test IndexRequest with inline documents"""
    # Given: Documents array
    documents = [
        {"text": "doc1", "metadata": {}},
        {"text": "doc2", "metadata": {}}
    ]
    data = {"documents": documents, "strategy": "chunking"}
    # When: Model is instantiated
    request = IndexRequest(**data)
    # Then: Documents are set
    assert len(request.documents) == 2
```

#### UT-2: Configuration Tests
**Test File:** `tests/unit/server/test_config.py`

```python
from rag_factory_server.config import Config

def test_config_loading():
    """Test configuration loading from file"""
    # Given: Config file path
    # When: Config is loaded
    # Then: All settings are correctly loaded
    pass

def test_config_hot_reload():
    """Test configuration hot-reloading"""
    # Given: Running config instance
    # When: Config file is modified and reload is called
    # Then: New values are loaded
    pass

def test_config_validation():
    """Test invalid configuration raises error"""
    # Given: Invalid config data
    # When: Config is validated
    # Then: ValidationError with specific issues
    pass
```

#### UT-3: Logging Utility Tests
**Test File:** `tests/unit/server/test_logging.py`

```python
from rag_factory_server.utils.logging import RequestLogger

def test_log_request():
    """Test request logging"""
    # Given: RequestLogger instance
    logger = RequestLogger(max_logs=100)
    # When: Request is logged
    logger.log_request("/query", {"query": "test"}, 200, 150)
    # Then: Log is stored
    assert len(logger.get_logs()) == 1

def test_log_rotation():
    """Test logs are rotated when max is reached"""
    # Given: Logger with max 10 logs
    logger = RequestLogger(max_logs=10)
    # When: 20 requests are logged
    for i in range(20):
        logger.log_request("/query", {}, 200, 100)
    # Then: Only last 10 are kept
    assert len(logger.get_logs()) == 10

def test_log_filtering():
    """Test log filtering by endpoint"""
    # Given: Logger with multiple endpoints logged
    logger = RequestLogger()
    logger.log_request("/query", {}, 200, 100)
    logger.log_request("/index", {}, 200, 200)
    # When: Logs are filtered by endpoint
    query_logs = logger.get_logs(endpoint="/query")
    # Then: Only matching logs returned
    assert len(query_logs) == 1
```

### Integration Tests

#### IT-1: Query Endpoint Integration
**Test File:** `tests/integration/server/test_query_endpoint.py`

```python
import pytest
from fastapi.testclient import TestClient
from rag_factory_server.main import app

client = TestClient(app)

def test_query_endpoint_success():
    """Test successful query request"""
    # Given: Indexed documents exist (setup in fixture)
    # When: POST request to /query
    response = client.post("/query", json={
        "query": "test query",
        "strategies": ["reranking"],
        "top_k": 5
    })
    # Then: Response is 200 with results
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "timing" in data
    assert data["query"] == "test query"

def test_query_endpoint_with_multiple_strategies():
    """Test query with multiple strategies"""
    # Given: Indexed documents
    # When: Query with multiple strategies
    response = client.post("/query", json={
        "query": "test",
        "strategies": ["reranking", "query_expansion"]
    })
    # Then: All strategies are applied
    assert response.status_code == 200
    data = response.json()
    assert len(data["strategies_used"]) == 2

def test_query_endpoint_missing_index():
    """Test query when no documents indexed"""
    # Given: Empty index (fresh server)
    # When: Query is sent
    response = client.post("/query", json={
        "query": "test"
    })
    # Then: 404 error with clear message
    assert response.status_code == 404
    assert "no indexed documents" in response.json()["detail"].lower()

def test_query_endpoint_invalid_strategy():
    """Test query with invalid strategy name"""
    # Given: Invalid strategy name
    # When: Query is sent
    response = client.post("/query", json={
        "query": "test",
        "strategies": ["invalid_strategy"]
    })
    # Then: 400 error with suggestions
    assert response.status_code == 400
    assert "invalid strategy" in response.json()["detail"].lower()

def test_query_endpoint_validation_error():
    """Test query with invalid payload"""
    # Given: Invalid request (missing query)
    # When: POST request sent
    response = client.post("/query", json={
        "strategies": ["reranking"]
    })
    # Then: 422 validation error
    assert response.status_code == 422
```

#### IT-2: Index Endpoint Integration
**Test File:** `tests/integration/server/test_index_endpoint.py`

```python
def test_index_endpoint_with_path(tmp_path):
    """Test indexing documents from file path"""
    # Given: Directory with test documents
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "test.txt").write_text("Test document content")

    # When: POST request to /index
    response = client.post("/index", json={
        "path": str(doc_dir),
        "strategy": "context_aware_chunking"
    })

    # Then: Documents are indexed successfully
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["documents_indexed"] > 0
    assert "time_ms" in data

def test_index_endpoint_with_documents():
    """Test indexing inline documents"""
    # Given: Document array
    documents = [
        {"text": "Document 1 content", "metadata": {"source": "test"}},
        {"text": "Document 2 content", "metadata": {"source": "test"}}
    ]

    # When: POST request with documents
    response = client.post("/index", json={
        "documents": documents,
        "strategy": "semantic_chunking"
    })

    # Then: Documents are indexed
    assert response.status_code == 200
    data = response.json()
    assert data["documents_indexed"] == 2

def test_index_endpoint_invalid_path():
    """Test indexing with non-existent path"""
    # Given: Invalid path
    # When: POST request sent
    response = client.post("/index", json={
        "path": "/non/existent/path",
        "strategy": "chunking"
    })
    # Then: 400 error
    assert response.status_code == 400

def test_index_endpoint_missing_path_and_documents():
    """Test indexing without path or documents"""
    # Given: Neither path nor documents provided
    # When: POST request sent
    response = client.post("/index", json={
        "strategy": "chunking"
    })
    # Then: 400 error with clear message
    assert response.status_code == 400
    assert "path or documents" in response.json()["detail"].lower()
```

#### IT-3: Strategies Endpoint Integration
**Test File:** `tests/integration/server/test_strategies_endpoint.py`

```python
def test_strategies_endpoint():
    """Test listing all strategies"""
    # Given: Server with strategies loaded
    # When: GET request to /strategies
    response = client.get("/strategies")
    # Then: All strategies are returned
    assert response.status_code == 200
    data = response.json()
    assert "strategies" in data
    assert len(data["strategies"]) > 0

    # Verify strategy structure
    strategy = data["strategies"][0]
    assert "name" in strategy
    assert "type" in strategy
    assert "description" in strategy

def test_strategies_endpoint_with_filter():
    """Test filtering strategies by type"""
    # Given: Strategies of different types
    # When: GET request with type filter
    response = client.get("/strategies?type=retrieval")
    # Then: Only retrieval strategies returned
    assert response.status_code == 200
    data = response.json()
    for strategy in data["strategies"]:
        assert strategy["type"] == "retrieval"
```

#### IT-4: Health Check Integration
**Test File:** `tests/integration/server/test_health_endpoint.py`

```python
def test_health_endpoint():
    """Test health check endpoint"""
    # Given: Running server
    # When: GET request to /health
    response = client.get("/health")
    # Then: Health status returned
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert "version" in data

def test_health_endpoint_performance():
    """Test health check responds quickly"""
    # Given: Running server
    # When: GET request to /health
    import time
    start = time.time()
    response = client.get("/health")
    duration = time.time() - start
    # Then: Response time < 100ms
    assert response.status_code == 200
    assert duration < 0.1
```

#### IT-5: Configuration Reload Integration
**Test File:** `tests/integration/server/test_config_reload.py`

```python
def test_config_reload_endpoint(tmp_path):
    """Test configuration hot-reload"""
    # Given: Running server with config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text("setting: value1")

    # When: Config file is modified and reload called
    config_file.write_text("setting: value2")
    response = client.post("/config/reload", json={
        "config_path": str(config_file)
    })

    # Then: New config is loaded
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

def test_config_reload_invalid_file():
    """Test reload with invalid config file"""
    # Given: Invalid config path
    # When: Reload is called
    response = client.post("/config/reload", json={
        "config_path": "/invalid/config.yaml"
    })
    # Then: Error is returned
    assert response.status_code == 400
```

#### IT-6: Logging Endpoint Integration
**Test File:** `tests/integration/server/test_logs_endpoint.py`

```python
def test_logs_endpoint():
    """Test retrieving request logs"""
    # Given: Server has processed some requests
    client.get("/health")
    client.get("/strategies")

    # When: GET request to /logs
    response = client.get("/logs")

    # Then: Recent logs are returned
    assert response.status_code == 200
    data = response.json()
    assert "logs" in data
    assert len(data["logs"]) >= 2

def test_logs_endpoint_with_limit():
    """Test logs with limit parameter"""
    # Given: Multiple requests logged
    for _ in range(20):
        client.get("/health")

    # When: Logs requested with limit
    response = client.get("/logs?limit=10")

    # Then: Only 10 logs returned
    assert response.status_code == 200
    data = response.json()
    assert len(data["logs"]) == 10
```

#### IT-7: HTML UI Integration
**Test File:** `tests/integration/server/test_ui.py`

```python
def test_root_serves_html():
    """Test root endpoint serves HTML UI"""
    # Given: Server running
    # When: GET request to root
    response = client.get("/")
    # Then: HTML is returned
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "RAG Factory Dev Server" in response.text

def test_ui_contains_warning():
    """Test UI displays development-only warning"""
    # Given: Server running
    # When: UI is loaded
    response = client.get("/")
    # Then: Warning is visible
    assert "DEVELOPMENT SERVER" in response.text
    assert "NOT FOR PRODUCTION" in response.text
```

#### IT-8: CORS and OpenAPI Integration
**Test File:** `tests/integration/server/test_openapi.py`

```python
def test_openapi_json():
    """Test OpenAPI spec is available"""
    # Given: Server running
    # When: GET /openapi.json
    response = client.get("/openapi.json")
    # Then: Valid OpenAPI spec returned
    assert response.status_code == 200
    spec = response.json()
    assert spec["openapi"]
    assert "paths" in spec
    assert "/query" in spec["paths"]

def test_swagger_ui():
    """Test Swagger UI is accessible"""
    # Given: Server running
    # When: GET /docs
    response = client.get("/docs")
    # Then: Swagger UI HTML is returned
    assert response.status_code == 200
    assert "swagger" in response.text.lower()

def test_cors_headers():
    """Test CORS headers are set"""
    # Given: Server with CORS enabled
    # When: Request with Origin header
    response = client.get("/health", headers={
        "Origin": "http://localhost:3000"
    })
    # Then: CORS headers present
    assert "access-control-allow-origin" in response.headers
```

#### IT-9: Error Handling Integration
**Test File:** `tests/integration/server/test_error_handling.py`

```python
def test_404_error():
    """Test 404 for non-existent endpoint"""
    # Given: Invalid endpoint
    # When: GET request
    response = client.get("/nonexistent")
    # Then: 404 error
    assert response.status_code == 404

def test_405_method_not_allowed():
    """Test 405 for wrong HTTP method"""
    # Given: Endpoint that only accepts POST
    # When: GET request is sent
    response = client.get("/query")
    # Then: 405 error
    assert response.status_code == 405

def test_500_internal_error_handling():
    """Test 500 errors are handled gracefully"""
    # Given: Scenario that causes internal error (mock library error)
    # When: Request is processed
    # Then: 500 error with message, no stack trace
    # (Requires mocking library to raise exception)
    pass
```

#### IT-10: Full Workflow Integration
**Test File:** `tests/integration/server/test_full_workflow.py`

```python
def test_complete_workflow(tmp_path):
    """Test complete index and query workflow"""
    # Given: Test documents
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "doc1.txt").write_text("Python is a programming language")
    (doc_dir / "doc2.txt").write_text("FastAPI is a web framework")

    # When: Documents are indexed
    index_response = client.post("/index", json={
        "path": str(doc_dir),
        "strategy": "context_aware_chunking"
    })
    assert index_response.status_code == 200

    # And: Query is executed
    query_response = client.post("/query", json={
        "query": "What is Python?",
        "strategies": ["reranking"]
    })

    # Then: Relevant results are returned
    assert query_response.status_code == 200
    data = query_response.json()
    assert len(data["results"]) > 0
    assert "Python" in data["results"][0]["text"]
```

---

## Code Quality Requirements

### CQ-1: Code Style
- Follow PEP 8 style guide
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use Black for code formatting
- Use isort for import sorting

### CQ-2: Documentation
- Docstrings for all endpoints (Google style)
- OpenAPI descriptions for all endpoints
- README with installation and usage
- API documentation via Swagger/ReDoc
- HTML UI has inline help text

### CQ-3: Test Coverage
- Minimum 80% code coverage
- All endpoints must have integration tests
- All models must have validation tests
- Error scenarios must be tested

### CQ-4: Error Handling
- FastAPI exception handlers for all error types
- User-friendly error messages
- Appropriate HTTP status codes
- Logging of all errors

### CQ-5: Type Safety
- All functions have type hints
- Pydantic models for all requests/responses
- Mypy type checking passes with no errors

### CQ-6: Performance
- Async endpoints for I/O operations
- Connection pooling if needed
- Efficient request/response serialization
- Response time monitoring

### CQ-7: Security Awareness
- Document all security limitations
- No sensitive data in logs
- Input validation on all endpoints
- Warning banner in UI and docs

---

## Definition of Done

- [ ] All functional requirements implemented
- [ ] All acceptance criteria met
- [ ] All unit tests written and passing
- [ ] All integration tests written and passing
- [ ] Code coverage ≥ 80%
- [ ] Type checking (mypy) passes
- [ ] Code style checks (Black, isort, flake8) pass
- [ ] OpenAPI documentation complete and accurate
- [ ] HTML UI functional and tested manually
- [ ] README with installation and usage examples
- [ ] Development-only warnings prominent in all docs and UI
- [ ] Tested on Linux, macOS, and Windows
- [ ] Server can be started with `rag-factory serve` command
- [ ] Integration with library's public API verified
- [ ] Peer review completed
- [ ] Manual testing of all features completed

---

## Testing Checklist

### Before Development
- [ ] Review Epic 4 strategies to understand API
- [ ] Set up FastAPI + Uvicorn environment
- [ ] Design API endpoint structure
- [ ] Create Pydantic models for all requests/responses
- [ ] Write OpenAPI descriptions

### During Development
- [ ] Write unit test before implementing each endpoint
- [ ] Test each endpoint with Postman/curl
- [ ] Verify OpenAPI docs are accurate
- [ ] Test HTML UI in browser
- [ ] Verify error handling for edge cases

### After Implementation
- [ ] Run full test suite and achieve 80%+ coverage
- [ ] Test on different operating systems
- [ ] Test with different browsers (Chrome, Firefox, Safari)
- [ ] Load test with multiple concurrent requests
- [ ] Verify CORS works with frontend applications
- [ ] Test configuration hot-reload
- [ ] Verify all warnings are displayed
- [ ] Performance test with large documents

---

## Example Usage Scenarios

### Scenario 1: Start Server and Query
```bash
# Start dev server
rag-factory serve --config dev_config.yaml --port 8000

# Index documents (from another terminal)
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"path": "./docs", "strategy": "context_aware_chunking"}'

# Query documents
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the action items?", "strategies": ["reranking", "query_expansion"]}'
```

### Scenario 2: Use HTML UI
```bash
# Start server
rag-factory serve

# Open browser to http://localhost:8000
# Use web UI to:
# 1. Index documents
# 2. Submit queries
# 3. View results
# 4. Check request logs
```

### Scenario 3: Development with Hot-Reload
```bash
# Start server with config
rag-factory serve --config config.yaml

# Modify config.yaml
# Reload without restart
curl -X POST http://localhost:8000/config/reload
```

### Scenario 4: API Exploration
```bash
# Start server
rag-factory serve

# Open Swagger UI
# Navigate to http://localhost:8000/docs

# Explore and test API interactively
```

---

## Risk Management

### Risk 1: FastAPI Learning Curve
- **Risk:** Team unfamiliar with FastAPI
- **Mitigation:** FastAPI has excellent documentation, start with simple endpoints
- **Contingency:** Use Flask if FastAPI proves too complex

### Risk 2: CORS Issues
- **Risk:** Frontend integration blocked by CORS
- **Mitigation:** Enable CORS middleware from start, test with frontend early
- **Contingency:** Document CORS configuration clearly

### Risk 3: Performance with Large Documents
- **Risk:** Server may be slow with large document sets
- **Mitigation:** Use async I/O, implement streaming for large responses
- **Contingency:** Document performance limitations clearly

### Risk 4: HTML/JS Complexity
- **Risk:** Single-page UI may become complex
- **Mitigation:** Keep UI simple, focus on functionality over aesthetics
- **Contingency:** Use simple templates, avoid complex JavaScript

---

## Success Metrics

- Server starts and serves all endpoints successfully
- HTML UI is functional and user-friendly
- All unit tests pass with ≥80% coverage
- All integration tests pass
- OpenAPI documentation is complete and accurate
- Developer feedback positive (manual testing)
- Server validates library's public API design
- Clear warnings about development-only use
- Works on all target platforms
- Performance acceptable for development/POC use

---

## Future Considerations (Out of Scope)

- Authentication/authorization
- Rate limiting
- Horizontal scaling
- Database persistence
- Production deployment options
- Monitoring and alerting
- SSL/TLS support
- Containerization (Docker)
- Cloud deployment guides

**Note:** These are explicitly out of scope for this development server. If production deployment is needed, a separate production-ready implementation should be created.
