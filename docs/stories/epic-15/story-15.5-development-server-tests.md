# Story 15.5: Add Development Server Tests

**Story ID:** 15.5  
**Epic:** Epic 15 - Test Coverage Improvements  
**Story Points:** 5  
**Priority:** Medium  
**Dependencies:** Epic 8.5 (Development Server)

---

## User Story

**As a** developer  
**I want** comprehensive tests for the development server  
**So that** I can ensure the HTTP API works correctly for demos and POCs

---

## Detailed Requirements

### Functional Requirements

> [!WARNING]
> **Current Development Server Coverage**: 0% (No tests found)
> **Target Coverage**: 60%+
> **Missing Features**: All server features lack tests - HTTP endpoints, health checks, API docs, server lifecycle
> **Note**: This is a completely new test area

1. **HTTP Endpoint Tests**
   - Test POST /query endpoint
   - Test POST /index endpoint
   - Test GET /strategies endpoint
   - Test request/response formats
   - Test error handling

2. **Health Check Tests**
   - Test GET /health endpoint
   - Test health status reporting
   - Test service availability checks

3. **API Documentation Tests**
   - Test Swagger/OpenAPI endpoint
   - Test documentation completeness
   - Test example requests/responses

4. **Server Lifecycle Tests**
   - Test server startup
   - Test graceful shutdown
   - Test hot-reload configuration

---

## Acceptance Criteria

### AC1: HTTP Endpoint Tests
- [ ] Test file `tests/unit/server/test_endpoints.py` created
- [ ] Test POST /query with valid request
- [ ] Test POST /query with invalid request
- [ ] Test POST /index with documents
- [ ] Test POST /index with invalid data
- [ ] Test GET /strategies returns list
- [ ] Test error responses (400, 404, 500)
- [ ] Test response format (JSON)
- [ ] Minimum 12 test cases

### AC2: Health Check Tests
- [ ] Test file `tests/unit/server/test_health.py` created
- [ ] Test GET /health returns 200
- [ ] Test health status includes service checks
- [ ] Test health check with unavailable services
- [ ] Test health check response format
- [ ] Minimum 6 test cases

### AC3: API Documentation Tests
- [ ] Test file `tests/unit/server/test_api_docs.py` created
- [ ] Test GET /docs returns Swagger UI
- [ ] Test OpenAPI spec generation
- [ ] Test all endpoints documented
- [ ] Test example requests in docs
- [ ] Minimum 5 test cases

### AC4: Server Lifecycle Tests
- [ ] Test file `tests/unit/server/test_server_lifecycle.py` created
- [ ] Test server starts successfully
- [ ] Test server binds to correct port
- [ ] Test graceful shutdown
- [ ] Test hot-reload on config change
- [ ] Minimum 6 test cases

### AC5: Integration Tests
- [ ] Test file `tests/integration/server/test_server_integration.py` created
- [ ] Test full query workflow via HTTP
- [ ] Test full indexing workflow via HTTP
- [ ] Test concurrent requests
- [ ] Test server with real factory
- [ ] Minimum 8 test cases

### AC6: Test Quality
- [ ] All tests pass (100% success rate)
- [ ] Development server coverage reaches 60%+ (from 0%)
- [ ] Tests use FastAPI TestClient
- [ ] Proper cleanup after each test
- [ ] Type hints validated
- [ ] Linting passes

---

## Technical Specifications

### File Structure

```
tests/unit/server/
├── __init__.py                 # NEW
├── test_endpoints.py           # NEW
├── test_health.py              # NEW
├── test_api_docs.py            # NEW
└── test_server_lifecycle.py   # NEW

tests/integration/server/
├── __init__.py                 # NEW
└── test_server_integration.py # NEW
```

### Endpoint Test Template

```python
"""Unit tests for server HTTP endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from rag_factory.server.app import create_app

class TestServerEndpoints:
    """Test suite for HTTP API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    def test_query_endpoint_success(self, client):
        """Test POST /query with valid request."""
        with patch('rag_factory.server.app.RAGFactory') as mock_factory:
            mock_factory.return_value.query.return_value = {
                "answer": "Test answer",
                "chunks": []
            }
            
            response = client.post("/query", json={
                "query": "What is RAG?",
                "strategy": "reranking",
                "top_k": 5
            })
            
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert data["answer"] == "Test answer"
    
    def test_query_endpoint_invalid_request(self, client):
        """Test POST /query with missing required fields."""
        response = client.post("/query", json={
            "strategy": "reranking"
            # Missing 'query' field
        })
        
        assert response.status_code == 400
        assert "error" in response.json()
    
    def test_index_endpoint_success(self, client):
        """Test POST /index with valid documents."""
        with patch('rag_factory.server.app.RAGFactory') as mock_factory:
            mock_factory.return_value.index.return_value = {
                "indexed": 5,
                "chunks": 25
            }
            
            response = client.post("/index", json={
                "documents": [
                    {"text": "Document 1"},
                    {"text": "Document 2"}
                ],
                "strategy": "context_aware_chunking"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert "indexed" in data
    
    def test_list_strategies_endpoint(self, client):
        """Test GET /strategies returns available strategies."""
        response = client.get("/strategies")
        
        assert response.status_code == 200
        data = response.json()
        assert "indexing" in data
        assert "retrieval" in data
        assert isinstance(data["indexing"], list)
        assert isinstance(data["retrieval"], list)
```

### Health Check Test Template

```python
"""Unit tests for health check endpoint."""
import pytest
from fastapi.testclient import TestClient
from rag_factory.server.app import create_app

class TestHealthEndpoint:
    """Test suite for health check endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    def test_health_check_success(self, client):
        """Test GET /health returns healthy status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
    
    def test_health_check_includes_service_status(self, client):
        """Test health check includes individual service statuses."""
        response = client.get("/health")
        
        data = response.json()
        services = data["services"]
        
        # Should check database, embedding, llm services
        assert "database" in services
        assert "embedding" in services
        assert "llm" in services
```

### Server Lifecycle Test Template

```python
"""Unit tests for server lifecycle."""
import pytest
import asyncio
from rag_factory.server.app import create_app, start_server, stop_server

class TestServerLifecycle:
    """Test suite for server startup and shutdown."""
    
    @pytest.mark.asyncio
    async def test_server_starts_successfully(self):
        """Test that server starts without errors."""
        app = create_app()
        
        # Start server in background
        server_task = asyncio.create_task(
            start_server(app, host="127.0.0.1", port=8000)
        )
        
        # Give it time to start
        await asyncio.sleep(0.5)
        
        # Server should be running
        assert not server_task.done()
        
        # Cleanup
        server_task.cancel()
    
    def test_graceful_shutdown(self):
        """Test server shuts down gracefully."""
        app = create_app()
        
        # Start and stop
        stop_server(app)
        
        # Should complete without errors
        assert True
```

### Testing Strategy

1. **Unit Tests**
   - Use FastAPI TestClient
   - Mock RAGFactory and services
   - Test endpoint logic in isolation
   - Fast execution

2. **Integration Tests**
   - Start real server
   - Make HTTP requests
   - Test with real factory (mocked services)
   - Test concurrent requests

3. **API Contract Tests**
   - Validate request/response schemas
   - Test OpenAPI spec accuracy
   - Verify error response formats

---

## Definition of Done

- [ ] All 4 new unit test files created
- [ ] All 1 new integration test file created
- [ ] All tests pass (100% success rate)
- [ ] Development server coverage reaches 60%+ (from 0%)
- [ ] Tests use FastAPI TestClient
- [ ] Type checking passes
- [ ] Linting passes
- [ ] PR merged

---

## Notes

- Current development server has 0% test coverage
- Use FastAPI TestClient for all HTTP endpoint tests
- Server should be tested without actually starting on a port (use TestClient)
- Integration tests can start actual server on random port
