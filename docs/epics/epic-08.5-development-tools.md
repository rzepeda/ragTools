# Epic 8.5: Development Tools (CLI & Dev Server)

**Epic Goal:** Create lightweight development tools for testing, debugging, and POC demonstrations. These are NOT production-ready clients but development aids.

**Epic Story Points Total:** 16

**Dependencies:** Epic 4 (need working strategies to test)

---

## Story 8.5.1: Build CLI for Strategy Testing

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

## Story 8.5.2: Create Lightweight Dev Server for POCs

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

## Sprint Planning

**Sprint 8:** All stories (8.5.1 - 8.5.2) = 16 points + Epic 8 Story 8.2

---

## Important Notes

**These are NOT production tools:**
- No authentication or authorization
- No rate limiting
- No horizontal scaling
- Minimal error handling
- Development/testing only

**These ARE useful for:**
- Quick experimentation
- Testing strategies
- POC demonstrations
- Validating the library's public API
- Educational purposes

---

## Technical Stack

**CLI:**
- Click or Typer (command-line framework)
- Rich (colorized output)
- tqdm (progress bars)

**Dev Server:**
- FastAPI (HTTP framework)
- Uvicorn (ASGI server)
- Simple HTML/JavaScript UI

---

## Success Criteria

- [ ] CLI can index documents using any strategy
- [ ] CLI can query with multiple strategies
- [ ] CLI validates configuration files
- [ ] CLI has interactive REPL mode
- [ ] Dev server serves HTTP endpoints
- [ ] Dev server has simple HTML UI
- [ ] Both tools use library as a client would
- [ ] Documentation clearly states "development only"
- [ ] Examples provided for common use cases
