# RAG Factory System - Epics Overview

This directory contains detailed documentation for each epic in the RAG Factory System project.

---

## Project Overview

Build a modular RAG (Retrieval Augmented Generation) factory system with a unified interface that supports multiple RAG strategies. Each strategy will be implemented as a separate class that adheres to a common interface, allowing for flexible composition and selection of RAG approaches based on use case requirements.

**This is a Python library/module designed to be imported and used programmatically, NOT a standalone application.**

---

## Quick Navigation

### Foundation Epics (Required for MVP)

1. **[Epic 1: Core Infrastructure & Factory Pattern](./epic-01-core-infrastructure.md)** (31 points)
   - Design RAG Strategy Interface
   - Implement RAG Factory
   - Build Strategy Composition Engine
   - Create Configuration Management System
   - Setup Package Structure & Distribution

2. **[Epic 2: Database & Storage Infrastructure](./epic-02-database-storage.md)** (13 points)
   - Set Up Vector Database with PG Vector
   - Implement Database Repository Pattern

3. **[Epic 3: Core Services Layer](./epic-03-core-services.md)** (16 points)
   - Build Embedding Service
   - Implement LLM Service Adapter

---

### Strategy Epics (Core Functionality)

4. **[Epic 4: Priority RAG Strategies](./epic-04-priority-strategies.md)** (34 points) ⭐ **MVP**
   - Implement Context-Aware Chunking Strategy
   - Implement Re-ranking Strategy
   - Implement Query Expansion Strategy

5. **[Epic 5: Agentic & Advanced Retrieval Strategies](./epic-05-agentic-advanced.md)** (34 points)
   - Implement Agentic RAG Strategy
   - Implement Hierarchical RAG Strategy
   - Implement Self-Reflective RAG Strategy

6. **[Epic 6: Multi-Query & Contextual Strategies](./epic-06-multi-query-contextual.md)** (26 points)
   - Implement Multi-Query RAG Strategy
   - Implement Contextual Retrieval Strategy

7. **[Epic 7: Advanced & Experimental Strategies](./epic-07-experimental-strategies.md)** (63 points) ⚠️ **High Complexity**
   - Implement Knowledge Graph Strategy
   - Implement Late Chunking Strategy
   - Implement Fine-Tuned Embeddings Strategy

---

### Quality & Developer Experience Epics

8. **[Epic 8: Observability & Quality Assurance](./epic-08-observability.md)** (21 points)
   - Build Monitoring & Logging System
   - Create Evaluation Framework

8.5. **[Epic 8.5: Development Tools](./epic-08.5-development-tools.md)** (16 points)
   - Build CLI for Strategy Testing
   - Create Lightweight Dev Server for POCs

9. **[Epic 9: Documentation & Developer Experience](./epic-09-documentation.md)** (26 points)
   - Write Developer Documentation
   - Create Example Implementations

---

## Epic Breakdown by Story Points

| Epic | Name | Points | Sprint(s) |
|------|------|--------|-----------|
| Epic 1 | Core Infrastructure & Factory Pattern | 31 | Sprint 1 |
| Epic 2 | Database & Storage Infrastructure | 13 | Sprint 1 |
| Epic 3 | Core Services Layer | 16 | Sprint 2 |
| Epic 4 | Priority RAG Strategies | 34 | Sprint 3-4 |
| Epic 5 | Agentic & Advanced Retrieval | 34 | Sprint 5-6 |
| Epic 6 | Multi-Query & Contextual | 26 | Sprint 7 |
| Epic 7 | Advanced & Experimental | 63 | Sprint 9-10 |
| Epic 8 | Observability & Quality | 21 | Sprint 4, 8 |
| Epic 8.5 | Development Tools | 16 | Sprint 8 |
| Epic 9 | Documentation & Dev Experience | 26 | Sprint 11 |
| **TOTAL** | | **280** | **11 Sprints** |

---

## Recommended Implementation Path

### Phase 1: MVP (Foundation + High-Impact Trio)
**Goal:** Importable library with the three recommended strategies

1. ✅ **Epic 1** - Core Infrastructure (Sprint 1)
2. ✅ **Epic 2** - Database & Storage (Sprint 1)
3. ✅ **Epic 3** - Core Services (Sprint 2)
4. ✅ **Epic 4** - Priority Strategies (Sprint 3-4) ⭐
5. ✅ **Epic 8 (Story 8.1)** - Basic Logging (Sprint 4)
6. ✅ **Epic 8.5 (Story 8.5.1)** - CLI Tool (Sprint 8)
7. ✅ **Epic 9 (Story 9.2)** - Simple Example (Sprint 11)

**Deliverable:** Working library with context-aware chunking, re-ranking, and query expansion.

---

### Phase 2: Advanced Strategies (Sprint 5-7)

8. **Epic 5** - Agentic & Advanced Retrieval
9. **Epic 6** - Multi-Query & Contextual

**Deliverable:** 5 additional strategies for specialized use cases.

---

### Phase 3: Quality & Polish (Sprint 8-11)

10. **Epic 8** - Full Observability
11. **Epic 8.5** - Development Tools
12. **Epic 9** - Complete Documentation

**Deliverable:** Production-ready library with full documentation.

---

### Phase 4: Experimental (Sprint 9-10)

13. **Epic 7** - Advanced & Experimental Strategies

**Deliverable:** Optional experimental strategies for cutting-edge use cases.

---

## Dependencies Graph

```
Epic 1 (Core Infrastructure)
  └─→ Epic 2 (Database)
       └─→ Epic 3 (Services)
            ├─→ Epic 4 (Priority Strategies) ⭐ MVP
            │    ├─→ Epic 5 (Agentic & Advanced)
            │    │    └─→ Epic 7 (Experimental)
            │    ├─→ Epic 6 (Multi-Query & Contextual)
            │    ├─→ Epic 8 (Observability)
            │    └─→ Epic 8.5 (Dev Tools)
            └─→ Epic 9 (Documentation - requires all epics)
```

---

## Success Metrics

### Performance Metrics
- Retrieval accuracy > 85% (measured via evaluation framework)
- Query latency < 2 seconds (p95)
- Cost per query < $0.02

### Code Quality Metrics
- Test coverage > 80%
- Type coverage > 90%
- Zero critical security vulnerabilities

### Developer Experience
- Documentation completeness score > 90%
- Time to first working example < 30 minutes
- Strategy addition time < 4 hours

---

## Risk Management

### High Risks
1. **Late Chunking Complexity** (Epic 7) - Extremely complex, may not be worth implementing
   - Mitigation: Mark as optional/experimental, implement last

2. **Knowledge Graph Performance** (Epic 7) - Can be slow with large datasets
   - Mitigation: Implement caching, limit graph depth, use indexes

3. **Cost Overruns** - Multiple LLM calls can get expensive
   - Mitigation: Implement caching, batch operations, cost tracking

### Medium Risks
1. **Integration Complexity** - Many external services to integrate
   - Mitigation: Use adapter pattern, mock extensively in tests

2. **Configuration Explosion** - Too many knobs to tune
   - Mitigation: Provide sensible defaults, configuration presets

---

## Technical Stack Summary

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

---

## Getting Started

1. Start with [Epic 1: Core Infrastructure](./epic-01-core-infrastructure.md)
2. Review the recommended implementation path above
3. Each epic file contains detailed acceptance criteria and technical notes
4. Story points estimates use Fibonacci sequence (1, 2, 3, 5, 8, 13, 21)

---

## Notes

- Each sprint assumes ~25-30 points capacity
- Stories can be broken down further during sprint planning
- Technical spikes may be needed for Epic 7 stories
- Consider pairing on complex stories (13+ points)
- Epic 7 is optional and can be deferred
