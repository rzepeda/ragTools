# Epic 5: Agentic & Advanced Retrieval Strategies

**Epic Goal:** Implement advanced strategies that add intelligence and flexibility to retrieval: Agentic RAG, Self-Reflective RAG, and Hierarchical RAG.

**Epic Story Points Total:** 34

**Dependencies:** Epic 4 (basic strategies should be working first)

---

## Story 5.1: Implement Agentic RAG Strategy

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

## Story 5.2: Implement Hierarchical RAG Strategy

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

## Story 5.3: Implement Self-Reflective RAG Strategy

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

## Sprint Planning

**Sprint 5:** Stories 5.1, 5.3 (21 points)
**Sprint 6:** Story 5.2 (13 points)

---

## Technical Stack

**Agentic:**
- LangGraph or similar agent framework
- Anthropic Computer Use
- Custom tool definitions

**Hierarchical:**
- Database schema extensions
- Metadata management

**Self-Reflective:**
- LLM service for grading
- Retry logic framework

---

## Success Criteria

- [ ] Agentic RAG can select appropriate tools based on query
- [ ] Hierarchical RAG retrieves correct parent chunks
- [ ] Self-reflective RAG improves results through iteration
- [ ] All strategies integrate with pipeline
- [ ] Performance metrics tracked
- [ ] Integration tests passing
