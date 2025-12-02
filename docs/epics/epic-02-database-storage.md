# Epic 2: Database & Storage Infrastructure

**Epic Goal:** Set up the database layer with PostgreSQL + pgvector for vector storage and implement repository patterns for data access.

**Epic Story Points Total:** 13

**Dependencies:** Epic 1 (Story 1.1 for interface definitions)

---

## Story 2.1: Set Up Vector Database with PG Vector

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

## Story 2.2: Implement Database Repository Pattern

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

## Sprint Planning

This epic is recommended for **Sprint 1** along with Epic 1.

**Total Sprint 1:** 44 points (Epic 1 + Epic 2)

---

## Technical Stack

**Database:**
- PostgreSQL 15+ with pgvector extension
- Recommended: Neon (managed PostgreSQL)

**Python Libraries:**
- psycopg2 or asyncpg (database connection)
- alembic (migrations)

---

## Success Criteria

- [ ] PostgreSQL with pgvector is running (local or Neon)
- [ ] Chunks table created with vector column
- [ ] Documents table created
- [ ] ChunkRepository can perform CRUD operations
- [ ] DocumentRepository can perform CRUD operations
- [ ] All repository tests passing
- [ ] Connection pooling configured
