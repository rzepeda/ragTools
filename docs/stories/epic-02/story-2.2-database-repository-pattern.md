# Story 2.2: Implement Database Repository Pattern

**Story ID:** 2.2
**Epic:** Epic 2 - Database & Storage Infrastructure
**Story Points:** 8
**Priority:** High
**Dependencies:** Story 2.1 (database setup), Story 1.1 (interface definitions)

---

## User Story

**As a** developer
**I want** repository classes for database operations
**So that** database logic is abstracted from strategies

---

## Detailed Requirements

### Functional Requirements

1. **Base Repository Interface**
   - Abstract base repository defining common CRUD operations
   - Generic type support for different entity types
   - Consistent error handling across all repositories
   - Transaction context management

2. **DocumentRepository**
   - **Create Operations**
     - `create(document: DocumentCreate) -> Document`: Create single document
     - `bulk_create(documents: List[DocumentCreate]) -> List[Document]`: Bulk insert

   - **Read Operations**
     - `get_by_id(document_id: UUID) -> Optional[Document]`: Get by ID
     - `get_by_content_hash(content_hash: str) -> Optional[Document]`: Get by hash for deduplication
     - `list_all(skip: int, limit: int) -> List[Document]`: Paginated listing
     - `count() -> int`: Total count
     - `get_by_status(status: str) -> List[Document]`: Filter by processing status

   - **Update Operations**
     - `update(document_id: UUID, updates: DocumentUpdate) -> Document`: Update document
     - `update_status(document_id: UUID, status: str) -> Document`: Update status

   - **Delete Operations**
     - `delete(document_id: UUID) -> bool`: Delete document (cascades to chunks)
     - `bulk_delete(document_ids: List[UUID]) -> int`: Delete multiple

3. **ChunkRepository**
   - **Create Operations**
     - `create(chunk: ChunkCreate) -> Chunk`: Create single chunk
     - `bulk_create(chunks: List[ChunkCreate]) -> List[Chunk]`: Bulk insert for performance

   - **Read Operations**
     - `get_by_id(chunk_id: UUID) -> Optional[Chunk]`: Get by ID
     - `get_by_document(document_id: UUID, skip: int, limit: int) -> List[Chunk]`: Get chunks for document
     - `count_by_document(document_id: UUID) -> int`: Count chunks per document

   - **Vector Search Operations**
     - `search_similar(embedding: List[float], top_k: int, threshold: float) -> List[Chunk]`: Semantic search
     - `search_similar_with_filter(embedding: List[float], top_k: int, document_ids: List[UUID]) -> List[Chunk]`: Filtered search
     - `search_similar_with_metadata(embedding: List[float], top_k: int, metadata_filter: Dict) -> List[Chunk]`: Metadata-filtered search

   - **Update Operations**
     - `update_embedding(chunk_id: UUID, embedding: List[float]) -> Chunk`: Add/update embedding
     - `bulk_update_embeddings(updates: List[Tuple[UUID, List[float]]]) -> int`: Batch embedding updates

   - **Delete Operations**
     - `delete(chunk_id: UUID) -> bool`: Delete single chunk
     - `delete_by_document(document_id: UUID) -> int`: Delete all chunks for document

4. **Transaction Management**
   - Support for atomic operations across multiple repository calls
   - Context manager for transaction boundaries
   - Rollback on errors
   - Nested transaction support

5. **Error Handling**
   - Custom exception classes for repository errors
   - `RepositoryError`: Base exception
   - `EntityNotFoundError`: When entity doesn't exist
   - `DuplicateEntityError`: When unique constraint violated
   - `DatabaseConnectionError`: When database unavailable
   - Proper error messages with context

6. **Performance Optimizations**
   - Batch operations for bulk inserts/updates
   - Lazy loading for relationships
   - Query result caching (optional)
   - Connection reuse from pool

### Non-Functional Requirements

1. **Performance**
   - Single document operations <10ms
   - Bulk operations handle 1000+ entities efficiently
   - Vector search with filters <100ms
   - Minimal query overhead from abstraction

2. **Testability**
   - All repository methods unit testable
   - Support for test fixtures and mocking
   - In-memory database support for tests
   - Isolated test transactions

3. **Maintainability**
   - Clear separation of concerns
   - Consistent naming conventions
   - Comprehensive documentation
   - Type hints on all methods

4. **Reliability**
   - Automatic retry on transient failures
   - Connection timeout handling
   - Transaction rollback on errors
   - Graceful degradation

5. **Extensibility**
   - Easy to add new repository methods
   - Support for custom query builders
   - Pluggable caching strategies
   - Support for different database backends (future)

---

## Acceptance Criteria

### AC1: Base Repository Structure
- [ ] `BaseRepository` abstract class defined
- [ ] Common CRUD methods defined in base class
- [ ] Generic type support for entities
- [ ] Session management integrated

### AC2: DocumentRepository Implementation
- [ ] All CRUD operations implemented
- [ ] Deduplication via content hash works
- [ ] Status filtering works correctly
- [ ] Pagination works with skip/limit
- [ ] Bulk operations handle large datasets

### AC3: ChunkRepository Implementation
- [ ] All CRUD operations implemented
- [ ] Vector similarity search works
- [ ] Filtered vector search works (by document_id)
- [ ] Metadata-based filtering works
- [ ] Bulk embedding updates efficient

### AC4: Vector Search Accuracy
- [ ] Cosine similarity returns correct results
- [ ] Top-k results ordered by similarity
- [ ] Similarity threshold filtering works
- [ ] Combined filters (vector + metadata) work

### AC5: Transaction Management
- [ ] Transaction context manager works
- [ ] Multiple operations in single transaction
- [ ] Automatic rollback on errors
- [ ] Commit only on success

### AC6: Error Handling
- [ ] Custom exceptions defined
- [ ] Appropriate exceptions raised
- [ ] Error messages include context
- [ ] Database errors properly caught and wrapped

### AC7: Performance Requirements
- [ ] Single operations <10ms
- [ ] Bulk insert >1000 items/second
- [ ] Vector search <100ms
- [ ] Connection pooling utilized

### AC8: Testing
- [ ] All repository methods have unit tests
- [ ] Integration tests for vector search
- [ ] Transaction rollback tests
- [ ] Error handling tests
- [ ] Performance benchmark tests

---

## Technical Specifications

### File Structure
```
rag_factory/
├── repositories/
│   ├── __init__.py
│   ├── base.py              # Base repository
│   ├── document.py          # DocumentRepository
│   ├── chunk.py             # ChunkRepository
│   └── exceptions.py        # Custom exceptions
│
tests/
├── unit/
│   └── repositories/
│       ├── test_base.py
│       ├── test_document_repository.py
│       └── test_chunk_repository.py
│
├── integration/
│   └── repositories/
│       ├── test_document_integration.py
│       └── test_chunk_integration.py
```

### Base Repository
```python
# rag_factory/repositories/base.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List
from sqlalchemy.orm import Session
from contextlib import contextmanager

T = TypeVar('T')

class BaseRepository(ABC, Generic[T]):
    """Abstract base repository for database operations."""

    def __init__(self, session: Session):
        self.session = session

    @abstractmethod
    def get_by_id(self, entity_id) -> Optional[T]:
        """Get entity by ID."""
        pass

    @abstractmethod
    def create(self, entity) -> T:
        """Create new entity."""
        pass

    @abstractmethod
    def update(self, entity_id, updates) -> T:
        """Update entity."""
        pass

    @abstractmethod
    def delete(self, entity_id) -> bool:
        """Delete entity."""
        pass

    def commit(self):
        """Commit current transaction."""
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise RepositoryError(f"Commit failed: {str(e)}")

    def rollback(self):
        """Rollback current transaction."""
        self.session.rollback()
```

### Custom Exceptions
```python
# rag_factory/repositories/exceptions.py

class RepositoryError(Exception):
    """Base exception for repository errors."""
    pass

class EntityNotFoundError(RepositoryError):
    """Raised when entity is not found."""
    def __init__(self, entity_type: str, entity_id):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} with id {entity_id} not found")

class DuplicateEntityError(RepositoryError):
    """Raised when unique constraint is violated."""
    def __init__(self, entity_type: str, field: str, value):
        self.entity_type = entity_type
        self.field = field
        self.value = value
        super().__init__(f"{entity_type} with {field}={value} already exists")

class DatabaseConnectionError(RepositoryError):
    """Raised when database connection fails."""
    pass

class InvalidQueryError(RepositoryError):
    """Raised when query parameters are invalid."""
    pass
```

### DocumentRepository
```python
# rag_factory/repositories/document.py
from typing import Optional, List
from uuid import UUID
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from .base import BaseRepository
from .exceptions import EntityNotFoundError, DuplicateEntityError, DatabaseConnectionError
from ..database.models import Document

class DocumentRepository(BaseRepository[Document]):
    """Repository for Document operations."""

    def get_by_id(self, document_id: UUID) -> Optional[Document]:
        """Get document by ID."""
        try:
            return self.session.query(Document).filter(
                Document.document_id == document_id
            ).first()
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to query document: {str(e)}")

    def get_by_content_hash(self, content_hash: str) -> Optional[Document]:
        """Get document by content hash for deduplication."""
        try:
            return self.session.query(Document).filter(
                Document.content_hash == content_hash
            ).first()
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to query document: {str(e)}")

    def create(self, filename: str, source_path: str, content_hash: str,
               metadata: dict = None) -> Document:
        """Create new document."""
        # Check for duplicates
        existing = self.get_by_content_hash(content_hash)
        if existing:
            raise DuplicateEntityError("Document", "content_hash", content_hash)

        try:
            document = Document(
                filename=filename,
                source_path=source_path,
                content_hash=content_hash,
                metadata=metadata or {},
                status="pending"
            )
            self.session.add(document)
            self.session.flush()
            return document
        except IntegrityError as e:
            raise DuplicateEntityError("Document", "unknown", str(e))
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to create document: {str(e)}")

    def bulk_create(self, documents: List[dict]) -> List[Document]:
        """Bulk create documents."""
        try:
            doc_objects = [
                Document(
                    filename=doc["filename"],
                    source_path=doc["source_path"],
                    content_hash=doc["content_hash"],
                    metadata=doc.get("metadata", {}),
                    status=doc.get("status", "pending")
                )
                for doc in documents
            ]
            self.session.bulk_save_objects(doc_objects, return_defaults=True)
            self.session.flush()
            return doc_objects
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Bulk create failed: {str(e)}")

    def update(self, document_id: UUID, **updates) -> Document:
        """Update document fields."""
        document = self.get_by_id(document_id)
        if not document:
            raise EntityNotFoundError("Document", document_id)

        try:
            for key, value in updates.items():
                if hasattr(document, key):
                    setattr(document, key, value)
            self.session.flush()
            return document
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to update document: {str(e)}")

    def update_status(self, document_id: UUID, status: str) -> Document:
        """Update document status."""
        return self.update(document_id, status=status)

    def delete(self, document_id: UUID) -> bool:
        """Delete document (cascades to chunks)."""
        document = self.get_by_id(document_id)
        if not document:
            raise EntityNotFoundError("Document", document_id)

        try:
            self.session.delete(document)
            self.session.flush()
            return True
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to delete document: {str(e)}")

    def list_all(self, skip: int = 0, limit: int = 100) -> List[Document]:
        """List documents with pagination."""
        try:
            return self.session.query(Document)\
                .order_by(Document.created_at.desc())\
                .offset(skip)\
                .limit(limit)\
                .all()
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to list documents: {str(e)}")

    def count(self) -> int:
        """Count total documents."""
        try:
            return self.session.query(func.count(Document.document_id)).scalar()
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to count documents: {str(e)}")

    def get_by_status(self, status: str) -> List[Document]:
        """Get documents by status."""
        try:
            return self.session.query(Document).filter(
                Document.status == status
            ).all()
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to query by status: {str(e)}")
```

### ChunkRepository
```python
# rag_factory/repositories/chunk.py
from typing import Optional, List, Tuple
from uuid import UUID
from sqlalchemy import func, text
from sqlalchemy.exc import SQLAlchemyError
from .base import BaseRepository
from .exceptions import EntityNotFoundError, DatabaseConnectionError, InvalidQueryError
from ..database.models import Chunk

class ChunkRepository(BaseRepository[Chunk]):
    """Repository for Chunk operations with vector search."""

    def get_by_id(self, chunk_id: UUID) -> Optional[Chunk]:
        """Get chunk by ID."""
        try:
            return self.session.query(Chunk).filter(
                Chunk.chunk_id == chunk_id
            ).first()
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to query chunk: {str(e)}")

    def get_by_document(self, document_id: UUID, skip: int = 0,
                       limit: int = 100) -> List[Chunk]:
        """Get all chunks for a document."""
        try:
            return self.session.query(Chunk)\
                .filter(Chunk.document_id == document_id)\
                .order_by(Chunk.chunk_index)\
                .offset(skip)\
                .limit(limit)\
                .all()
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to query chunks: {str(e)}")

    def count_by_document(self, document_id: UUID) -> int:
        """Count chunks for a document."""
        try:
            return self.session.query(func.count(Chunk.chunk_id))\
                .filter(Chunk.document_id == document_id)\
                .scalar()
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to count chunks: {str(e)}")

    def create(self, document_id: UUID, chunk_index: int, text: str,
               embedding: Optional[List[float]] = None,
               metadata: dict = None) -> Chunk:
        """Create new chunk."""
        try:
            chunk = Chunk(
                document_id=document_id,
                chunk_index=chunk_index,
                text=text,
                embedding=embedding,
                metadata=metadata or {}
            )
            self.session.add(chunk)
            self.session.flush()
            return chunk
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to create chunk: {str(e)}")

    def bulk_create(self, chunks: List[dict]) -> List[Chunk]:
        """Bulk create chunks for performance."""
        try:
            chunk_objects = [
                Chunk(
                    document_id=chunk["document_id"],
                    chunk_index=chunk["chunk_index"],
                    text=chunk["text"],
                    embedding=chunk.get("embedding"),
                    metadata=chunk.get("metadata", {})
                )
                for chunk in chunks
            ]
            self.session.bulk_save_objects(chunk_objects, return_defaults=True)
            self.session.flush()
            return chunk_objects
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Bulk create failed: {str(e)}")

    def update_embedding(self, chunk_id: UUID, embedding: List[float]) -> Chunk:
        """Update chunk embedding."""
        chunk = self.get_by_id(chunk_id)
        if not chunk:
            raise EntityNotFoundError("Chunk", chunk_id)

        try:
            chunk.embedding = embedding
            self.session.flush()
            return chunk
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to update embedding: {str(e)}")

    def bulk_update_embeddings(self, updates: List[Tuple[UUID, List[float]]]) -> int:
        """Bulk update embeddings."""
        try:
            count = 0
            for chunk_id, embedding in updates:
                self.session.query(Chunk).filter(
                    Chunk.chunk_id == chunk_id
                ).update({"embedding": embedding})
                count += 1
            self.session.flush()
            return count
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Bulk update failed: {str(e)}")

    def search_similar(self, embedding: List[float], top_k: int = 5,
                      threshold: float = 0.0) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks using vector similarity.

        Returns list of (chunk, similarity_score) tuples.
        Similarity score: 1 - cosine_distance (higher is more similar).
        """
        if not embedding:
            raise InvalidQueryError("Embedding vector cannot be empty")
        if top_k < 1:
            raise InvalidQueryError("top_k must be at least 1")

        try:
            # Convert embedding to pgvector format
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            # Query with cosine distance
            query = text(f"""
                SELECT chunk_id, document_id, chunk_index, text, embedding,
                       metadata, created_at, updated_at,
                       1 - (embedding <=> :embedding::vector) as similarity
                FROM chunks
                WHERE embedding IS NOT NULL
                  AND 1 - (embedding <=> :embedding::vector) >= :threshold
                ORDER BY embedding <=> :embedding::vector
                LIMIT :top_k
            """)

            results = self.session.execute(
                query,
                {"embedding": embedding_str, "threshold": threshold, "top_k": top_k}
            ).fetchall()

            # Convert to Chunk objects with similarity scores
            chunks_with_scores = []
            for row in results:
                chunk = Chunk(
                    chunk_id=row[0],
                    document_id=row[1],
                    chunk_index=row[2],
                    text=row[3],
                    embedding=row[4],
                    metadata=row[5],
                    created_at=row[6],
                    updated_at=row[7]
                )
                similarity = float(row[8])
                chunks_with_scores.append((chunk, similarity))

            return chunks_with_scores

        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Vector search failed: {str(e)}")

    def search_similar_with_filter(self, embedding: List[float], top_k: int,
                                   document_ids: List[UUID]) -> List[Tuple[Chunk, float]]:
        """Search similar chunks filtered by document IDs."""
        if not document_ids:
            raise InvalidQueryError("document_ids cannot be empty")

        try:
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            doc_ids_str = ",".join([f"'{str(doc_id)}'" for doc_id in document_ids])

            query = text(f"""
                SELECT chunk_id, document_id, chunk_index, text, embedding,
                       metadata, created_at, updated_at,
                       1 - (embedding <=> :embedding::vector) as similarity
                FROM chunks
                WHERE embedding IS NOT NULL
                  AND document_id IN ({doc_ids_str})
                ORDER BY embedding <=> :embedding::vector
                LIMIT :top_k
            """)

            results = self.session.execute(
                query,
                {"embedding": embedding_str, "top_k": top_k}
            ).fetchall()

            chunks_with_scores = []
            for row in results:
                chunk = Chunk(
                    chunk_id=row[0],
                    document_id=row[1],
                    chunk_index=row[2],
                    text=row[3],
                    embedding=row[4],
                    metadata=row[5],
                    created_at=row[6],
                    updated_at=row[7]
                )
                similarity = float(row[8])
                chunks_with_scores.append((chunk, similarity))

            return chunks_with_scores

        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Filtered vector search failed: {str(e)}")

    def delete(self, chunk_id: UUID) -> bool:
        """Delete chunk."""
        chunk = self.get_by_id(chunk_id)
        if not chunk:
            raise EntityNotFoundError("Chunk", chunk_id)

        try:
            self.session.delete(chunk)
            self.session.flush()
            return True
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to delete chunk: {str(e)}")

    def delete_by_document(self, document_id: UUID) -> int:
        """Delete all chunks for a document."""
        try:
            count = self.session.query(Chunk).filter(
                Chunk.document_id == document_id
            ).delete()
            self.session.flush()
            return count
        except SQLAlchemyError as e:
            raise DatabaseConnectionError(f"Failed to delete chunks: {str(e)}")
```

---

## Unit Tests

### Test File Locations
- `tests/unit/repositories/test_document_repository.py`
- `tests/unit/repositories/test_chunk_repository.py`
- `tests/unit/repositories/test_exceptions.py`

### Test Cases

#### TC2.2.1: DocumentRepository CRUD Tests
```python
import pytest
from uuid import uuid4
from rag_factory.repositories.document import DocumentRepository
from rag_factory.repositories.exceptions import EntityNotFoundError, DuplicateEntityError

@pytest.fixture
def doc_repo(db_session):
    return DocumentRepository(db_session)

def test_create_document(doc_repo):
    """Test creating a new document."""
    doc = doc_repo.create(
        filename="test.txt",
        source_path="/path/test.txt",
        content_hash="hash123"
    )

    assert doc.document_id is not None
    assert doc.filename == "test.txt"
    assert doc.status == "pending"

def test_create_duplicate_document_raises_error(doc_repo):
    """Test creating duplicate document raises DuplicateEntityError."""
    doc_repo.create("test.txt", "/path/test.txt", "hash123")

    with pytest.raises(DuplicateEntityError):
        doc_repo.create("test2.txt", "/path/test2.txt", "hash123")

def test_get_by_id(doc_repo):
    """Test retrieving document by ID."""
    doc = doc_repo.create("test.txt", "/path/test.txt", "hash123")
    retrieved = doc_repo.get_by_id(doc.document_id)

    assert retrieved is not None
    assert retrieved.document_id == doc.document_id

def test_get_by_id_not_found(doc_repo):
    """Test get_by_id returns None for non-existent ID."""
    result = doc_repo.get_by_id(uuid4())
    assert result is None

def test_get_by_content_hash(doc_repo):
    """Test retrieving document by content hash."""
    doc = doc_repo.create("test.txt", "/path/test.txt", "hash123")
    retrieved = doc_repo.get_by_content_hash("hash123")

    assert retrieved is not None
    assert retrieved.document_id == doc.document_id

def test_update_document(doc_repo):
    """Test updating document fields."""
    doc = doc_repo.create("test.txt", "/path/test.txt", "hash123")
    updated = doc_repo.update(doc.document_id, filename="updated.txt")

    assert updated.filename == "updated.txt"

def test_update_status(doc_repo):
    """Test updating document status."""
    doc = doc_repo.create("test.txt", "/path/test.txt", "hash123")
    updated = doc_repo.update_status(doc.document_id, "completed")

    assert updated.status == "completed"

def test_update_nonexistent_raises_error(doc_repo):
    """Test updating non-existent document raises error."""
    with pytest.raises(EntityNotFoundError):
        doc_repo.update(uuid4(), filename="test.txt")

def test_delete_document(doc_repo):
    """Test deleting a document."""
    doc = doc_repo.create("test.txt", "/path/test.txt", "hash123")
    result = doc_repo.delete(doc.document_id)

    assert result is True
    assert doc_repo.get_by_id(doc.document_id) is None

def test_delete_nonexistent_raises_error(doc_repo):
    """Test deleting non-existent document raises error."""
    with pytest.raises(EntityNotFoundError):
        doc_repo.delete(uuid4())

def test_list_all_pagination(doc_repo):
    """Test listing documents with pagination."""
    # Create 10 documents
    for i in range(10):
        doc_repo.create(f"test{i}.txt", f"/path/test{i}.txt", f"hash{i}")

    # Get first page
    page1 = doc_repo.list_all(skip=0, limit=5)
    assert len(page1) == 5

    # Get second page
    page2 = doc_repo.list_all(skip=5, limit=5)
    assert len(page2) == 5

    # Verify no overlap
    page1_ids = {doc.document_id for doc in page1}
    page2_ids = {doc.document_id for doc in page2}
    assert page1_ids.isdisjoint(page2_ids)

def test_count_documents(doc_repo):
    """Test counting total documents."""
    for i in range(5):
        doc_repo.create(f"test{i}.txt", f"/path/test{i}.txt", f"hash{i}")

    count = doc_repo.count()
    assert count == 5

def test_get_by_status(doc_repo):
    """Test filtering documents by status."""
    doc1 = doc_repo.create("test1.txt", "/path/test1.txt", "hash1")
    doc2 = doc_repo.create("test2.txt", "/path/test2.txt", "hash2")
    doc_repo.update_status(doc1.document_id, "completed")

    pending = doc_repo.get_by_status("pending")
    completed = doc_repo.get_by_status("completed")

    assert len(pending) == 1
    assert len(completed) == 1
    assert pending[0].document_id == doc2.document_id

def test_bulk_create_documents(doc_repo):
    """Test bulk creating documents."""
    documents = [
        {"filename": f"test{i}.txt", "source_path": f"/path/test{i}.txt",
         "content_hash": f"hash{i}"}
        for i in range(100)
    ]

    created = doc_repo.bulk_create(documents)
    assert len(created) == 100
    assert all(doc.document_id is not None for doc in created)
```

#### TC2.2.2: ChunkRepository CRUD Tests
```python
import pytest
from uuid import uuid4
import numpy as np
from rag_factory.repositories.chunk import ChunkRepository
from rag_factory.repositories.exceptions import EntityNotFoundError

@pytest.fixture
def chunk_repo(db_session):
    return ChunkRepository(db_session)

@pytest.fixture
def sample_document(doc_repo):
    return doc_repo.create("test.txt", "/path/test.txt", "hash123")

def test_create_chunk(chunk_repo, sample_document):
    """Test creating a chunk."""
    embedding = np.random.rand(1536).tolist()
    chunk = chunk_repo.create(
        document_id=sample_document.document_id,
        chunk_index=0,
        text="Sample text",
        embedding=embedding
    )

    assert chunk.chunk_id is not None
    assert chunk.text == "Sample text"
    assert chunk.embedding is not None

def test_create_chunk_without_embedding(chunk_repo, sample_document):
    """Test creating chunk without embedding."""
    chunk = chunk_repo.create(
        document_id=sample_document.document_id,
        chunk_index=0,
        text="Sample text"
    )

    assert chunk.embedding is None

def test_get_by_id(chunk_repo, sample_document):
    """Test retrieving chunk by ID."""
    chunk = chunk_repo.create(sample_document.document_id, 0, "Test")
    retrieved = chunk_repo.get_by_id(chunk.chunk_id)

    assert retrieved is not None
    assert retrieved.chunk_id == chunk.chunk_id

def test_get_by_document(chunk_repo, sample_document):
    """Test retrieving all chunks for a document."""
    # Create 5 chunks
    for i in range(5):
        chunk_repo.create(sample_document.document_id, i, f"Chunk {i}")

    chunks = chunk_repo.get_by_document(sample_document.document_id)
    assert len(chunks) == 5
    # Should be ordered by chunk_index
    assert chunks[0].chunk_index == 0
    assert chunks[4].chunk_index == 4

def test_count_by_document(chunk_repo, sample_document):
    """Test counting chunks for a document."""
    for i in range(3):
        chunk_repo.create(sample_document.document_id, i, f"Chunk {i}")

    count = chunk_repo.count_by_document(sample_document.document_id)
    assert count == 3

def test_update_embedding(chunk_repo, sample_document):
    """Test updating chunk embedding."""
    chunk = chunk_repo.create(sample_document.document_id, 0, "Test")
    embedding = np.random.rand(1536).tolist()

    updated = chunk_repo.update_embedding(chunk.chunk_id, embedding)
    assert updated.embedding is not None
    assert len(updated.embedding) == 1536

def test_bulk_create_chunks(chunk_repo, sample_document):
    """Test bulk creating chunks."""
    chunks_data = [
        {
            "document_id": sample_document.document_id,
            "chunk_index": i,
            "text": f"Chunk {i}",
            "embedding": np.random.rand(1536).tolist()
        }
        for i in range(100)
    ]

    created = chunk_repo.bulk_create(chunks_data)
    assert len(created) == 100

def test_bulk_update_embeddings(chunk_repo, sample_document):
    """Test bulk updating embeddings."""
    # Create chunks without embeddings
    chunks = [
        chunk_repo.create(sample_document.document_id, i, f"Chunk {i}")
        for i in range(10)
    ]

    # Prepare updates
    updates = [
        (chunk.chunk_id, np.random.rand(1536).tolist())
        for chunk in chunks
    ]

    count = chunk_repo.bulk_update_embeddings(updates)
    assert count == 10

def test_delete_chunk(chunk_repo, sample_document):
    """Test deleting a chunk."""
    chunk = chunk_repo.create(sample_document.document_id, 0, "Test")
    result = chunk_repo.delete(chunk.chunk_id)

    assert result is True
    assert chunk_repo.get_by_id(chunk.chunk_id) is None

def test_delete_by_document(chunk_repo, sample_document):
    """Test deleting all chunks for a document."""
    for i in range(5):
        chunk_repo.create(sample_document.document_id, i, f"Chunk {i}")

    count = chunk_repo.delete_by_document(sample_document.document_id)
    assert count == 5

    remaining = chunk_repo.count_by_document(sample_document.document_id)
    assert remaining == 0
```

#### TC2.2.3: Vector Search Tests
```python
import pytest
import numpy as np
from rag_factory.repositories.chunk import ChunkRepository
from rag_factory.repositories.exceptions import InvalidQueryError

def test_search_similar(chunk_repo, sample_document):
    """Test vector similarity search."""
    # Create base embedding
    base_embedding = np.random.rand(1536)

    # Create similar and dissimilar embeddings
    similar = base_embedding + np.random.rand(1536) * 0.1
    dissimilar = np.random.rand(1536)

    # Create chunks
    chunk_repo.create(sample_document.document_id, 0, "Similar", similar.tolist())
    chunk_repo.create(sample_document.document_id, 1, "Dissimilar", dissimilar.tolist())

    # Search
    results = chunk_repo.search_similar(base_embedding.tolist(), top_k=2)

    assert len(results) == 2
    # First result should be more similar
    assert results[0][1] > results[1][1]
    assert results[0][0].text == "Similar"

def test_search_similar_with_threshold(chunk_repo, sample_document):
    """Test similarity search with threshold filtering."""
    base_embedding = np.random.rand(1536)
    similar = base_embedding + np.random.rand(1536) * 0.1
    dissimilar = np.random.rand(1536)

    chunk_repo.create(sample_document.document_id, 0, "Similar", similar.tolist())
    chunk_repo.create(sample_document.document_id, 1, "Dissimilar", dissimilar.tolist())

    # High threshold should filter out dissimilar
    results = chunk_repo.search_similar(
        base_embedding.tolist(),
        top_k=10,
        threshold=0.8
    )

    assert len(results) <= 2
    # All results should meet threshold
    assert all(score >= 0.8 for _, score in results)

def test_search_similar_empty_embedding_raises_error(chunk_repo):
    """Test search with empty embedding raises error."""
    with pytest.raises(InvalidQueryError):
        chunk_repo.search_similar([], top_k=5)

def test_search_similar_invalid_top_k_raises_error(chunk_repo):
    """Test search with invalid top_k raises error."""
    embedding = np.random.rand(1536).tolist()

    with pytest.raises(InvalidQueryError):
        chunk_repo.search_similar(embedding, top_k=0)

def test_search_similar_with_filter(chunk_repo, doc_repo):
    """Test vector search filtered by document IDs."""
    # Create two documents
    doc1 = doc_repo.create("doc1.txt", "/path/doc1.txt", "hash1")
    doc2 = doc_repo.create("doc2.txt", "/path/doc2.txt", "hash2")

    embedding = np.random.rand(1536).tolist()

    # Create chunks for both documents
    chunk_repo.create(doc1.document_id, 0, "Doc1 chunk", embedding)
    chunk_repo.create(doc2.document_id, 0, "Doc2 chunk", embedding)

    # Search filtered to doc1 only
    results = chunk_repo.search_similar_with_filter(
        embedding,
        top_k=10,
        document_ids=[doc1.document_id]
    )

    assert len(results) == 1
    assert results[0][0].document_id == doc1.document_id

def test_search_returns_similarity_scores(chunk_repo, sample_document):
    """Test that search returns correct similarity scores."""
    embedding = np.random.rand(1536).tolist()
    chunk_repo.create(sample_document.document_id, 0, "Test", embedding)

    results = chunk_repo.search_similar(embedding, top_k=1)

    # Searching for identical embedding should return ~1.0 similarity
    assert len(results) == 1
    chunk, score = results[0]
    assert 0.99 <= score <= 1.0
```

#### TC2.2.4: Transaction Tests
```python
import pytest
from rag_factory.repositories.document import DocumentRepository
from rag_factory.repositories.chunk import ChunkRepository

def test_transaction_commit(db_session):
    """Test successful transaction commits changes."""
    doc_repo = DocumentRepository(db_session)

    doc = doc_repo.create("test.txt", "/path/test.txt", "hash123")
    doc_repo.commit()

    # Verify in new session
    db_session.close()
    new_session = get_new_session()
    doc_repo2 = DocumentRepository(new_session)
    retrieved = doc_repo2.get_by_id(doc.document_id)

    assert retrieved is not None

def test_transaction_rollback(db_session):
    """Test rollback reverts changes."""
    doc_repo = DocumentRepository(db_session)

    doc = doc_repo.create("test.txt", "/path/test.txt", "hash123")
    doc_id = doc.document_id

    doc_repo.rollback()

    # Document should not exist after rollback
    retrieved = doc_repo.get_by_id(doc_id)
    assert retrieved is None

def test_multiple_operations_in_transaction(db_session):
    """Test multiple operations in single transaction."""
    doc_repo = DocumentRepository(db_session)
    chunk_repo = ChunkRepository(db_session)

    # Create document and chunks in same transaction
    doc = doc_repo.create("test.txt", "/path/test.txt", "hash123")
    chunk1 = chunk_repo.create(doc.document_id, 0, "Chunk 1")
    chunk2 = chunk_repo.create(doc.document_id, 1, "Chunk 2")

    doc_repo.commit()

    # Both document and chunks should exist
    assert doc_repo.get_by_id(doc.document_id) is not None
    assert chunk_repo.count_by_document(doc.document_id) == 2

def test_automatic_rollback_on_error(db_session):
    """Test transaction rolls back automatically on error."""
    doc_repo = DocumentRepository(db_session)

    try:
        doc1 = doc_repo.create("test.txt", "/path/test.txt", "hash123")
        # Try to create duplicate (should fail)
        doc2 = doc_repo.create("test2.txt", "/path/test2.txt", "hash123")
        doc_repo.commit()
    except:
        doc_repo.rollback()

    # First document should not be committed
    count = doc_repo.count()
    assert count == 0
```

---

## Integration Tests

### Test File Location
`tests/integration/repositories/test_repository_integration.py`

### Test Scenarios

#### IS2.2.1: Full Repository Workflow
```python
@pytest.mark.integration
def test_complete_document_lifecycle(db_session):
    """Test complete document and chunk lifecycle."""
    doc_repo = DocumentRepository(db_session)
    chunk_repo = ChunkRepository(db_session)

    # Create document
    doc = doc_repo.create("article.txt", "/docs/article.txt", "abc123")
    assert doc.status == "pending"

    # Add chunks with embeddings
    embeddings = [np.random.rand(1536).tolist() for _ in range(5)]
    chunks = []
    for i, emb in enumerate(embeddings):
        chunk = chunk_repo.create(
            document_id=doc.document_id,
            chunk_index=i,
            text=f"Paragraph {i}",
            embedding=emb
        )
        chunks.append(chunk)

    # Update document status
    doc = doc_repo.update_status(doc.document_id, "completed")
    assert doc.status == "completed"

    # Perform vector search
    query_embedding = embeddings[0]
    results = chunk_repo.search_similar(query_embedding, top_k=3)

    assert len(results) >= 1
    # First result should be the matching chunk
    assert results[0][0].chunk_index == 0

    # Delete document (should cascade to chunks)
    doc_repo.delete(doc.document_id)
    remaining_chunks = chunk_repo.count_by_document(doc.document_id)
    assert remaining_chunks == 0

@pytest.mark.integration
def test_concurrent_repository_access(db_pool):
    """Test multiple repository instances accessing database concurrently."""
    import concurrent.futures

    def create_document(thread_id):
        session = db_pool.get_session()
        doc_repo = DocumentRepository(session)
        doc = doc_repo.create(
            f"doc_{thread_id}.txt",
            f"/path/doc_{thread_id}.txt",
            f"hash_{thread_id}"
        )
        doc_repo.commit()
        session.close()
        return doc.document_id

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(create_document, i) for i in range(20)]
        doc_ids = [f.result() for f in concurrent.futures.as_completed(futures)]

    # All documents should be created
    assert len(doc_ids) == 20
    assert len(set(doc_ids)) == 20  # All unique
```

---

## Definition of Done

- [ ] BaseRepository abstract class implemented
- [ ] DocumentRepository fully implemented with all methods
- [ ] ChunkRepository fully implemented with all methods
- [ ] Custom exception classes defined
- [ ] All unit tests pass (>90% coverage)
- [ ] All integration tests pass
- [ ] Vector search tested with real embeddings
- [ ] Transaction management tested
- [ ] Performance benchmarks meet requirements
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] No linting errors

---

## Performance Benchmarks

Run these benchmarks to verify performance requirements:

```python
# tests/benchmarks/test_repository_performance.py

import pytest
import time
import numpy as np

@pytest.mark.benchmark
def test_single_document_operation_performance(doc_repo):
    """Single document operations should be <10ms."""
    start = time.time()
    doc = doc_repo.create("test.txt", "/path/test.txt", "hash123")
    duration = (time.time() - start) * 1000  # Convert to ms

    assert duration < 10, f"Create took {duration:.2f}ms (expected <10ms)"

@pytest.mark.benchmark
def test_bulk_insert_performance(chunk_repo, sample_document):
    """Bulk insert should handle >1000 chunks/second."""
    chunks_data = [
        {
            "document_id": sample_document.document_id,
            "chunk_index": i,
            "text": f"Chunk {i}",
            "embedding": np.random.rand(1536).tolist()
        }
        for i in range(5000)
    ]

    start = time.time()
    chunk_repo.bulk_create(chunks_data)
    duration = time.time() - start

    throughput = 5000 / duration
    assert throughput > 1000, f"Throughput {throughput:.0f} chunks/sec (expected >1000)"

@pytest.mark.benchmark
def test_vector_search_performance(chunk_repo, sample_document):
    """Vector search should be <100ms."""
    # Create 10000 chunks with embeddings
    chunks_data = [
        {
            "document_id": sample_document.document_id,
            "chunk_index": i,
            "text": f"Chunk {i}",
            "embedding": np.random.rand(1536).tolist()
        }
        for i in range(10000)
    ]
    chunk_repo.bulk_create(chunks_data)

    # Perform search
    query_embedding = np.random.rand(1536).tolist()
    start = time.time()
    results = chunk_repo.search_similar(query_embedding, top_k=10)
    duration = (time.time() - start) * 1000

    assert duration < 100, f"Search took {duration:.2f}ms (expected <100ms)"
```

---

## Notes for Developers

1. **Session Management**: Always use repositories with a session from the connection pool.

2. **Transaction Boundaries**: Call `commit()` or `rollback()` explicitly for transaction control.

3. **Error Handling**: Catch specific repository exceptions and handle appropriately.

4. **Performance**: Use bulk operations for inserting/updating many entities.

5. **Vector Search**: Embedding dimensions must match the database column definition (default 1536).

6. **Testing**: Use test database and fixtures to isolate tests.

7. **Type Hints**: All repository methods have complete type hints for better IDE support.
