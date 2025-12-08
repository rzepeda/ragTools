# Story 11.1: Define Service Interfaces

**Story ID:** 11.1
**Epic:** Epic 11 - Dependency Injection & Service Interface Decoupling
**Story Points:** 8
**Priority:** High
**Dependencies:** None

---

## User Story

**As a** developer
**I want** clear interfaces for all external services
**So that** strategies depend on contracts, not implementations

---

## Detailed Requirements

### Functional Requirements

1.  **ILLMService Interface**
    - Define `complete()` method for text generation
    - Define `stream_complete()` method for streaming generation
    - Support standard parameters: prompt, max_tokens, temperature
    - Support provider-specific kwargs
    - Return type hints for all methods

2.  **IEmbeddingService Interface**
    - Define `embed()` method for single text
    - Define `embed_batch()` method for multiple texts
    - Define `get_dimension()` method
    - Return type hints (list of floats)

3.  **IGraphService Interface**
    - Define `create_node()` method
    - Define `create_relationship()` method
    - Define `query()` method for Cypher execution
    - Support properties for nodes and relationships

4.  **IRerankingService Interface**
    - Define `rerank()` method
    - Input: query, list of documents, top_k
    - Output: list of (index, score) tuples

5.  **IDatabaseService Interface**
    - Define `store_chunks()` method
    - Define `search_chunks()` method
    - Define `get_chunk()` method
    - Support vector search parameters

### Non-Functional Requirements

1.  **Code Quality**
    - All interfaces must be Abstract Base Classes (ABC)
    - Full type hinting for all methods
    - Docstrings for all classes and methods (Google style)
    - No implementation logic in interfaces (pure contracts)

2.  **Documentation**
    - Usage examples for each interface
    - Clear parameter descriptions

---

## Acceptance Criteria

### AC1: ILLMService
- [ ] Interface defined as ABC
- [ ] `complete` method signature matches requirements
- [ ] `stream_complete` method signature matches requirements
- [ ] Type hints correct
- [ ] Docstrings complete

### AC2: IEmbeddingService
- [ ] Interface defined as ABC
- [ ] `embed` method signature matches requirements
- [ ] `embed_batch` method signature matches requirements
- [ ] `get_dimension` method included
- [ ] Type hints correct

### AC3: IGraphService
- [ ] Interface defined as ABC
- [ ] `create_node` method signature matches requirements
- [ ] `create_relationship` method signature matches requirements
- [ ] `query` method signature matches requirements
- [ ] Type hints correct

### AC4: IRerankingService
- [ ] Interface defined as ABC
- [ ] `rerank` method signature matches requirements
- [ ] Type hints correct

### AC5: IDatabaseService
- [ ] Interface defined as ABC
- [ ] `store_chunks` method signature matches requirements
- [ ] `search_chunks` method signature matches requirements
- [ ] `get_chunk` method signature matches requirements
- [ ] Type hints correct

### AC6: Testing
- [ ] Unit tests creating mock implementations of each interface
- [ ] Verify that concrete classes must implement abstract methods

---

## Technical Specifications

### File Structure
```
rag_factory/
├── services/
│   ├── __init__.py
│   ├── interfaces.py        # All service interfaces
```

### Interface Definitions
```python
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, List, Dict, Any, Tuple

class ILLMService(ABC):
    """Interface for Large Language Model services"""
    
    @abstractmethod
    async def complete(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate completion for prompt."""
        pass
    
    @abstractmethod
    async def stream_complete(
        self, 
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream completion tokens."""
        pass

class IEmbeddingService(ABC):
    """Interface for embedding generation services"""
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass

class IGraphService(ABC):
    """Interface for graph database services"""
    
    @abstractmethod
    async def create_node(self, label: str, properties: Dict[str, Any]) -> str:
        """Create a node in the graph."""
        pass
    
    @abstractmethod
    async def create_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """Create relationship between nodes."""
        pass
    
    @abstractmethod
    async def query(self, cypher_query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute Cypher query."""
        pass

class IRerankingService(ABC):
    """Interface for document reranking services"""
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Rerank documents by relevance to query."""
        pass

class IDatabaseService(ABC):
    """Interface for database operations"""
    
    @abstractmethod
    async def store_chunks(self, chunks: List[Dict[str, Any]]):
        """Store document chunks"""
        pass
    
    @abstractmethod
    async def search_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search chunks by similarity"""
        pass
    
    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> Dict[str, Any]:
        """Retrieve chunk by ID"""
        pass
```
