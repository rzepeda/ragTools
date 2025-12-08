# Story 12.2: Create IIndexingStrategy Interface

**Story ID:** 12.2
**Epic:** Epic 12 - Indexing/Retrieval Pipeline Separation & Capability System
**Story Points:** 8
**Priority:** High
**Dependencies:** Story 12.1, Epic 11 (Dependency Injection)

---

## User Story

**As a** developer
**I want** a separate interface for indexing strategies
**So that** indexing is independent from retrieval

---

## Detailed Requirements

### Functional Requirements

1.  **Indexing Context**
    -   Create `IndexingContext` class
    -   Hold shared resources (database service)
    -   Hold configuration
    -   Hold metrics dictionary

2.  **IIndexingStrategy Interface**
    -   Define `IIndexingStrategy` abstract base class
    -   Initialize with config and dependencies (from Epic 11)
    -   Validate dependencies on initialization
    -   Define `produces()` abstract method (returns capabilities)
    -   Define `requires_services()` abstract method (returns service dependencies)
    -   Define `process()` abstract method (indexes documents)

3.  **Example Implementation**
    -   Create `VectorEmbeddingIndexing` example class
    -   Implement all abstract methods
    -   Demonstrate dependency usage
    -   Demonstrate context usage

### Non-Functional Requirements

1.  **Maintainability**
    -   Clear interface contract
    -   Type hints for all methods
    -   Docstrings for all methods

2.  **Extensibility**
    -   Easy to add new indexing strategies
    -   Support for async operations

---

## Acceptance Criteria

### AC1: Indexing Context
- [ ] `IndexingContext` class implemented
- [ ] Stores database service and config
- [ ] Supports metrics tracking

### AC2: IIndexingStrategy Interface
- [ ] `IIndexingStrategy` ABC defined
- [ ] `__init__` validates dependencies
- [ ] `produces` method defined
- [ ] `requires_services` method defined
- [ ] `process` method defined

### AC3: Example Implementation
- [ ] `VectorEmbeddingIndexing` implemented
- [ ] Correctly declares capabilities and services
- [ ] `process` method works as expected

### AC4: Testing
- [ ] Unit tests for `IndexingContext`
- [ ] Unit tests for `IIndexingStrategy` (using mock implementation)
- [ ] Unit tests for `VectorEmbeddingIndexing`

---

## Technical Specifications

### File Structure
```
rag_factory/
├── core/
│   ├── indexing_interface.py # New file for indexing interface
│   └── ...
tests/
├── unit/
│   ├── core/
│   │   └── test_indexing_interface.py
│   └── ...
```

### Code Definition

```python
from abc import ABC, abstractmethod
from typing import List, Set, Dict, Any
from .capabilities import IndexCapability, IndexingResult

class IndexingContext:
    """Shared context for indexing operations"""
    
    def __init__(
        self,
        database_service: 'IDatabaseService',
        config: Dict[str, Any] = None
    ):
        self.database = database_service
        self.config = config or {}
        self.metrics = {}  # For tracking performance

class IIndexingStrategy(ABC):
    """Interface for document indexing strategies"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        dependencies: 'StrategyDependencies'  # From Epic 11
    ):
        """
        Initialize indexing strategy.
        
        Args:
            config: Strategy-specific configuration
            dependencies: Injected services (validated at creation)
        """
        self.config = config
        self.deps = dependencies
        
        # Validate dependencies
        from rag_factory.core.dependencies import validate_dependencies
        validate_dependencies(self.deps, self.requires_services())
    
    @abstractmethod
    def produces(self) -> Set[IndexCapability]:
        """
        Declare what capabilities this strategy produces.
        
        Returns:
            Set of IndexCapability enums
            
        Example:
            return {IndexCapability.VECTORS, IndexCapability.CHUNKS, IndexCapability.DATABASE}
        """
        pass
    
    @abstractmethod
    def requires_services(self) -> Set['ServiceDependency']:
        """
        Declare what services this strategy requires.
        
        Returns:
            Set of ServiceDependency enums (from Epic 11)
            
        Example:
            return {ServiceDependency.EMBEDDING, ServiceDependency.DATABASE}
        """
        pass
    
    @abstractmethod
    async def process(
        self,
        documents: List['Document'],
        context: IndexingContext
    ) -> IndexingResult:
        """
        Process documents for indexing.
        
        Args:
            documents: Documents to index
            context: Shared indexing context
            
        Returns:
            IndexingResult with capabilities produced and metadata
        """
        pass
```
