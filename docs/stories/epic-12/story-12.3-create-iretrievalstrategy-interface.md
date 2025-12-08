# Story 12.3: Create IRetrievalStrategy Interface

**Story ID:** 12.3
**Epic:** Epic 12 - Indexing/Retrieval Pipeline Separation & Capability System
**Story Points:** 8
**Priority:** High
**Dependencies:** Story 12.1, Epic 11 (Dependency Injection)

---

## User Story

**As a** developer
**I want** retrieval strategies to declare capability requirements
**So that** compatibility can be validated

---

## Detailed Requirements

### Functional Requirements

1.  **Retrieval Context**
    -   Create `RetrievalContext` class
    -   Hold shared resources (database service)
    -   Hold configuration
    -   Hold metrics dictionary

2.  **IRetrievalStrategy Interface**
    -   Define `IRetrievalStrategy` abstract base class
    -   Initialize with config and dependencies (from Epic 11)
    -   Validate dependencies on initialization
    -   Define `requires()` abstract method (returns required capabilities)
    -   Define `requires_services()` abstract method (returns service dependencies)
    -   Define `retrieve()` abstract method (retrieves chunks)
    -   Remove `prepare_data()` method (legacy)

3.  **Example Implementation**
    -   Create `RerankingRetrieval` example class
    -   Implement all abstract methods
    -   Demonstrate dependency usage
    -   Demonstrate context usage
    -   Demonstrate multi-step retrieval (search + rerank)

### Non-Functional Requirements

1.  **Maintainability**
    -   Clear interface contract
    -   Type hints for all methods
    -   Docstrings for all methods

2.  **Extensibility**
    -   Easy to add new retrieval strategies
    -   Support for async operations

---

## Acceptance Criteria

### AC1: Retrieval Context
- [ ] `RetrievalContext` class implemented
- [ ] Stores database service and config
- [ ] Supports metrics tracking

### AC2: IRetrievalStrategy Interface
- [ ] `IRetrievalStrategy` ABC defined
- [ ] `__init__` validates dependencies
- [ ] `requires` method defined
- [ ] `requires_services` method defined
- [ ] `retrieve` method defined

### AC3: Example Implementation
- [ ] `RerankingRetrieval` implemented
- [ ] Correctly declares requirements and services
- [ ] `retrieve` method works as expected

### AC4: Testing
- [ ] Unit tests for `RetrievalContext`
- [ ] Unit tests for `IRetrievalStrategy` (using mock implementation)
- [ ] Unit tests for `RerankingRetrieval`

---

## Technical Specifications

### File Structure
```
rag_factory/
├── core/
│   ├── retrieval_interface.py # New file for retrieval interface
│   └── ...
tests/
├── unit/
│   ├── core/
│   │   └── test_retrieval_interface.py
│   └── ...
```

### Code Definition

```python
from abc import ABC, abstractmethod
from typing import List, Set, Dict, Any
from .capabilities import IndexCapability

class RetrievalContext:
    """Shared context for retrieval operations"""
    
    def __init__(
        self,
        database_service: 'IDatabaseService',
        config: Dict[str, Any] = None
    ):
        self.database = database_service
        self.config = config or {}
        self.metrics = {}

class IRetrievalStrategy(ABC):
    """Interface for document retrieval strategies"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        dependencies: 'StrategyDependencies'
    ):
        """
        Initialize retrieval strategy.
        
        Args:
            config: Strategy-specific configuration
            dependencies: Injected services
        """
        self.config = config
        self.deps = dependencies
        
        # Validate dependencies
        from rag_factory.core.dependencies import validate_dependencies
        validate_dependencies(self.deps, self.requires_services())
    
    @abstractmethod
    def requires(self) -> Set[IndexCapability]:
        """
        Declare what index capabilities this strategy requires.
        
        Returns:
            Set of required IndexCapability enums
            
        Example:
            return {IndexCapability.VECTORS, IndexCapability.CHUNKS}
        """
        pass
    
    @abstractmethod
    def requires_services(self) -> Set['ServiceDependency']:
        """
        Declare what services this strategy requires.
        
        Returns:
            Set of ServiceDependency enums
            
        Example:
            return {ServiceDependency.LLM, ServiceDependency.DATABASE}
        """
        pass
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        context: RetrievalContext,
        top_k: int = 10
    ) -> List['Chunk']:
        """
        Retrieve relevant chunks for query.
        
        Args:
            query: User query
            context: Shared retrieval context
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with scores
        """
        pass
```
