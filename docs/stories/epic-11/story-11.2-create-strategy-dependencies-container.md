# Story 11.2: Create StrategyDependencies Container

**Story ID:** 11.2
**Epic:** Epic 11 - Dependency Injection & Service Interface Decoupling
**Story Points:** 5
**Priority:** High
**Dependencies:** Story 11.1

---

## User Story

**As a** developer
**I want** a container for strategy dependencies
**So that** dependencies can be injected and validated

---

## Detailed Requirements

### Functional Requirements

1.  **StrategyDependencies Dataclass**
    - Container for all service interfaces defined in Story 11.1
    - Fields: `llm_service`, `embedding_service`, `graph_service`, `database_service`, `reranker_service`
    - All fields optional (default to None)
    - Type hints using interfaces from Story 11.1

2.  **Validation Logic**
    - `validate_for_strategy(required_services)` method
    - Input: Set of required `ServiceDependency` enums
    - Output: Tuple (is_valid, missing_services)
    - Check if required services are present in the container

3.  **ServiceDependency Enum**
    - Enumeration of all available service types
    - Values: LLM, EMBEDDING, GRAPH, DATABASE, RERANKER

4.  **Error Messaging**
    - `get_missing_services_message(required_services)` method
    - Return user-friendly string listing missing services

### Non-Functional Requirements

1.  **Code Quality**
    - Immutable dataclass (frozen=True) preferred if possible, or standard dataclass
    - Pure python, no external dependencies other than standard library and interfaces

2.  **Documentation**
    - Usage examples showing how to create and validate dependencies

---

## Acceptance Criteria

### AC1: ServiceDependency Enum
- [ ] Enum defined with all 5 service types
- [ ] Auto values used

### AC2: StrategyDependencies Container
- [ ] Dataclass defined
- [ ] All 5 service fields present
- [ ] All fields optional
- [ ] Correct type hints used

### AC3: Validation Logic
- [ ] `validate_for_strategy` correctly identifies missing services
- [ ] Returns True if all required services are present
- [ ] Returns False and list of missing if any are missing
- [ ] Handles empty requirements correctly (returns True)

### AC4: Error Messaging
- [ ] `get_missing_services_message` returns clear string
- [ ] Lists all missing services by name

### AC5: Testing
- [ ] Unit tests for validation logic with various combinations
- [ ] Unit tests for error message generation

---

## Technical Specifications

### File Structure
```
rag_factory/
├── services/
│   ├── __init__.py
│   ├── dependencies.py      # StrategyDependencies and ServiceDependency
```

### Implementation Details
```python
from dataclasses import dataclass
from typing import Optional, Set, Tuple, List
from enum import Enum, auto
from .interfaces import (
    ILLMService, 
    IEmbeddingService, 
    IGraphService, 
    IDatabaseService, 
    IRerankingService
)

class ServiceDependency(Enum):
    """Services that strategies may depend on"""
    LLM = auto()
    EMBEDDING = auto()
    GRAPH = auto()
    DATABASE = auto()
    RERANKER = auto()

@dataclass
class StrategyDependencies:
    """Container for injected services"""
    
    llm_service: Optional[ILLMService] = None
    embedding_service: Optional[IEmbeddingService] = None
    graph_service: Optional[IGraphService] = None
    database_service: Optional[IDatabaseService] = None
    reranker_service: Optional[IRerankingService] = None
    
    def validate_for_strategy(
        self,
        required_services: Set[ServiceDependency]
    ) -> Tuple[bool, List[ServiceDependency]]:
        """Validate that all required services are present."""
        missing = []
        
        if ServiceDependency.LLM in required_services and not self.llm_service:
            missing.append(ServiceDependency.LLM)
        if ServiceDependency.EMBEDDING in required_services and not self.embedding_service:
            missing.append(ServiceDependency.EMBEDDING)
        if ServiceDependency.GRAPH in required_services and not self.graph_service:
            missing.append(ServiceDependency.GRAPH)
        if ServiceDependency.DATABASE in required_services and not self.database_service:
            missing.append(ServiceDependency.DATABASE)
        if ServiceDependency.RERANKER in required_services and not self.reranker_service:
            missing.append(ServiceDependency.RERANKER)
        
        return (len(missing) == 0, missing)
    
    def get_missing_services_message(
        self,
        required_services: Set[ServiceDependency]
    ) -> str:
        """Get user-friendly error message for missing services."""
        is_valid, missing = self.validate_for_strategy(required_services)
        
        if is_valid:
            return ""
        
        service_names = [s.name for s in missing]
        return f"Missing required services: {', '.join(service_names)}"
```
