# Story 11.4: Update Strategy Base Classes for DI

**Story ID:** 11.4
**Epic:** Epic 11 - Dependency Injection & Service Interface Decoupling
**Story Points:** 5
**Priority:** High
**Dependencies:** Story 11.2

---

## User Story

**As a** developer
**I want** strategy base classes to accept dependencies
**So that** all strategies use dependency injection

---

## Detailed Requirements

### Functional Requirements

1.  **BaseStrategy Update**
    - Update `__init__` to accept `dependencies: StrategyDependencies`
    - Store dependencies in `self.deps`
    - Add `requires_services()` abstract method
    - Validate dependencies on initialization using `validate_for_strategy`
    - Raise `ValueError` with clear message if services are missing

2.  **Strategy Migration**
    - Update all existing strategies to use the new constructor signature
    - Implement `requires_services()` in all concrete strategies
    - Replace direct service instantiation/usage with `self.deps.service_name`

3.  **Backward Compatibility**
    - Optional: Support legacy initialization for a transition period (if needed)

### Non-Functional Requirements

1.  **Code Quality**
    - Clean error handling
    - Type hinting
    - Docstrings

2.  **Documentation**
    - Update developer guide on how to create new strategies with DI

---

## Acceptance Criteria

### AC1: BaseStrategy
- [ ] `__init__` accepts `StrategyDependencies`
- [ ] `requires_services` abstract method defined
- [ ] Validation logic implemented in `__init__`
- [ ] Raises `ValueError` for missing services

### AC2: Strategy Updates
- [ ] `QueryExpansionStrategy` updated
- [ ] `AgenticRAGStrategy` updated
- [ ] `KnowledgeGraphStrategy` updated
- [ ] `ContextualRetrievalStrategy` updated
- [ ] All other strategies updated

### AC3: Testing
- [ ] Unit tests for `BaseStrategy` validation
- [ ] Verify strategies fail to initialize without required services
- [ ] Verify strategies work with injected services

---

## Technical Specifications

### Implementation Details

```python
from abc import ABC, abstractmethod
from typing import Set, Dict, Any
from .dependencies import StrategyDependencies, ServiceDependency

class BaseStrategy(ABC):
    """Base class for all strategies"""
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        dependencies: StrategyDependencies
    ):
        """
        Initialize strategy with configuration and dependencies.
        
        Args:
            config: Strategy-specific configuration
            dependencies: Injected services
            
        Raises:
            ValueError: If required services are missing
        """
        self.config = config
        self.deps = dependencies
        
        # Validate dependencies
        required = self.requires_services()
        is_valid, missing = dependencies.validate_for_strategy(required)
        
        if not is_valid:
            service_names = [s.name for s in missing]
            raise ValueError(
                f"{self.__class__.__name__} requires services: {', '.join(service_names)}"
            )
    
    @abstractmethod
    def requires_services(self) -> Set[ServiceDependency]:
        """
        Declare what services this strategy requires.
        
        Returns:
            Set of required ServiceDependency enums
        """
        pass
```

**Example Strategy Implementation:**
```python
class QueryExpansionStrategy(BaseStrategy):
    """Query expansion using LLM"""
    
    def requires_services(self) -> Set[ServiceDependency]:
        return {ServiceDependency.LLM}
    
    async def expand_query(self, query: str) -> str:
        # Use injected LLM service
        prompt = f"Expand this query: {query}"
        return await self.deps.llm_service.complete(prompt)
```
