# Story 11.5: Update RAGFactory for DI

**Story ID:** 11.5
**Epic:** Epic 11 - Dependency Injection & Service Interface Decoupling
**Story Points:** 3
**Priority:** High
**Dependencies:** Story 11.4

---

## User Story

**As a** developer
**I want** the factory to inject dependencies into strategies
**So that** service configuration is centralized

---

## Detailed Requirements

### Functional Requirements

1.  **RAGFactory Constructor**
    - Accept all service implementations as optional arguments
    - Create and store a `StrategyDependencies` instance
    - Allow initializing with no services (for partial usage)

2.  **create_strategy Method**
    - Accept `override_deps` optional argument
    - Pass dependencies to strategy constructor
    - Use factory-level dependencies by default, override if provided

3.  **Error Handling**
    - Catch `ValueError` from strategy initialization (missing services)
    - Re-raise or wrap with context

### Non-Functional Requirements

1.  **Ease of Use**
    - Factory should be the main entry point for strategy creation
    - Simple API for common cases

2.  **Documentation**
    - Examples of creating factory with different service combinations (Local vs API)

---

## Acceptance Criteria

### AC1: Factory Initialization
- [ ] Constructor accepts all 5 service types
- [ ] Creates `StrategyDependencies` internally

### AC2: Strategy Creation
- [ ] `create_strategy` passes dependencies to strategy
- [ ] Supports dependency overrides

### AC3: Integration
- [ ] Verify factory works with updated strategies from Story 11.4
- [ ] Verify missing services raise appropriate errors

### AC4: Testing
- [ ] Unit tests for factory with various service configurations
- [ ] Test overrides

---

## Technical Specifications

### Implementation Details

```python
from typing import Optional, Dict, Any
from .services.interfaces import (
    ILLMService, IEmbeddingService, IGraphService, 
    IDatabaseService, IRerankingService
)
from .services.dependencies import StrategyDependencies

class RAGFactory:
    """Factory for creating strategies with dependency injection"""
    
    def __init__(
        self,
        llm_service: Optional[ILLMService] = None,
        embedding_service: Optional[IEmbeddingService] = None,
        graph_service: Optional[IGraphService] = None,
        database_service: Optional[IDatabaseService] = None,
        reranker_service: Optional[IRerankingService] = None
    ):
        self.dependencies = StrategyDependencies(
            llm_service=llm_service,
            embedding_service=embedding_service,
            graph_service=graph_service,
            database_service=database_service,
            reranker_service=reranker_service
        )
        self._strategy_registry = {} # Assume populated
    
    def create_strategy(
        self,
        strategy_name: str,
        config: Dict[str, Any],
        override_deps: Optional[StrategyDependencies] = None
    ):
        if strategy_name not in self._strategy_registry:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy_class = self._strategy_registry[strategy_name]
        deps = override_deps or self.dependencies
        
        # Strategy constructor will validate dependencies
        return strategy_class(config=config, dependencies=deps)
```
