# Story 11.6: Implement Consistency Checker

**Story ID:** 11.6
**Epic:** Epic 11 - Dependency Injection & Service Interface Decoupling
**Story Points:** 5
**Priority:** Medium
**Dependencies:** Story 11.5

---

## User Story

**As a** developer
**I want** warnings when capabilities and services are inconsistent
**So that** I can catch mistakes without blocking valid edge cases

---

## Detailed Requirements

### Functional Requirements

1.  **ConsistencyChecker Class**
    - `check_indexing_strategy(strategy)`: Returns list of warnings
    - `check_retrieval_strategy(strategy)`: Returns list of warnings
    - Warn, do not error (allow flexibility)

2.  **Indexing Checks**
    - VECTORS capability -> Requires EMBEDDING service
    - GRAPH capability -> Requires GRAPH service
    - DATABASE capability -> Requires DATABASE service
    - IN_MEMORY capability -> Should NOT require DATABASE service (usually)

3.  **Retrieval Checks**
    - VECTORS requirement -> Requires DATABASE service (for search)
    - GRAPH requirement -> Requires GRAPH service
    - KEYWORDS requirement -> Requires DATABASE service

### Non-Functional Requirements

1.  **Usability**
    - Warning messages should be clear and actionable
    - Explain *why* it's inconsistent

2.  **Integration**
    - Integrate into Factory (log warnings on strategy creation)

---

## Acceptance Criteria

### AC1: ConsistencyChecker Implementation
- [ ] Class implemented with check methods
- [ ] Covers all defined inconsistency patterns

### AC2: Warning Logic
- [ ] Returns correct warnings for mismatched capabilities/services
- [ ] Returns empty list for consistent strategies

### AC3: Factory Integration
- [ ] Factory uses checker when creating strategies
- [ ] Logs warnings to console/logger

### AC4: Testing
- [ ] Unit tests with mock strategies (consistent and inconsistent)
- [ ] Verify warnings are generated correctly

---

## Technical Specifications

### Implementation Details

```python
import logging
from .dependencies import ServiceDependency
from .interfaces import IndexCapability # Assuming this enum exists from previous epics

logger = logging.getLogger(__name__)

class ConsistencyChecker:
    """Checks consistency between capabilities and services"""
    
    def check_indexing_strategy(self, strategy) -> list[str]:
        warnings = []
        produces = strategy.produces()
        requires_services = strategy.requires_services()
        name = strategy.__class__.__name__
        
        if IndexCapability.VECTORS in produces and ServiceDependency.EMBEDDING not in requires_services:
            warnings.append(f"⚠️ {name}: Produces VECTORS but doesn't require EMBEDDING service.")
            
        if IndexCapability.GRAPH in produces and ServiceDependency.GRAPH not in requires_services:
            warnings.append(f"⚠️ {name}: Produces GRAPH but doesn't require GRAPH service.")
            
        return warnings

    def check_retrieval_strategy(self, strategy) -> list[str]:
        warnings = []
        requires_caps = strategy.requires()
        requires_services = strategy.requires_services()
        name = strategy.__class__.__name__
        
        if IndexCapability.VECTORS in requires_caps and ServiceDependency.DATABASE not in requires_services:
             warnings.append(f"⚠️ {name}: Requires VECTORS capability but doesn't require DATABASE service.")
             
        return warnings
```
