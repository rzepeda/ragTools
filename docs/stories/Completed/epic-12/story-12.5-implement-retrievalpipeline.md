# Story 12.5: Implement RetrievalPipeline

**Story ID:** 12.5
**Epic:** Epic 12 - Indexing/Retrieval Pipeline Separation & Capability System
**Story Points:** 8
**Priority:** High
**Dependencies:** Story 12.3

---

## User Story

**As a** developer
**I want** a pipeline that executes retrieval strategies in sequence
**So that** multiple retrieval strategies can be chained

---

## Detailed Requirements

### Functional Requirements

1.  **RetrievalPipeline Class**
    -   Create `RetrievalPipeline` class
    -   Initialize with list of `IRetrievalStrategy` and `RetrievalContext`
    -   Execute strategies in sequence (chaining)
    -   Pass context between strategies

2.  **Requirement Aggregation**
    -   Implement `get_requirements()` method
    -   Return union of all strategy capability requirements
    -   Implement `get_service_requirements()` method
    -   Return union of all strategy service requirements

3.  **Execution Logic**
    -   Implement `retrieve()` method
    -   Pass query/results through each strategy
    -   Allow strategies to refine/rerank results
    -   Track execution time and metrics
    -   Handle errors gracefully

### Non-Functional Requirements

1.  **Performance**
    -   Efficient execution of strategy chain
    -   Minimal overhead from pipeline logic

2.  **Observability**
    -   Log pipeline execution steps
    -   Track metrics for each strategy

---

## Acceptance Criteria

### AC1: RetrievalPipeline Implementation
- [ ] `RetrievalPipeline` class implemented
- [ ] Initializes with strategies and context
- [ ] `get_requirements` returns correct set
- [ ] `get_service_requirements` returns correct set

### AC2: Execution Logic
- [ ] `retrieve` method executes strategies in order
- [ ] Results are correctly passed/refined
- [ ] Final results match expected output

### AC3: Error Handling
- [ ] Pipeline handles strategy failures appropriately
- [ ] Errors are logged with context

### AC4: Testing
- [ ] Unit tests for pipeline with multiple mock strategies
- [ ] Test requirement aggregation
- [ ] Test result chaining
- [ ] Test error handling scenarios

---

## Technical Specifications

### File Structure
```
rag_factory/
├── core/
│   ├── pipeline.py          # Existing file (add RetrievalPipeline)
│   └── ...
tests/
├── unit/
│   ├── core/
│   │   └── test_pipeline.py # Add tests for RetrievalPipeline
│   └── ...
```

### Code Definition

```python
from typing import List, Set, Optional
from .capabilities import IndexCapability
from .retrieval_interface import IRetrievalStrategy, RetrievalContext

class RetrievalPipeline:
    """Pipeline for executing retrieval strategies"""
    
    def __init__(
        self,
        strategies: List[IRetrievalStrategy],
        context: RetrievalContext
    ):
        """
        Create retrieval pipeline.
        
        Args:
            strategies: Ordered list of retrieval strategies
            context: Shared retrieval context
        """
        self.strategies = strategies
        self.context = context
    
    def get_requirements(self) -> Set[IndexCapability]:
        """
        Get combined requirements from all strategies.
        
        Returns:
            Union of all strategy requirements
        """
        all_reqs = set()
        for strategy in self.strategies:
            all_reqs.update(strategy.requires())
        return all_reqs
    
    def get_service_requirements(self) -> Set['ServiceDependency']:
        """
        Get combined service requirements from all strategies.
        
        Returns:
            Union of all service requirements
        """
        all_reqs = set()
        for strategy in self.strategies:
            all_reqs.update(strategy.requires_services())
        return all_reqs
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List['Chunk']:
        """
        Execute retrieval pipeline.
        
        Args:
            query: User query
            top_k: Number of results
            
        Returns:
            Retrieved chunks after all strategies applied
        """
        current_query = query
        results = None
        
        for strategy in self.strategies:
            # Each strategy processes the query and/or refines results
            # Note: In a real implementation, we might need to handle 
            # how strategies consume previous results vs just the query.
            # For this interface, we assume strategies have access to 
            # context or we might need to adjust the interface to pass 
            # previous results explicitly if needed.
            # Based on the epic, it seems strategies might just use context/query
            # but let's assume standard retrieval for now.
            results = await strategy.retrieve(current_query, self.context, top_k)
        
        return results
```
