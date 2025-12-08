# Story 12.4: Implement IndexingPipeline

**Story ID:** 12.4
**Epic:** Epic 12 - Indexing/Retrieval Pipeline Separation & Capability System
**Story Points:** 13
**Priority:** High
**Dependencies:** Story 12.2

---

## User Story

**As a** developer
**I want** a pipeline that executes indexing strategies in sequence
**So that** multiple indexing strategies can be combined

---

## Detailed Requirements

### Functional Requirements

1.  **IndexingPipeline Class**
    -   Create `IndexingPipeline` class
    -   Initialize with list of `IIndexingStrategy` and `IndexingContext`
    -   Execute strategies in sequence
    -   Pass context between strategies

2.  **Capability Aggregation**
    -   Implement `get_capabilities()` method
    -   Return union of all strategy capabilities
    -   Support checking capabilities before execution

3.  **Execution Logic**
    -   Implement `index()` method
    -   Process documents through each strategy
    -   Aggregate results (capabilities, metadata, counts)
    -   Track execution time and metrics
    -   Handle errors gracefully (stop pipeline or continue based on config)

### Non-Functional Requirements

1.  **Performance**
    -   Efficient execution of strategies
    -   Minimal overhead from pipeline logic

2.  **Observability**
    -   Log pipeline execution steps
    -   Track metrics for each strategy

---

## Acceptance Criteria

### AC1: IndexingPipeline Implementation
- [ ] `IndexingPipeline` class implemented
- [ ] Initializes with strategies and context
- [ ] `get_capabilities` returns correct set

### AC2: Execution Logic
- [ ] `index` method executes strategies in order
- [ ] Results are correctly aggregated
- [ ] Metadata from all strategies is preserved

### AC3: Error Handling
- [ ] Pipeline handles strategy failures appropriately
- [ ] Errors are logged with context

### AC4: Testing
- [ ] Unit tests for pipeline with multiple mock strategies
- [ ] Test capability aggregation
- [ ] Test result aggregation
- [ ] Test error handling scenarios

---

## Technical Specifications

### File Structure
```
rag_factory/
├── core/
│   ├── pipeline.py          # New file for pipelines
│   └── ...
tests/
├── unit/
│   ├── core/
│   │   └── test_pipeline.py
│   └── ...
```

### Code Definition

```python
from typing import List, Set, Optional
from .capabilities import IndexCapability, IndexingResult
from .indexing_interface import IIndexingStrategy, IndexingContext

class IndexingPipeline:
    """Pipeline for executing indexing strategies"""
    
    def __init__(
        self,
        strategies: List[IIndexingStrategy],
        context: IndexingContext
    ):
        """
        Create indexing pipeline.
        
        Args:
            strategies: Ordered list of indexing strategies
            context: Shared indexing context
        """
        self.strategies = strategies
        self.context = context
        self._last_result: Optional[IndexingResult] = None
    
    def get_capabilities(self) -> Set[IndexCapability]:
        """
        Get combined capabilities from all strategies.
        
        Returns:
            Union of all strategy capabilities
        """
        if self._last_result:
            return self._last_result.capabilities
        
        # Return declared capabilities if not executed yet
        all_caps = set()
        for strategy in self.strategies:
            all_caps.update(strategy.produces())
        return all_caps
    
    async def index(
        self,
        documents: List['Document']
    ) -> IndexingResult:
        """
        Execute indexing pipeline.
        
        Args:
            documents: Documents to index
            
        Returns:
            Combined IndexingResult from all strategies
        """
        all_capabilities = set()
        all_metadata = {}
        total_chunks = 0
        
        for strategy in self.strategies:
            strategy_name = strategy.__class__.__name__
            
            # Execute strategy
            result = await strategy.process(documents, self.context)
            
            # Aggregate results
            all_capabilities.update(result.capabilities)
            all_metadata[strategy_name] = result.metadata
            total_chunks = max(total_chunks, result.chunk_count)
        
        self._last_result = IndexingResult(
            capabilities=all_capabilities,
            metadata=all_metadata,
            document_count=len(documents),
            chunk_count=total_chunks
        )
        
        return self._last_result
```
