# Story 15.2: Add Missing Service Interface Tests

**Story ID:** 15.2  
**Epic:** Epic 15 - Test Coverage Improvements  
**Story Points:** 5  
**Priority:** High  
**Dependencies:** Epic 11 (Service Interfaces)

---

## User Story

**As a** developer  
**I want** comprehensive tests for all service interfaces  
**So that** I can ensure service implementations correctly follow their contracts

---

## Detailed Requirements

### Functional Requirements

1. **IRerankingService Interface Tests**
   - Test interface definition (abstract methods)
   - Test `rerank()` method signature and return type
   - Test that concrete implementations must implement all methods
   - Test error handling for invalid inputs
   - **Pattern to Follow**: Use same structure as `ILLMService`, `IEmbeddingService`, and `IDatabaseService` tests in `tests/unit/services/test_interfaces.py`

2. **IGraphService Interface Tests**
   - Test interface definition (abstract methods)
   - Test node operations (create, read, update, delete)
   - Test relationship operations (create, query, delete)
   - Test graph traversal methods
   - Test transaction support
   - Test error handling
   - **Pattern to Follow**: Similar to database service interface tests

3. **Service Implementation Validation**
   - Verify Neo4j service implements IGraphService correctly (referenced in `test_service_implementations.py` but no dedicated tests)
   - Verify Cohere reranking service implements IRerankingService correctly (found in `test_reranker_service.py` and `test_reranker_selector.py`)
   - Verify cosine similarity reranker implements IRerankingService correctly (`tests/unit/strategies/reranking/test_cosine_reranker.py` exists)
   - **Note**: Some implementation tests exist but interface contract tests are missing

### Non-Functional Requirements

1. **Consistency**
   - Follow same testing patterns as existing service interface tests
   - Use similar test structure to ILLMService, IEmbeddingService, IDatabaseService tests

2. **Coverage**
   - Achieve 100% coverage for interface definitions
   - Cover all abstract methods and their signatures

---

## Acceptance Criteria

### AC1: IRerankingService Tests
- [ ] Test file `tests/unit/services/test_reranking_interface.py` created
- [ ] Test that IRerankingService is abstract and cannot be instantiated
- [ ] Test `rerank()` method signature (query, documents, top_k)
- [ ] Test return type is List[RankedDocument] or similar
- [ ] Test incomplete implementations raise TypeError
- [ ] Minimum 10 test cases for interface

### AC2: IGraphService Tests
- [ ] Test file `tests/unit/services/test_graph_interface.py` created
- [ ] Test that IGraphService is abstract and cannot be instantiated
- [ ] Test node CRUD operation signatures
- [ ] Test relationship operation signatures
- [ ] Test graph traversal method signatures
- [ ] Test transaction method signatures
- [ ] Test incomplete implementations raise TypeError
- [ ] Minimum 15 test cases for interface

### AC3: Implementation Validation Tests
- [ ] Neo4j service tests verify IGraphService implementation
- [ ] Cohere reranker tests verify IRerankingService implementation
- [ ] Cosine reranker tests verify IRerankingService implementation
- [ ] All implementations pass interface contract tests

### AC4: Test Quality
- [ ] All tests pass with 100% success rate
- [ ] Tests use proper mocking for external dependencies
- [ ] Tests follow existing service test patterns
- [ ] Type hints validated with mypy
- [ ] Code quality validated with pylint

---

## Technical Specifications

### File Structure

```
tests/unit/services/
├── test_interfaces.py              # Existing - covers ILLMService, IEmbeddingService, IDatabaseService
│                                   # This file should be used as the template
├── test_reranking_interface.py     # NEW - IRerankingService tests
└── test_graph_interface.py         # NEW - IGraphService tests

# Existing implementation tests (for reference):
tests/unit/strategies/reranking/
├── test_reranker_service.py        # Has some Cohere tests
├── test_reranker_selector.py       # Has Cohere selection tests
└── test_cosine_reranker.py         # Cosine similarity implementation

tests/integration/services/
└── test_service_implementations.py # References Neo4j but no dedicated tests
```

### IRerankingService Test Template

```python
"""Unit tests for IRerankingService interface."""
import pytest
from abc import ABC
from typing import List
from rag_factory.services.interfaces import IRerankingService, RankedDocument

class TestIRerankingServiceInterface:
    """Test suite for IRerankingService interface definition."""
    
    def test_interface_is_abstract(self):
        """Test that IRerankingService cannot be instantiated directly."""
        with pytest.raises(TypeError):
            IRerankingService()
    
    def test_interface_requires_rerank_method(self):
        """Test that concrete class must implement rerank method."""
        class IncompleteReranker(IRerankingService):
            pass
        
        with pytest.raises(TypeError):
            IncompleteReranker()
    
    def test_rerank_method_signature(self):
        """Test rerank method has correct signature."""
        import inspect
        sig = inspect.signature(IRerankingService.rerank)
        params = sig.parameters
        
        assert 'query' in params
        assert 'documents' in params
        assert 'top_k' in params
        assert sig.return_annotation == List[RankedDocument]
    
    def test_minimal_concrete_implementation(self):
        """Test a minimal concrete implementation works."""
        class MinimalReranker(IRerankingService):
            def rerank(self, query: str, documents: List[str], top_k: int) -> List[RankedDocument]:
                return []
        
        reranker = MinimalReranker()
        assert isinstance(reranker, IRerankingService)
        result = reranker.rerank("test", ["doc1"], 5)
        assert isinstance(result, list)
```

### IGraphService Test Template

```python
"""Unit tests for IGraphService interface."""
import pytest
from abc import ABC
from rag_factory.services.interfaces import IGraphService

class TestIGraphServiceInterface:
    """Test suite for IGraphService interface definition."""
    
    def test_interface_is_abstract(self):
        """Test that IGraphService cannot be instantiated directly."""
        with pytest.raises(TypeError):
            IGraphService()
    
    def test_interface_requires_all_methods(self):
        """Test that concrete class must implement all abstract methods."""
        class IncompleteGraphService(IGraphService):
            def create_node(self, **kwargs):
                pass
        
        with pytest.raises(TypeError):
            IncompleteGraphService()
    
    def test_node_operation_signatures(self):
        """Test node operation methods have correct signatures."""
        import inspect
        
        # Test create_node
        sig = inspect.signature(IGraphService.create_node)
        assert 'label' in sig.parameters or 'properties' in sig.parameters
        
        # Test get_node
        sig = inspect.signature(IGraphService.get_node)
        assert 'node_id' in sig.parameters
        
        # Similar for update_node, delete_node
    
    def test_relationship_operation_signatures(self):
        """Test relationship operation methods have correct signatures."""
        import inspect
        
        sig = inspect.signature(IGraphService.create_relationship)
        assert 'from_node' in sig.parameters or 'to_node' in sig.parameters
```

### Testing Strategy

1. **Unit Tests**
   - Test interface definitions in isolation
   - Mock all external dependencies
   - Focus on contract validation

2. **Integration Tests**
   - Verify implementations satisfy interface contracts
   - Test with real service instances (where applicable)
   - Validate error handling

---

## Definition of Done

- [ ] Both new test files created and committed
- [ ] All tests pass (100% success rate)
- [ ] Type checking passes (mypy)
- [ ] Linting passes (pylint)
- [ ] Code review completed
- [ ] PR merged

---

## Notes

- These interfaces already exist in the codebase but lack dedicated interface tests
- Other service interfaces (ILLMService, IEmbeddingService, IDatabaseService) have tests in `test_interfaces.py`
- This story brings IRerankingService and IGraphService testing to the same standard
