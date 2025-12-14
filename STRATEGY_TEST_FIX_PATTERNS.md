# Quick Reference: Common Test Fix Patterns

## Pattern 1: Add `requires_services()` Method

**When to use:** `TypeError: Can't instantiate abstract class [Strategy] without an implementation for abstract method 'requires_services'`

**Solution:**
```python
def requires_services(self):
    """Declare required services."""
    from rag_factory.services.dependencies import ServiceDependency
    return set()  # For strategies with no dependencies
    
    # OR for strategies with specific dependencies:
    return {ServiceDependency.LLM, ServiceDependency.DATABASE}
```

**Files affected:** Any class inheriting from `IRAGStrategy`

---

## Pattern 2: Update to StrategyDependencies API

**When to use:** `TypeError: [Strategy].__init__() got an unexpected keyword argument 'vector_store_service'`

**Old Pattern (WRONG):**
```python
strategy = Strategy(
    vector_store_service=mock_vector_store,
    llm_service=mock_llm,
    config=config
)
```

**New Pattern (CORRECT):**
```python
from rag_factory.services.dependencies import StrategyDependencies

dependencies = StrategyDependencies(
    llm_service=mock_llm,
    embedding_service=mock_embedding_service,
    database_service=mock_database_service
)

strategy = Strategy(
    config=config.dict() if hasattr(config, 'dict') else config.__dict__,
    dependencies=dependencies
)
```

---

## Pattern 3: Add Missing Abstract Methods

**When to use:** `TypeError: Can't instantiate abstract class [Strategy] without an implementation for abstract method '[method_name]'`

### For `aretrieve()`:
```python
async def aretrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
    """Async retrieve (delegates to sync version for now)."""
    # TODO: Implement true async version
    return self.retrieve(query, top_k)
```

### For `requires_services()`:
See Pattern 1 above.

---

## Pattern 4: Fix Strategy Instantiation

**When to use:** `TypeError: [Strategy].__init__() missing 2 required positional arguments: 'config' and 'dependencies'`

**Old Pattern (WRONG):**
```python
strategy = MyStrategy()
```

**New Pattern (CORRECT):**
```python
from rag_factory.services.dependencies import StrategyDependencies

config = {}  # or actual config dict
dependencies = StrategyDependencies()
strategy = MyStrategy(config, dependencies)
```

---

## Pattern 5: Replace Deprecated `initialize` References

**When to use:** `AttributeError: type object 'IRAGStrategy' has no attribute 'initialize'`

**Old Pattern (WRONG):**
```python
sig = inspect.signature(IRAGStrategy.initialize)
```

**New Pattern (CORRECT):**
```python
# Check __init__ instead
sig = inspect.signature(IRAGStrategy.__init__)

# OR check requires_services
sig = inspect.signature(IRAGStrategy.requires_services)
```

---

## Pattern 6: Fix AttributeError for Missing Service

**When to use:** `AttributeError: '[Strategy]' object has no attribute 'vector_store'`

**Problem:** Code trying to access service that doesn't exist

**Solution:** Use dependency injection pattern
```python
# WRONG:
if hasattr(self.vector_store, 'asearch'):
    return await self.vector_store.asearch(...)

# CORRECT:
if self.deps.database_service:
    if hasattr(self.deps.database_service, 'asearch_similar'):
        return await self.deps.database_service.asearch_similar(...)
```

---

## Pattern 7: Fix Parameter Name Mismatches

**When to use:** `TypeError: [Class].__init__() got an unexpected keyword argument '[param_name]'`

**Example:**
```python
# WRONG:
StrategyDependencies(reranking_service=mock_service)

# CORRECT:
StrategyDependencies(reranker_service=mock_service)
```

**Solution:** Check the actual parameter names in the class definition

---

## Quick Checklist for New Strategy Tests

When creating a new strategy test, ensure:

- [ ] Strategy class has `requires_services()` method
- [ ] Strategy class has `aretrieve()` async method
- [ ] Tests use `StrategyDependencies` for service injection
- [ ] Strategy instantiation includes `config` and `dependencies` arguments
- [ ] No references to deprecated `initialize` method
- [ ] Service access uses `self.deps.[service_name]` pattern
- [ ] All required services are provided in `StrategyDependencies`

---

## Common Service Dependencies

```python
from rag_factory.services.dependencies import ServiceDependency

# Available service types:
ServiceDependency.LLM           # Large Language Model service
ServiceDependency.EMBEDDING     # Embedding service
ServiceDependency.DATABASE      # Database service
ServiceDependency.GRAPH         # Graph database service
ServiceDependency.RERANKER      # Reranking service
```

---

## Example: Complete Test Strategy Class

```python
from rag_factory.strategies.base import IRAGStrategy, Chunk, PreparedData
from rag_factory.services.dependencies import ServiceDependency
from typing import List, Dict, Any

class TestStrategy(IRAGStrategy):
    """Example test strategy with all required methods."""
    
    def requires_services(self):
        """Declare required services."""
        return {ServiceDependency.LLM, ServiceDependency.DATABASE}
    
    def initialize(self, config):
        """Initialize strategy."""
        self.config = config
    
    def prepare_data(self, documents: List[Dict[str, Any]]) -> PreparedData:
        """Prepare data."""
        return PreparedData(chunks=[], embeddings=[], index_metadata={})
    
    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        """Retrieve chunks."""
        return []
    
    async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
        """Async retrieve."""
        return self.retrieve(query, top_k)
    
    def process_query(self, query: str, context: List[Chunk]) -> str:
        """Process query."""
        return "test response"

# Usage in test:
from rag_factory.services.dependencies import StrategyDependencies

def test_my_strategy():
    dependencies = StrategyDependencies(
        llm_service=mock_llm,
        database_service=mock_db
    )
    
    config = {"chunk_size": 512}
    strategy = TestStrategy(config, dependencies)
    
    # Test the strategy...
```

---

## Files to Reference

- **Base Interface:** `rag_factory/strategies/base.py`
- **Dependencies:** `rag_factory/services/dependencies.py`
- **Example Strategy:** `rag_factory/strategies/contextual/strategy.py`
- **Example Tests:** `tests/integration/strategies/test_base_integration.py`

---

**Last Updated:** 2025-12-13  
**Version:** 1.0  
**Status:** Active - Use for all new strategy test development
