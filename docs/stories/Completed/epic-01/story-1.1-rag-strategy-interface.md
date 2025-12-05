# Story 1.1: Design RAG Strategy Interface

**Story ID:** 1.1
**Epic:** Epic 1 - Core Infrastructure & Factory Pattern
**Story Points:** 5
**Priority:** Critical
**Dependencies:** None

---

## User Story

**As a** developer
**I want** a unified interface for all RAG strategies
**So that** different strategies can be used interchangeably and combined

---

## Detailed Requirements

### Functional Requirements

1. **Interface Definition**
   - Define `IRAGStrategy` as an Abstract Base Class (ABC) in Python
   - Include type hints for all methods and parameters
   - Support both synchronous and asynchronous execution modes

2. **Core Methods**
   - `prepare_data(documents: List[Document]) -> PreparedData`: Prepare and chunk documents for retrieval
   - `retrieve(query: str, top_k: int) -> List[Chunk]`: Retrieve relevant chunks based on query
   - `process_query(query: str, context: List[Chunk]) -> str`: Process query with retrieved context
   - `initialize(config: StrategyConfig) -> None`: Initialize strategy with configuration

3. **Configuration Parameters**
   - `chunk_size`: Size of text chunks (default: 512 tokens)
   - `chunk_overlap`: Overlap between chunks (default: 50 tokens)
   - `top_k`: Number of results to retrieve (default: 5)
   - `strategy_name`: Identifier for the strategy
   - `metadata`: Additional strategy-specific parameters

4. **Return Types**
   - `Chunk`: Dataclass containing text, metadata, score, source_id
   - `PreparedData`: Dataclass containing chunks, embeddings, index_metadata
   - `QueryResult`: Dataclass containing answer, chunks_used, metadata, strategy_info

5. **Metadata Structure**
   - Track which strategy was used
   - Record execution time
   - Store confidence scores
   - Include source document references

### Non-Functional Requirements

1. **Performance**
   - Interface methods should add minimal overhead (<1ms)
   - Type checking should be enforced at development time

2. **Maintainability**
   - Clear documentation for each method
   - Examples provided in docstrings
   - Consistent naming conventions

3. **Extensibility**
   - Easy to add new methods without breaking existing implementations
   - Support for optional methods via default implementations

---

## Acceptance Criteria

### AC1: Interface Definition
- [x] `IRAGStrategy` class defined as ABC with `@abstractmethod` decorators
- [x] All methods have complete type hints using `typing` module
- [x] Interface includes both sync and async method signatures

### AC2: Core Methods Implementation
- [x] `prepare_data()` method signature defined with proper input/output types
- [x] `retrieve()` method signature defined with query and top_k parameters
- [x] `process_query()` method signature defined with query and context
- [x] `initialize()` method for configuration injection

### AC3: Configuration Support
- [x] `StrategyConfig` dataclass defined with all required parameters
- [x] Default values specified for common parameters
- [x] Validation logic for configuration parameters

### AC4: Return Types
- [x] `Chunk` dataclass with text, metadata, score, source_id fields
- [x] `PreparedData` dataclass with chunks, embeddings, index_metadata
- [x] `QueryResult` dataclass with answer, chunks_used, metadata, strategy_info

### AC5: Metadata Structure
- [x] Metadata includes strategy identifier (in StrategyConfig)
- [x] Metadata includes execution timestamp (in QueryResult metadata)
- [x] Metadata includes performance metrics (in QueryResult metadata)
- [x] Metadata includes source document references (in Chunk)

### AC6: Documentation
- [x] Each method has comprehensive docstring
- [x] Usage examples provided in module docstring
- [x] Type hints documented for complex types

---

## Technical Specifications

### File Location
`rag_factory/strategies/base.py`

### Dependencies
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
```

### Interface Skeleton
```python
@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]
    score: float
    source_id: str
    chunk_id: str

@dataclass
class StrategyConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    strategy_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class IRAGStrategy(ABC):
    """Abstract base class for RAG strategies."""

    @abstractmethod
    def initialize(self, config: StrategyConfig) -> None:
        """Initialize the strategy with configuration."""
        pass

    @abstractmethod
    def prepare_data(self, documents: List[Dict[str, Any]]) -> Any:
        """Prepare documents for retrieval."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        """Retrieve relevant chunks."""
        pass

    @abstractmethod
    async def aretrieve(self, query: str, top_k: int) -> List[Chunk]:
        """Async retrieve relevant chunks."""
        pass
```

---

## Unit Tests

### Test File Location
`tests/unit/strategies/test_base.py`

### Test Cases

#### TC1.1: Interface Definition Tests
```python
def test_interface_is_abstract():
    """Test that IRAGStrategy cannot be instantiated directly."""
    with pytest.raises(TypeError):
        IRAGStrategy()

def test_interface_requires_all_abstract_methods():
    """Test that concrete class must implement all abstract methods."""
    class IncompleteStrategy(IRAGStrategy):
        def initialize(self, config):
            pass

    with pytest.raises(TypeError):
        IncompleteStrategy()
```

#### TC1.2: Configuration Dataclass Tests
```python
def test_strategy_config_defaults():
    """Test StrategyConfig has correct default values."""
    config = StrategyConfig()
    assert config.chunk_size == 512
    assert config.chunk_overlap == 50
    assert config.top_k == 5
    assert isinstance(config.metadata, dict)

def test_strategy_config_custom_values():
    """Test StrategyConfig accepts custom values."""
    config = StrategyConfig(
        chunk_size=1024,
        top_k=10,
        strategy_name="test_strategy"
    )
    assert config.chunk_size == 1024
    assert config.top_k == 10
    assert config.strategy_name == "test_strategy"

def test_strategy_config_validation():
    """Test StrategyConfig validates parameter ranges."""
    # Should raise error for invalid chunk_size
    with pytest.raises(ValueError):
        StrategyConfig(chunk_size=-1)
```

#### TC1.3: Chunk Dataclass Tests
```python
def test_chunk_creation():
    """Test Chunk can be created with all fields."""
    chunk = Chunk(
        text="Sample text",
        metadata={"key": "value"},
        score=0.95,
        source_id="doc_123",
        chunk_id="chunk_456"
    )
    assert chunk.text == "Sample text"
    assert chunk.score == 0.95

def test_chunk_serialization():
    """Test Chunk can be serialized to dict."""
    chunk = Chunk(
        text="Sample",
        metadata={},
        score=0.8,
        source_id="doc_1",
        chunk_id="chunk_1"
    )
    chunk_dict = asdict(chunk)
    assert isinstance(chunk_dict, dict)
    assert "text" in chunk_dict
```

#### TC1.4: Concrete Implementation Tests
```python
def test_minimal_concrete_implementation():
    """Test a minimal concrete implementation works."""
    class MinimalStrategy(IRAGStrategy):
        def initialize(self, config: StrategyConfig) -> None:
            self.config = config

        def prepare_data(self, documents):
            return []

        def retrieve(self, query: str, top_k: int):
            return []

        async def aretrieve(self, query: str, top_k: int):
            return []

    strategy = MinimalStrategy()
    assert isinstance(strategy, IRAGStrategy)

def test_concrete_implementation_initialize():
    """Test concrete implementation can be initialized."""
    class TestStrategy(IRAGStrategy):
        def initialize(self, config: StrategyConfig):
            self.config = config
        # ... other methods

    strategy = TestStrategy()
    config = StrategyConfig(strategy_name="test")
    strategy.initialize(config)
    assert strategy.config.strategy_name == "test"
```

#### TC1.5: Type Hint Validation Tests
```python
def test_type_hints_present():
    """Test that all methods have proper type hints."""
    import inspect

    sig = inspect.signature(IRAGStrategy.retrieve)
    assert sig.return_annotation == List[Chunk]

def test_async_method_signature():
    """Test async method is properly defined."""
    import inspect
    assert inspect.iscoroutinefunction(IRAGStrategy.aretrieve)
```

---

## Integration Tests

### Test File Location
`tests/integration/strategies/test_base_integration.py`

### Test Scenarios

#### IS1.1: Strategy Lifecycle Test
```python
@pytest.mark.integration
def test_strategy_full_lifecycle():
    """Test complete lifecycle: initialize -> prepare -> retrieve."""
    class DummyStrategy(IRAGStrategy):
        def initialize(self, config: StrategyConfig):
            self.config = config
            self.data_prepared = False

        def prepare_data(self, documents):
            self.data_prepared = True
            return {"status": "prepared"}

        def retrieve(self, query: str, top_k: int):
            if not self.data_prepared:
                raise RuntimeError("Data not prepared")
            return [
                Chunk(
                    text=f"Result {i}",
                    metadata={},
                    score=0.9,
                    source_id=f"doc_{i}",
                    chunk_id=f"chunk_{i}"
                )
                for i in range(top_k)
            ]

        async def aretrieve(self, query: str, top_k: int):
            return self.retrieve(query, top_k)

    # Initialize
    strategy = DummyStrategy()
    config = StrategyConfig(top_k=3)
    strategy.initialize(config)

    # Prepare data
    result = strategy.prepare_data([{"text": "doc1"}])
    assert result["status"] == "prepared"

    # Retrieve
    chunks = strategy.retrieve("test query", top_k=3)
    assert len(chunks) == 3
    assert all(isinstance(c, Chunk) for c in chunks)
```

#### IS1.2: Multiple Strategy Implementation Test
```python
@pytest.mark.integration
def test_multiple_strategies_implement_interface():
    """Test that multiple different strategies can implement the interface."""
    class StrategyA(IRAGStrategy):
        # Implementation A
        pass

    class StrategyB(IRAGStrategy):
        # Implementation B
        pass

    strategies = [StrategyA(), StrategyB()]
    for strategy in strategies:
        assert isinstance(strategy, IRAGStrategy)
```

#### IS1.3: Async Method Integration Test
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_retrieve_works():
    """Test async retrieve method works correctly."""
    class AsyncStrategy(IRAGStrategy):
        def initialize(self, config):
            self.config = config

        def prepare_data(self, documents):
            return None

        def retrieve(self, query: str, top_k: int):
            return []

        async def aretrieve(self, query: str, top_k: int):
            # Simulate async operation
            await asyncio.sleep(0.1)
            return [
                Chunk("text", {}, 0.9, "doc_1", "chunk_1")
            ]

    strategy = AsyncStrategy()
    strategy.initialize(StrategyConfig())
    results = await strategy.aretrieve("query", 5)
    assert len(results) == 1
```

---

## Definition of Done

- [x] All code passes type checking with mypy
- [x] All unit tests pass (100% coverage of base.py)
- [x] All integration tests pass
- [ ] Code reviewed by at least one team member
- [x] Documentation complete with examples
- [ ] No linting errors (flake8, pylint) - need to verify
- [x] Changes committed to feature branch

---

## Testing Checklist

### Unit Testing
- [x] Interface cannot be instantiated
- [x] Abstract methods are enforced
- [x] Dataclasses have correct defaults
- [x] Configuration validation works
- [x] Concrete implementations work
- [x] Type hints are correct

### Integration Testing
- [x] Full strategy lifecycle works
- [x] Multiple strategies can coexist
- [x] Async methods work correctly
- [x] Error handling is appropriate

### Code Quality
- [x] Type hints on all methods
- [x] Docstrings on all public methods
- [x] No TODO comments remaining
- [x] Consistent code style

---

## Notes for Developers

1. **Start with the dataclasses**: Define `Chunk`, `StrategyConfig`, and other data types first
2. **Keep it simple**: Don't over-engineer the interface initially
3. **Think about extension**: Consider what methods future strategies might need
4. **Document thoroughly**: Good documentation now saves time later
5. **Test abstract constraints**: Ensure the ABC enforcement works correctly
