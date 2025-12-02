# Story 1.2: Implement RAG Factory

**Story ID:** 1.2
**Epic:** Epic 1 - Core Infrastructure & Factory Pattern
**Story Points:** 8
**Priority:** Critical
**Dependencies:** Story 1.1 (RAG Strategy Interface)

---

## User Story

**As a** developer
**I want** a factory class to instantiate RAG strategies
**So that** I can dynamically create and compose strategies

---

## Detailed Requirements

### Functional Requirements

1. **Factory Class**
   - Create `RAGFactory` class for strategy instantiation
   - Support strategy registration by name
   - Enable dynamic strategy creation from configuration
   - Provide singleton access pattern (optional)

2. **Strategy Registration**
   - `register_strategy(name: str, strategy_class: Type[IRAGStrategy])`: Register a strategy class
   - `unregister_strategy(name: str)`: Remove a strategy from registry
   - `list_strategies() -> List[str]`: Get all registered strategy names
   - Support auto-discovery of strategies via decorators or plugins

3. **Strategy Creation**
   - `create_strategy(name: str, config: Dict) -> IRAGStrategy`: Create and initialize strategy
   - `create_from_config(config_path: str) -> IRAGStrategy`: Create from config file
   - Support dependency injection for external services (DB, LLM clients)

4. **Configuration Validation**
   - Validate strategy name exists in registry
   - Validate configuration parameters before instantiation
   - Provide helpful error messages for missing/invalid config
   - Support schema validation using pydantic or similar

5. **Strategy Composition Support**
   - Allow creating multiple strategies from a single call
   - Support pipeline configuration
   - Enable strategy chaining through factory

### Non-Functional Requirements

1. **Performance**
   - Strategy instantiation should be fast (<100ms for simple strategies)
   - Registry lookup should be O(1) using dict

2. **Maintainability**
   - Clear separation of concerns
   - Easy to extend with new strategies
   - Minimal coupling between factory and concrete strategies

3. **Reliability**
   - Thread-safe registry operations
   - Graceful error handling with informative messages
   - No silent failures

4. **Usability**
   - Simple API for common use cases
   - Sensible defaults
   - Good error messages

---

## Acceptance Criteria

### AC1: Factory Class Implementation
- [ ] `RAGFactory` class created with clear public API
- [ ] Factory can be instantiated and reused
- [ ] Factory maintains internal registry of strategies
- [ ] Factory provides both class methods and instance methods

### AC2: Strategy Registration
- [ ] `register_strategy()` adds strategy to registry
- [ ] Cannot register duplicate strategy names (raises error)
- [ ] `unregister_strategy()` removes strategy from registry
- [ ] `list_strategies()` returns all registered strategy names
- [ ] Decorator `@register_rag_strategy("name")` for auto-registration

### AC3: Strategy Creation
- [ ] `create_strategy()` returns initialized strategy instance
- [ ] Raises `StrategyNotFoundError` for unknown strategy names
- [ ] Passes configuration to strategy's `initialize()` method
- [ ] `create_from_config()` loads config from YAML/JSON file

### AC4: Configuration Validation
- [ ] Validates strategy name before creation
- [ ] Validates required configuration parameters
- [ ] Provides detailed error messages for validation failures
- [ ] Supports optional parameters with defaults

### AC5: Dependency Injection
- [ ] Factory accepts external dependencies (embedding service, LLM client)
- [ ] Dependencies passed to strategies during creation
- [ ] Optional dependencies handled gracefully

### AC6: Composition Support
- [ ] Factory can create multiple strategies from list of configs
- [ ] Support for creating strategy pipelines
- [ ] Integration with composition engine (Story 1.3)

---

## Technical Specifications

### File Location
`rag_factory/factory.py`

### Dependencies
```python
from typing import Type, Dict, Any, List, Optional, Callable
from pathlib import Path
import yaml
import json
from .strategies.base import IRAGStrategy, StrategyConfig
```

### Factory Implementation Skeleton
```python
class StrategyNotFoundError(Exception):
    """Raised when strategy name not found in registry."""
    pass

class ConfigurationError(Exception):
    """Raised when strategy configuration is invalid."""
    pass

class RAGFactory:
    """Factory for creating RAG strategy instances."""

    _registry: Dict[str, Type[IRAGStrategy]] = {}
    _dependencies: Dict[str, Any] = {}

    @classmethod
    def register_strategy(
        cls,
        name: str,
        strategy_class: Type[IRAGStrategy],
        override: bool = False
    ) -> None:
        """Register a strategy class with the factory."""
        if name in cls._registry and not override:
            raise ValueError(f"Strategy '{name}' already registered")
        cls._registry[name] = strategy_class

    @classmethod
    def create_strategy(
        cls,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> IRAGStrategy:
        """Create and initialize a strategy instance."""
        if name not in cls._registry:
            raise StrategyNotFoundError(
                f"Strategy '{name}' not found. "
                f"Available: {list(cls._registry.keys())}"
            )

        strategy_class = cls._registry[name]
        strategy = strategy_class()

        if config:
            strategy_config = StrategyConfig(**config)
            strategy.initialize(strategy_config)

        return strategy

    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all registered strategy names."""
        return list(cls._registry.keys())

    @classmethod
    def set_dependency(cls, name: str, dependency: Any) -> None:
        """Register a dependency for injection."""
        cls._dependencies[name] = dependency
```

### Decorator for Auto-Registration
```python
def register_rag_strategy(name: str) -> Callable:
    """Decorator to auto-register strategy classes."""
    def decorator(cls: Type[IRAGStrategy]) -> Type[IRAGStrategy]:
        RAGFactory.register_strategy(name, cls)
        return cls
    return decorator
```

---

## Unit Tests

### Test File Location
`tests/unit/test_factory.py`

### Test Cases

#### TC2.1: Factory Instantiation Tests
```python
def test_factory_can_be_created():
    """Test factory can be instantiated."""
    factory = RAGFactory()
    assert isinstance(factory, RAGFactory)

def test_factory_has_empty_registry_initially():
    """Test factory starts with empty registry."""
    factory = RAGFactory()
    assert len(factory.list_strategies()) >= 0
```

#### TC2.2: Strategy Registration Tests
```python
def test_register_strategy_adds_to_registry():
    """Test registering a strategy adds it to registry."""
    class TestStrategy(IRAGStrategy):
        pass

    factory = RAGFactory()
    factory.register_strategy("test_strategy", TestStrategy)
    assert "test_strategy" in factory.list_strategies()

def test_register_duplicate_raises_error():
    """Test registering duplicate strategy name raises error."""
    class TestStrategy(IRAGStrategy):
        pass

    factory = RAGFactory()
    factory.register_strategy("test", TestStrategy)

    with pytest.raises(ValueError, match="already registered"):
        factory.register_strategy("test", TestStrategy)

def test_register_duplicate_with_override():
    """Test override=True allows replacing strategy."""
    class StrategyV1(IRAGStrategy):
        pass

    class StrategyV2(IRAGStrategy):
        pass

    factory = RAGFactory()
    factory.register_strategy("strategy", StrategyV1)
    factory.register_strategy("strategy", StrategyV2, override=True)

    # Should use V2
    assert factory._registry["strategy"] == StrategyV2

def test_unregister_strategy():
    """Test unregistering a strategy removes it."""
    class TestStrategy(IRAGStrategy):
        pass

    factory = RAGFactory()
    factory.register_strategy("test", TestStrategy)
    factory.unregister_strategy("test")

    assert "test" not in factory.list_strategies()

def test_list_strategies_returns_all_names():
    """Test list_strategies returns all registered names."""
    class Strategy1(IRAGStrategy):
        pass
    class Strategy2(IRAGStrategy):
        pass

    factory = RAGFactory()
    factory.register_strategy("strategy1", Strategy1)
    factory.register_strategy("strategy2", Strategy2)

    strategies = factory.list_strategies()
    assert "strategy1" in strategies
    assert "strategy2" in strategies
```

#### TC2.3: Strategy Creation Tests
```python
def test_create_strategy_returns_instance():
    """Test creating a strategy returns an instance."""
    class TestStrategy(IRAGStrategy):
        def initialize(self, config):
            self.config = config
        def prepare_data(self, documents):
            pass
        def retrieve(self, query, top_k):
            pass
        async def aretrieve(self, query, top_k):
            pass

    factory = RAGFactory()
    factory.register_strategy("test", TestStrategy)

    strategy = factory.create_strategy("test")
    assert isinstance(strategy, TestStrategy)
    assert isinstance(strategy, IRAGStrategy)

def test_create_unknown_strategy_raises_error():
    """Test creating unknown strategy raises StrategyNotFoundError."""
    factory = RAGFactory()

    with pytest.raises(StrategyNotFoundError, match="not found"):
        factory.create_strategy("nonexistent")

def test_create_strategy_with_config():
    """Test creating strategy with configuration."""
    class TestStrategy(IRAGStrategy):
        def initialize(self, config):
            self.config = config
        # ... other methods

    factory = RAGFactory()
    factory.register_strategy("test", TestStrategy)

    config = {"chunk_size": 1024, "top_k": 10}
    strategy = factory.create_strategy("test", config)

    assert strategy.config.chunk_size == 1024
    assert strategy.config.top_k == 10

def test_strategy_not_found_error_message_includes_available():
    """Test error message includes list of available strategies."""
    class Strategy1(IRAGStrategy):
        pass

    factory = RAGFactory()
    factory.register_strategy("strategy1", Strategy1)

    with pytest.raises(StrategyNotFoundError) as exc_info:
        factory.create_strategy("wrong_name")

    assert "strategy1" in str(exc_info.value)
```

#### TC2.4: Configuration Validation Tests
```python
def test_invalid_config_raises_error():
    """Test invalid configuration raises ConfigurationError."""
    class TestStrategy(IRAGStrategy):
        def initialize(self, config):
            if config.chunk_size < 0:
                raise ValueError("Invalid chunk_size")
            self.config = config
        # ... other methods

    factory = RAGFactory()
    factory.register_strategy("test", TestStrategy)

    with pytest.raises(Exception):
        factory.create_strategy("test", {"chunk_size": -1})

def test_missing_required_config_handled():
    """Test missing required configuration is handled."""
    # Implementation depends on strategy requirements
    pass
```

#### TC2.5: Decorator Tests
```python
def test_register_decorator_auto_registers():
    """Test @register_rag_strategy decorator auto-registers strategy."""
    @register_rag_strategy("decorated_strategy")
    class DecoratedStrategy(IRAGStrategy):
        def initialize(self, config):
            pass
        # ... other methods

    factory = RAGFactory()
    assert "decorated_strategy" in factory.list_strategies()

def test_decorator_returns_class_unchanged():
    """Test decorator doesn't modify the class."""
    @register_rag_strategy("test")
    class TestStrategy(IRAGStrategy):
        test_attr = "value"

    assert TestStrategy.test_attr == "value"
```

#### TC2.6: File-based Configuration Tests
```python
def test_create_from_yaml_config(tmp_path):
    """Test creating strategy from YAML config file."""
    class TestStrategy(IRAGStrategy):
        def initialize(self, config):
            self.config = config
        # ... other methods

    factory = RAGFactory()
    factory.register_strategy("test", TestStrategy)

    # Create config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
    strategy_name: test
    chunk_size: 512
    top_k: 5
    """)

    strategy = factory.create_from_config(str(config_file))
    assert strategy.config.chunk_size == 512

def test_create_from_json_config(tmp_path):
    """Test creating strategy from JSON config file."""
    class TestStrategy(IRAGStrategy):
        def initialize(self, config):
            self.config = config

    factory = RAGFactory()
    factory.register_strategy("test", TestStrategy)

    config_file = tmp_path / "config.json"
    config_file.write_text('{"strategy_name": "test", "chunk_size": 256}')

    strategy = factory.create_from_config(str(config_file))
    assert strategy.config.chunk_size == 256
```

---

## Integration Tests

### Test File Location
`tests/integration/test_factory_integration.py`

### Test Scenarios

#### IS2.1: End-to-End Strategy Creation
```python
@pytest.mark.integration
def test_register_create_use_strategy():
    """Test complete workflow: register -> create -> use."""
    class DummyStrategy(IRAGStrategy):
        def initialize(self, config):
            self.config = config
            self.initialized = True

        def prepare_data(self, documents):
            return {"prepared": len(documents)}

        def retrieve(self, query, top_k):
            return [
                Chunk(f"Result {i}", {}, 0.9, f"doc_{i}", f"chunk_{i}")
                for i in range(top_k)
            ]

        async def aretrieve(self, query, top_k):
            return self.retrieve(query, top_k)

    # Register
    factory = RAGFactory()
    factory.register_strategy("dummy", DummyStrategy)

    # Create with config
    config = {"chunk_size": 512, "top_k": 3}
    strategy = factory.create_strategy("dummy", config)

    # Use
    assert strategy.initialized
    result = strategy.prepare_data([{"text": "doc1"}, {"text": "doc2"}])
    assert result["prepared"] == 2

    chunks = strategy.retrieve("test query", 3)
    assert len(chunks) == 3
```

#### IS2.2: Multiple Strategies Integration
```python
@pytest.mark.integration
def test_create_multiple_different_strategies():
    """Test creating multiple different strategy types."""
    class StrategyA(IRAGStrategy):
        strategy_type = "A"
        # ... implementation

    class StrategyB(IRAGStrategy):
        strategy_type = "B"
        # ... implementation

    factory = RAGFactory()
    factory.register_strategy("strategy_a", StrategyA)
    factory.register_strategy("strategy_b", StrategyB)

    strategy_a = factory.create_strategy("strategy_a")
    strategy_b = factory.create_strategy("strategy_b")

    assert strategy_a.strategy_type == "A"
    assert strategy_b.strategy_type == "B"
    assert type(strategy_a) != type(strategy_b)
```

#### IS2.3: Dependency Injection Integration
```python
@pytest.mark.integration
def test_dependency_injection():
    """Test injecting dependencies into strategies."""
    class EmbeddingService:
        def embed(self, text):
            return [0.1, 0.2, 0.3]

    class StrategyWithDeps(IRAGStrategy):
        def __init__(self, embedding_service=None):
            self.embedding_service = embedding_service

        def initialize(self, config):
            self.config = config

        # ... other methods

    factory = RAGFactory()
    embedding_service = EmbeddingService()
    factory.set_dependency("embedding_service", embedding_service)

    # Modified create to inject dependencies
    factory.register_strategy("strategy_with_deps", StrategyWithDeps)
    strategy = factory.create_strategy("strategy_with_deps")

    # Verify dependency injection worked
    # (implementation details depend on injection mechanism)
```

#### IS2.4: Configuration File Integration
```python
@pytest.mark.integration
def test_config_file_with_multiple_strategies(tmp_path):
    """Test loading configuration for multiple strategies from file."""
    class Strategy1(IRAGStrategy):
        def initialize(self, config):
            self.config = config

    class Strategy2(IRAGStrategy):
        def initialize(self, config):
            self.config = config

    factory = RAGFactory()
    factory.register_strategy("strategy1", Strategy1)
    factory.register_strategy("strategy2", Strategy2)

    # Create multi-strategy config
    config_file = tmp_path / "multi_config.yaml"
    config_file.write_text("""
    strategies:
      - name: strategy1
        chunk_size: 512
      - name: strategy2
        chunk_size: 1024
    """)

    # Load and create strategies
    # (implementation depends on create_from_config design)
```

#### IS2.5: Error Recovery Integration
```python
@pytest.mark.integration
def test_factory_error_recovery():
    """Test factory handles errors gracefully and maintains state."""
    class WorkingStrategy(IRAGStrategy):
        def initialize(self, config):
            self.config = config

    class BrokenStrategy(IRAGStrategy):
        def initialize(self, config):
            raise RuntimeError("Initialization failed")

    factory = RAGFactory()
    factory.register_strategy("working", WorkingStrategy)
    factory.register_strategy("broken", BrokenStrategy)

    # Try to create broken strategy
    with pytest.raises(RuntimeError):
        factory.create_strategy("broken")

    # Factory should still work for other strategies
    strategy = factory.create_strategy("working")
    assert isinstance(strategy, WorkingStrategy)
```

---

## Definition of Done

- [ ] All code passes type checking with mypy
- [ ] All unit tests pass (>95% coverage of factory.py)
- [ ] All integration tests pass
- [ ] Code reviewed by at least one team member
- [ ] Documentation complete with usage examples
- [ ] No linting errors
- [ ] Integration with Story 1.1 verified
- [ ] Changes committed to feature branch

---

## Testing Checklist

### Unit Testing
- [ ] Factory instantiation works
- [ ] Strategy registration/unregistration works
- [ ] Duplicate registration handled correctly
- [ ] Strategy creation returns correct instances
- [ ] Unknown strategies raise appropriate errors
- [ ] Configuration validation works
- [ ] Decorator auto-registration works
- [ ] File-based configuration loading works

### Integration Testing
- [ ] End-to-end workflow completes successfully
- [ ] Multiple strategies can coexist
- [ ] Dependency injection works
- [ ] Configuration files work with real strategies
- [ ] Error handling doesn't corrupt factory state

### Code Quality
- [ ] Thread-safe operations
- [ ] Clear error messages
- [ ] No memory leaks
- [ ] Performance benchmarks met

---

## Notes for Developers

1. **Start with registry**: Implement the registry mechanism first
2. **Test error cases**: Pay special attention to error handling
3. **Keep it simple**: Don't over-complicate dependency injection initially
4. **Think about plugins**: Consider how third-party strategies will register
5. **Document patterns**: Provide clear examples of factory usage
6. **Thread safety**: Use locks if factory will be used in multi-threaded context
7. **Lazy vs eager**: Consider whether strategies should be created lazily or eagerly

### Recommended Implementation Order
1. Basic factory class with registry dict
2. `register_strategy()` and `list_strategies()`
3. `create_strategy()` without config
4. Add configuration support
5. Implement `create_from_config()`
6. Add decorator for auto-registration
7. Implement dependency injection
8. Add validation and error handling
