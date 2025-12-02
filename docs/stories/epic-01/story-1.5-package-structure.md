# Story 1.5: Setup Package Structure & Distribution

**Story ID:** 1.5
**Epic:** Epic 1 - Core Infrastructure & Factory Pattern
**Story Points:** 5
**Priority:** High
**Dependencies:** Stories 1.1, 1.2, 1.3, 1.4

---

## User Story

**As a** developer
**I want** proper package structure with installability
**So that** the library can be easily imported and distributed

---

## Detailed Requirements

### Functional Requirements

1. **Package Structure**
   - Create proper Python package structure with `__init__.py` files
   - Organize modules into logical subpackages
   - Define clear import paths
   - Follow Python packaging best practices

2. **Installation Setup**
   - Create `pyproject.toml` for modern Python packaging
   - Alternative: Support `setup.py` for backward compatibility
   - Define package metadata (name, version, author, etc.)
   - Specify dependencies and optional dependencies

3. **Dependency Management**
   - Create `requirements.txt` for core dependencies
   - Create `requirements-dev.txt` for development dependencies
   - Define optional dependency groups (e.g., "all", "embedding", "llm")
   - Pin dependency versions appropriately

4. **Versioning**
   - Implement semantic versioning (SemVer)
   - Single source of truth for version number
   - Support version introspection at runtime
   - Document versioning policy

5. **Distribution**
   - Build package for distribution
   - Upload to PyPI test server
   - Verify installation from PyPI test
   - Document release process

6. **Public API Definition**
   - Define what's exported from main `__init__.py`
   - Use `__all__` to control exports
   - Clear import paths for all public components
   - Hide internal implementation details

7. **Smoke Tests**
   - Test that all public APIs can be imported
   - Test package installation in clean environment
   - Test basic functionality after installation

### Non-Functional Requirements

1. **Usability**
   - Simple installation: `pip install rag-factory`
   - Clear import paths: `from rag_factory import RAGFactory`
   - Good documentation for installation

2. **Compatibility**
   - Support Python 3.8+
   - Work on Linux, macOS, Windows
   - Compatible with major Python distributions

3. **Maintainability**
   - Clean package structure
   - Easy to add new modules
   - Clear organization

4. **Distribution**
   - Small package size
   - Fast installation
   - Minimal required dependencies

---

## Acceptance Criteria

### AC1: Package Structure
- [ ] Proper package structure created with `__init__.py` files
- [ ] All modules organized into appropriate subpackages
- [ ] Structure follows Python packaging conventions
- [ ] No circular imports

### AC2: Installation Configuration
- [ ] `pyproject.toml` created with all metadata
- [ ] Package name, version, description defined
- [ ] Dependencies listed correctly
- [ ] Optional dependencies defined in groups

### AC3: Dependencies
- [ ] `requirements.txt` lists core dependencies
- [ ] `requirements-dev.txt` lists dev dependencies
- [ ] Dependency versions specified appropriately
- [ ] No unnecessary dependencies

### AC4: Versioning
- [ ] Version follows semantic versioning
- [ ] Version defined in single location
- [ ] `__version__` accessible at runtime
- [ ] Version command/attribute works

### AC5: Public API
- [ ] Main `__init__.py` exports all public classes
- [ ] Can import: `from rag_factory import RAGFactory, StrategyPipeline`
- [ ] Can import strategies: `from rag_factory.strategies import IRAGStrategy`
- [ ] `__all__` defined for each module

### AC6: Installation
- [ ] Package can be installed with `pip install -e .`
- [ ] Package can be built: `python -m build`
- [ ] Package can be installed from wheel
- [ ] Installation works in virtual environment

### AC7: Distribution
- [ ] Package uploaded to PyPI test server
- [ ] Package can be installed from PyPI test
- [ ] Package metadata correct on PyPI test

### AC8: Smoke Tests
- [ ] All public APIs can be imported
- [ ] Basic instantiation works
- [ ] No import errors

---

## Technical Specifications

### Package Structure
```
rag_factory/
├── __init__.py              # Main exports
├── __version__.py           # Version information
├── strategies/
│   ├── __init__.py          # Strategy exports
│   ├── base.py              # IRAGStrategy interface (from Story 1.1)
│   ├── reranking.py         # (Future: Story 2.1)
│   └── ...
├── services/
│   ├── __init__.py
│   ├── embedding.py         # (Future)
│   └── llm.py               # (Future)
├── repositories/
│   ├── __init__.py
│   └── ...                  # (Future)
├── factory.py               # RAGFactory (from Story 1.2)
├── pipeline.py              # StrategyPipeline (from Story 1.3)
├── config.py                # ConfigManager (from Story 1.4)
└── exceptions.py            # Custom exceptions

tests/
├── __init__.py
├── unit/
│   ├── __init__.py
│   ├── strategies/
│   │   └── test_base.py
│   ├── test_factory.py
│   ├── test_pipeline.py
│   └── test_config.py
└── integration/
    ├── __init__.py
    └── ...

docs/
├── epics/
├── stories/
└── ...

pyproject.toml
requirements.txt
requirements-dev.txt
README.md
LICENSE
.gitignore
```

### pyproject.toml
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "rag-factory"
version = "0.1.0"
description = "A flexible factory system for combining multiple RAG strategies"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["rag", "retrieval", "llm", "ai", "nlp"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
    "flake8>=6.0.0",
    "isort>=5.12.0",
]
watch = [
    "watchdog>=3.0.0",
]
all = [
    "rag-factory[dev,watch]",
]

[project.urls]
Homepage = "https://github.com/yourusername/rag-factory"
Documentation = "https://rag-factory.readthedocs.io"
Repository = "https://github.com/yourusername/rag-factory"
Issues = "https://github.com/yourusername/rag-factory/issues"

[tool.setuptools.packages.find]
where = ["."]
include = ["rag_factory*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-v --cov=rag_factory --cov-report=html --cov-report=term"

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.black]
line-length = 88
target-version = ['py38']

[tool.isort]
profile = "black"
line_length = 88
```

### requirements.txt
```txt
pydantic>=2.0.0,<3.0.0
pyyaml>=6.0,<7.0
```

### requirements-dev.txt
```txt
-r requirements.txt
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
black>=23.0.0
mypy>=1.0.0
flake8>=6.0.0
isort>=5.12.0
watchdog>=3.0.0
```

### rag_factory/__init__.py
```python
"""RAG Factory - A flexible factory system for combining multiple RAG strategies."""

from rag_factory.__version__ import __version__
from rag_factory.factory import RAGFactory, StrategyNotFoundError
from rag_factory.pipeline import StrategyPipeline, PipelineResult, ExecutionMode
from rag_factory.config import ConfigManager, ConfigurationError
from rag_factory.strategies.base import (
    IRAGStrategy,
    Chunk,
    StrategyConfig,
    PreparedData,
)

__all__ = [
    "__version__",
    "RAGFactory",
    "StrategyNotFoundError",
    "StrategyPipeline",
    "PipelineResult",
    "ExecutionMode",
    "ConfigManager",
    "ConfigurationError",
    "IRAGStrategy",
    "Chunk",
    "StrategyConfig",
    "PreparedData",
]
```

### rag_factory/__version__.py
```python
"""Version information for rag-factory."""

__version__ = "0.1.0"
```

### rag_factory/strategies/__init__.py
```python
"""RAG strategy implementations."""

from rag_factory.strategies.base import (
    IRAGStrategy,
    Chunk,
    StrategyConfig,
    PreparedData,
)

__all__ = [
    "IRAGStrategy",
    "Chunk",
    "StrategyConfig",
    "PreparedData",
]
```

---

## Unit Tests

### Test File Location
`tests/unit/test_package.py`

### Test Cases

#### TC5.1: Import Tests
```python
def test_import_main_package():
    """Test main package can be imported."""
    import rag_factory
    assert hasattr(rag_factory, '__version__')

def test_import_factory():
    """Test RAGFactory can be imported."""
    from rag_factory import RAGFactory
    assert RAGFactory is not None

def test_import_pipeline():
    """Test StrategyPipeline can be imported."""
    from rag_factory import StrategyPipeline
    assert StrategyPipeline is not None

def test_import_config():
    """Test ConfigManager can be imported."""
    from rag_factory import ConfigManager
    assert ConfigManager is not None

def test_import_base_strategy():
    """Test IRAGStrategy can be imported."""
    from rag_factory.strategies import IRAGStrategy
    assert IRAGStrategy is not None

def test_import_all_exports():
    """Test all items in __all__ can be imported."""
    import rag_factory

    for name in rag_factory.__all__:
        assert hasattr(rag_factory, name), f"Missing export: {name}"
```

#### TC5.2: Version Tests
```python
def test_version_format():
    """Test version follows semantic versioning."""
    from rag_factory import __version__
    import re

    # Should match X.Y.Z format
    pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$'
    assert re.match(pattern, __version__), f"Invalid version format: {__version__}"

def test_version_accessible():
    """Test version accessible from package."""
    import rag_factory
    assert hasattr(rag_factory, '__version__')
    assert isinstance(rag_factory.__version__, str)
```

#### TC5.3: Package Structure Tests
```python
def test_package_has_required_modules():
    """Test package contains all required modules."""
    import rag_factory

    required_modules = ['factory', 'pipeline', 'config', 'strategies']

    for module_name in required_modules:
        assert hasattr(rag_factory, module_name.capitalize()) or \
               module_name in dir(rag_factory), \
               f"Missing module: {module_name}"

def test_strategies_subpackage_exists():
    """Test strategies subpackage exists and is accessible."""
    from rag_factory import strategies
    assert hasattr(strategies, 'IRAGStrategy')

def test_no_circular_imports():
    """Test importing doesn't cause circular import errors."""
    try:
        from rag_factory import RAGFactory
        from rag_factory import StrategyPipeline
        from rag_factory import ConfigManager
        from rag_factory.strategies import IRAGStrategy
    except ImportError as e:
        pytest.fail(f"Circular import detected: {e}")
```

#### TC5.4: Dependency Tests
```python
def test_required_dependencies_installed():
    """Test all required dependencies are available."""
    required_packages = ['pydantic', 'yaml']

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            pytest.fail(f"Required dependency not installed: {package}")

def test_optional_dependencies_handled():
    """Test package works without optional dependencies."""
    # Should not fail if optional dependencies missing
    from rag_factory import RAGFactory
    assert RAGFactory is not None
```

---

## Integration Tests

### Test File Location
`tests/integration/test_package_integration.py`

### Test Scenarios

#### IS5.1: Installation Tests
```python
@pytest.mark.integration
def test_package_installable(tmp_path):
    """Test package can be installed in clean environment."""
    import subprocess
    import sys

    # Create virtual environment
    venv_dir = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    # Install package in editable mode
    pip_executable = venv_dir / "bin" / "pip"
    subprocess.run([str(pip_executable), "install", "-e", "."], check=True)

    # Test import in the venv
    python_executable = venv_dir / "bin" / "python"
    result = subprocess.run(
        [str(python_executable), "-c", "import rag_factory; print(rag_factory.__version__)"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert len(result.stdout.strip()) > 0
```

#### IS5.2: Smoke Test
```python
@pytest.mark.integration
def test_basic_usage_smoke_test():
    """Test basic usage works after import."""
    from rag_factory import RAGFactory, StrategyPipeline, ConfigManager
    from rag_factory.strategies import IRAGStrategy, Chunk, StrategyConfig

    # Can instantiate basic objects
    factory = RAGFactory()
    assert factory is not None

    pipeline = StrategyPipeline()
    assert pipeline is not None

    config = ConfigManager()
    assert config is not None

    # Can create basic data structures
    chunk = Chunk("text", {}, 0.9, "doc1", "chunk1")
    assert chunk.text == "text"

    strategy_config = StrategyConfig(chunk_size=512)
    assert strategy_config.chunk_size == 512
```

#### IS5.3: Full Workflow Test
```python
@pytest.mark.integration
def test_full_workflow_with_installed_package():
    """Test complete workflow using installed package."""
    from rag_factory import RAGFactory, StrategyPipeline
    from rag_factory.strategies import IRAGStrategy, Chunk

    # Define a test strategy
    class TestStrategy(IRAGStrategy):
        def initialize(self, config):
            self.config = config

        def prepare_data(self, documents):
            return {"prepared": True}

        def retrieve(self, query, top_k):
            return [Chunk(f"Result {i}", {}, 0.9, f"doc{i}", f"chunk{i}")
                    for i in range(top_k)]

        async def aretrieve(self, query, top_k):
            return self.retrieve(query, top_k)

    # Register strategy
    factory = RAGFactory()
    factory.register_strategy("test", TestStrategy)

    # Create strategy
    strategy = factory.create_strategy("test")

    # Use strategy in pipeline
    pipeline = StrategyPipeline()
    pipeline.add_stage(strategy, "test_stage")

    result = pipeline.execute("test query", top_k=3)

    assert len(result.final_results) == 3
```

#### IS5.4: Build and Distribution Test
```python
@pytest.mark.integration
def test_package_can_be_built():
    """Test package can be built for distribution."""
    import subprocess
    import sys

    # Build package
    result = subprocess.run(
        [sys.executable, "-m", "build"],
        capture_output=True,
        text=True
    )

    # Should succeed
    assert result.returncode == 0

    # Check dist directory created
    import os
    assert os.path.exists("dist")

    # Check wheel and sdist created
    dist_files = os.listdir("dist")
    assert any(f.endswith(".whl") for f in dist_files)
    assert any(f.endswith(".tar.gz") for f in dist_files)
```

---

## Manual Testing Checklist

### Installation Testing
- [ ] Clone fresh repository
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Activate venv: `source venv/bin/activate`
- [ ] Install package: `pip install -e .`
- [ ] Verify installation: `python -c "import rag_factory"`
- [ ] Check version: `python -c "import rag_factory; print(rag_factory.__version__)"`

### Import Testing
- [ ] Test main imports: `from rag_factory import RAGFactory, StrategyPipeline, ConfigManager`
- [ ] Test subpackage imports: `from rag_factory.strategies import IRAGStrategy`
- [ ] Test no import errors

### Build Testing
- [ ] Install build tools: `pip install build`
- [ ] Build package: `python -m build`
- [ ] Check dist/ directory created
- [ ] Verify wheel file created: `dist/rag_factory-*.whl`
- [ ] Verify sdist created: `dist/rag_factory-*.tar.gz`

### Distribution Testing (PyPI Test)
- [ ] Install twine: `pip install twine`
- [ ] Upload to test PyPI: `twine upload --repository testpypi dist/*`
- [ ] Create new venv for testing
- [ ] Install from test PyPI: `pip install --index-url https://test.pypi.org/simple/ rag-factory`
- [ ] Test import works

---

## Definition of Done

- [ ] All code passes type checking with mypy
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Package structure follows Python best practices
- [ ] `pyproject.toml` complete and valid
- [ ] All dependencies specified correctly
- [ ] Package can be installed with `pip install -e .`
- [ ] Package can be built successfully
- [ ] Package uploaded to PyPI test server
- [ ] All public APIs importable
- [ ] Smoke tests pass
- [ ] Documentation updated
- [ ] Code reviewed
- [ ] Changes committed to feature branch

---

## Testing Checklist

### Unit Testing
- [ ] All imports work
- [ ] Version format correct
- [ ] Package structure complete
- [ ] No circular imports
- [ ] Dependencies available

### Integration Testing
- [ ] Package installable in clean environment
- [ ] Smoke test passes
- [ ] Full workflow works
- [ ] Package can be built

### Manual Testing
- [ ] Installation in fresh virtualenv works
- [ ] All imports successful
- [ ] Build process completes
- [ ] Upload to test PyPI succeeds
- [ ] Installation from test PyPI works

---

## Notes for Developers

1. **Follow PEP 517/518**: Use modern Python packaging with `pyproject.toml`
2. **Semantic Versioning**: Always follow SemVer (MAJOR.MINOR.PATCH)
3. **Test in clean environment**: Always test installation in fresh virtualenv
4. **Pin dependencies carefully**: Too strict = compatibility issues, too loose = broken builds
5. **Document installation**: Keep installation docs up to date
6. **Test imports**: Ensure all public APIs are easily importable
7. **Keep it minimal**: Don't add unnecessary dependencies

### Recommended Implementation Order
1. Create package directory structure
2. Add `__init__.py` files with exports
3. Create `__version__.py`
4. Create `pyproject.toml`
5. Create requirements files
6. Test local installation (`pip install -e .`)
7. Write import tests
8. Test build process
9. Upload to test PyPI
10. Test installation from test PyPI

### Common Pitfalls to Avoid
- Missing `__init__.py` files
- Circular imports
- Incorrect dependency versions
- Missing metadata in `pyproject.toml`
- Not testing in clean environment
- Forgetting to export public APIs in `__all__`
- Version mismatch between `__version__.py` and `pyproject.toml`

### Release Checklist
1. Run all tests
2. Update version number
3. Update CHANGELOG
4. Commit changes
5. Create git tag
6. Build package: `python -m build`
7. Upload to test PyPI
8. Test installation from test PyPI
9. Upload to production PyPI
10. Create GitHub release
