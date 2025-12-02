# Development Standards
use the proyect vent to run the code

## Methodology

Test-Driven Development (TDD):
1. Write tests first (unit, then integration)
2. Implement code to pass tests
3. Refactor while maintaining test coverage

## Testing Requirements

### Unit Tests
- 100% code coverage required
- Location: `tests/unit/`
- Test file naming: `test_<module>.py`
- All abstract interfaces must be tested for:
  - Cannot be instantiated directly
  - Enforce abstract method implementation
  - Type hints validation

### Integration Tests
- Location: `tests/integration/`
- Marker: `@pytest.mark.integration`
- Test complete lifecycle workflows
- Test multiple implementations coexisting
- Test async operations with `@pytest.mark.asyncio`

### Dataclass Tests
- Test default values
- Test custom values
- Test validation logic
- Test serialization (asdict)

## Code Quality Standards

### Type Checking
```bash
mypy <module>/ --strict
```
- All functions require return type annotations
- All parameters require type hints

### Linting
```bash
flake8 <module>/ --max-line-length=100
pylint <module>/
```
- Target: 10.00/10 pylint score
- No linting errors permitted

### Code Style
- Abstract methods: Use `...` (ellipsis) as body
- Line length: 100 characters max
- Type hints: Required on all public methods

## Project Structure
```
project/
├── <package>/           # Source code
│   └── module/
│       └── file.py
├── tests/
│   ├── unit/
│   │   └── module/
│   │       └── test_file.py
│   └── integration/
│       └── module/
│           └── test_file_integration.py
├── requirements-dev.txt
├── pytest.ini
└── .pylintrc
```

## Definition of Done
- [ ] All tests pass (unit + integration)
- [ ] 100% code coverage
- [ ] mypy --strict passes
- [ ] flake8 passes
- [ ] pylint score 10.00/10
- [ ] Comprehensive docstrings with examples
