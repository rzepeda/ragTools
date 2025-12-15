# Story 15.3: Add CLI Command Tests

**Story ID:** 15.3  
**Epic:** Epic 15 - Test Coverage Improvements  
**Story Points:** 8  
**Priority:** High  
**Dependencies:** Epic 8.5 (CLI), Epic 14 (CLI Validation)

---

## User Story

**As a** developer  
**I want** comprehensive tests for all CLI commands  
**So that** I can ensure the CLI behaves correctly and provides a good user experience

---

## Detailed Requirements

### Functional Requirements

> [!NOTE]
> **Current CLI Coverage**: 33% (4 out of 12 commands tested)
> **Target Coverage**: 80%+
> **Existing Tests**: `validate-pipeline`, `check-consistency` commands have comprehensive tests that can serve as templates

1. **Index Command Tests**
   - Test argument parsing (--strategy, --documents, --config)
   - Test successful indexing execution
   - Test error handling (invalid strategy, missing documents)
   - Test exit codes (0 for success, non-zero for failure)
   - Test output formatting

2. **Query Command Tests**
   - Test argument parsing (--query, --strategy, --top-k, --config)
   - Test successful query execution
   - Test error handling (invalid strategy, empty query)
   - Test exit codes
   - Test output formatting (results display)

3. **List Strategies Command Tests**
   - Test listing all available strategies
   - Test filtering by type (indexing/retrieval)
   - Test output formatting (table or list)
   - Test with no strategies registered

4. **Config Validation Command Tests**
   - Test valid configuration validation
   - Test invalid configuration detection
   - Test detailed error messages
   - Test exit codes

5. **Benchmark Command Tests**
   - Test benchmark execution with dataset
   - Test metric collection and reporting
   - Test output formatting (tables, charts)
   - Test error handling

6. **Interactive REPL Mode Tests**
   - Test REPL startup
   - Test command execution in REPL
   - Test help command in REPL
   - Test exit command
   - **Note**: No existing REPL tests found - this is completely new functionality

### Non-Functional Requirements

1. **Test Isolation**
   - Each test should be independent
   - Use mocks to avoid actual indexing/querying
   - Clean up any created files/state

2. **Coverage**
   - Achieve 80%+ coverage for CLI commands
   - Cover happy path and error cases
   - Test all command-line arguments

---

## Acceptance Criteria

### AC1: Index Command Tests
- [ ] Test file `tests/unit/cli/test_index_command.py` created
- [ ] Test argument parsing with valid inputs
- [ ] Test successful indexing flow (mocked)
- [ ] Test error handling for invalid strategy names
- [ ] Test error handling for missing document paths
- [ ] Test exit code 0 for success
- [ ] Test exit code non-zero for failures
- [ ] Integration test with real factory (in `tests/integration/cli/`)
- [ ] Minimum 8 test cases

### AC2: Query Command Tests
- [ ] Test file `tests/unit/cli/test_query_command.py` created
- [ ] Test argument parsing with valid inputs
- [ ] Test successful query flow (mocked)
- [ ] Test error handling for invalid strategy names
- [ ] Test error handling for empty queries
- [ ] Test top-k parameter validation
- [ ] Test exit codes
- [ ] Test result formatting (console output)
- [ ] Integration test with real factory
- [ ] Minimum 10 test cases

### AC3: List Strategies Command Tests
- [ ] Test file `tests/unit/cli/test_list_strategies_command.py` created
- [ ] Test listing all strategies
- [ ] Test filtering by indexing strategies
- [ ] Test filtering by retrieval strategies
- [ ] Test output formatting (table format)
- [ ] Test with empty strategy registry
- [ ] Minimum 6 test cases

### AC4: Config Validation Command Tests
- [ ] Test file `tests/unit/cli/test_config_validation_command.py` created
- [ ] Test validation of valid config files
- [ ] Test detection of invalid config files
- [ ] Test detailed error messages for config errors
- [ ] Test exit codes (0 for valid, non-zero for invalid)
- [ ] Test with missing config file
- [ ] Minimum 8 test cases

### AC5: Benchmark Command Tests
- [ ] Test file `tests/unit/cli/test_benchmark_command.py` created
- [ ] Test benchmark execution with dataset
- [ ] Test metric collection (latency, accuracy, cost)
- [ ] Test output formatting (tables)
- [ ] Test error handling for missing datasets
- [ ] Test exit codes
- [ ] Minimum 6 test cases

### AC6: Interactive REPL Tests
- [ ] Test file `tests/unit/cli/test_repl_command.py` created
- [ ] Test REPL startup and initialization
- [ ] Test command execution within REPL
- [ ] Test help command display
- [ ] Test exit/quit commands
- [ ] Test error handling in REPL
- [ ] Minimum 8 test cases

### AC7: Integration Tests
- [ ] Integration test for index → query workflow
- [ ] Integration test for config validation → index workflow
- [ ] Integration test for benchmark execution
- [ ] All integration tests pass

### AC8: Test Quality
- [ ] All tests pass (100% success rate)
- [ ] CLI coverage increases from 33% to 80%+
- [ ] Tests use `click.testing.CliRunner` for CLI testing
- [ ] Proper mocking of factory and services
- [ ] Type hints validated
- [ ] Linting passes

---

## Technical Specifications

### File Structure

```
tests/unit/cli/
├── test_check_consistency_command.py    # Existing - ✅ Use as template
├── test_consistency_formatter.py        # Existing - ✅ Formatter pattern
├── test_formatters.py                   # Existing - ✅ Output formatting
├── test_validate_pipeline_command.py    # Existing - ✅ Use as template
├── test_validation_formatter.py         # Existing - ✅ Formatter pattern
├── test_validation.py                   # Existing
├── test_index_command.py                # NEW - Currently only integration test exists
├── test_query_command.py                # NEW - Currently only integration test exists
├── test_list_strategies_command.py      # NEW - No existing tests
├── test_config_validation_command.py    # NEW - No existing tests
├── test_benchmark_command.py            # NEW - No existing tests
└── test_repl_command.py                 # NEW - No existing tests

tests/integration/cli/
├── test_index_query_flow.py             # Existing - ⚠️ Has index/query integration tests
├── test_benchmark_integration.py        # NEW
└── test_config_validation_integration.py # NEW
```

> [!IMPORTANT]
> **Existing Test Patterns**: The `test_validate_pipeline_command.py` and `test_check_consistency_command.py` files provide excellent templates for:
> - Using `click.testing.CliRunner`
> - Mocking `RAGFactory` and services
> - Testing exit codes
> - Verifying output formatting

### Test Template (Index Command)

```python
"""Unit tests for index CLI command."""
import pytest
from click.testing import CliRunner
from unittest.mock import Mock, patch
from rag_factory.cli.commands.index import index_command

class TestIndexCommand:
    """Test suite for index command."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()
    
    def test_index_with_valid_arguments(self, runner):
        """Test index command with valid arguments."""
        with patch('rag_factory.cli.commands.index.RAGFactory') as mock_factory:
            mock_factory.return_value.create_indexing_pipeline.return_value = Mock()
            
            result = runner.invoke(index_command, [
                '--strategy', 'context_aware_chunking',
                '--documents', 'test_docs/',
                '--config', 'config.yaml'
            ])
            
            assert result.exit_code == 0
            assert 'Successfully indexed' in result.output
    
    def test_index_with_invalid_strategy(self, runner):
        """Test index command with invalid strategy name."""
        result = runner.invoke(index_command, [
            '--strategy', 'nonexistent_strategy',
            '--documents', 'test_docs/'
        ])
        
        assert result.exit_code != 0
        assert 'Invalid strategy' in result.output or 'Error' in result.output
    
    def test_index_with_missing_documents(self, runner):
        """Test index command with missing document path."""
        result = runner.invoke(index_command, [
            '--strategy', 'context_aware_chunking',
            '--documents', 'nonexistent_path/'
        ])
        
        assert result.exit_code != 0
        assert 'not found' in result.output.lower() or 'error' in result.output.lower()
```

### Testing Strategy

1. **Unit Tests**
   - Use `click.testing.CliRunner` to invoke commands
   - Mock `RAGFactory` and all service dependencies
   - Test argument parsing and validation
   - Test output formatting
   - Test exit codes

2. **Integration Tests**
   - Use real `RAGFactory` with test configuration
   - Test full command workflows
   - Verify actual file operations (with cleanup)
   - Test command chaining

3. **Mocking Strategy**
   - Mock factory creation to avoid service initialization
   - Mock indexing/retrieval operations
   - Mock file I/O where appropriate
   - Use temporary directories for file operations

---

## Definition of Done

- [ ] All 6 new unit test files created
- [ ] All 2 new integration test files created
- [ ] All tests pass (100% success rate)
- [ ] CLI coverage reaches 80%+ (from 33%)
- [ ] Type checking passes
- [ ] Linting passes
- [ ] Code review completed
- [ ] PR merged

---

## Notes

- Current CLI coverage is only 33% (4 out of 12 commands tested)
- Existing tests for `validate-pipeline` and `check-consistency` can serve as templates
- Use `click.testing.CliRunner` for all CLI tests (consistent with existing tests)
- This is a high-priority story because CLI is a primary user interface
