# Story 8.5.1: Build CLI for Strategy Testing

**Epic:** 8.5 - Development Tools (CLI & Dev Server)
**Story Points:** 8
**Priority:** Medium
**Dependencies:** Epic 4 (RAG Strategies)

---

## User Story

**As a** developer
**I want** a command-line interface for testing strategies
**So that** I can quickly experiment without writing code

---

## Business Value

- Reduces development time for strategy testing
- Validates the library's public API from a client perspective
- Enables rapid experimentation and benchmarking
- Provides educational tool for learning RAG strategies
- Facilitates debugging without writing integration code

---

## Requirements

### Functional Requirements

#### FR-1: Document Indexing Command
- **Requirement:** CLI must provide a command to index documents with configurable strategies
- **Input:** File path or directory path, strategy name, optional configuration
- **Output:** Progress indication, success/error messages, indexing statistics
- **Command Format:** `rag-factory index --path <path> --strategy <strategy_name> [--config <config_file>]`

#### FR-2: Query Command
- **Requirement:** CLI must provide a command to query indexed documents
- **Input:** Query string, list of strategies to apply, optional parameters
- **Output:** Query results with relevance scores, metadata, timing information
- **Command Format:** `rag-factory query "<query>" --strategies <strategy1,strategy2> [--top-k <n>] [--config <config_file>]`

#### FR-3: Strategy Listing Command
- **Requirement:** CLI must list all available strategies with descriptions
- **Input:** Optional filters (e.g., by type: chunking, retrieval, augmentation)
- **Output:** Formatted list of strategies with names, types, and descriptions
- **Command Format:** `rag-factory strategies --list [--type <type>]`

#### FR-4: Configuration Validation Command
- **Requirement:** CLI must validate configuration files for correctness
- **Input:** Path to YAML/JSON configuration file
- **Output:** Validation results with specific errors if any
- **Command Format:** `rag-factory config --validate <config_file>`

#### FR-5: Benchmark Command
- **Requirement:** CLI must run benchmarks using a test dataset
- **Input:** Path to benchmark dataset (JSON), strategies to test
- **Output:** Benchmark results including latency, accuracy metrics, comparison table
- **Command Format:** `rag-factory benchmark --dataset <dataset_file> [--strategies <strategies>] [--output <report_file>]`

#### FR-6: Interactive REPL Mode
- **Requirement:** CLI must provide an interactive REPL for exploration
- **Input:** Commands entered interactively
- **Output:** Interactive session with command history, auto-completion
- **Command Format:** `rag-factory repl [--config <config_file>]`

### Non-Functional Requirements

#### NFR-1: Usability
- Colorized output using Rich library
- Progress bars for long-running operations (tqdm or Rich progress)
- Clear error messages with suggestions
- Help text for all commands and options
- Examples in help output

#### NFR-2: Performance
- Streaming output for large operations
- Efficient memory usage for document processing
- Cancellation support (Ctrl+C handling)

#### NFR-3: Compatibility
- Python 3.8+ support
- Cross-platform (Linux, macOS, Windows)
- Works with all strategies from Epic 4

#### NFR-4: Architecture
- Uses library's public API only (validates API design)
- No direct access to internal implementation
- Delegates all business logic to the library

---

## Acceptance Criteria

### AC-1: Index Command Functionality
- **Given** a directory with documents
- **When** I run `rag-factory index --path ./docs --strategy context_aware_chunking`
- **Then** the CLI should:
  - Display progress bar showing indexing progress
  - Show number of documents processed
  - Display indexing time and statistics
  - Store indexed data in configured location
  - Handle errors gracefully with clear messages

### AC-2: Query Command Functionality
- **Given** indexed documents exist
- **When** I run `rag-factory query "What are action items?" --strategies reranking,query_expansion`
- **Then** the CLI should:
  - Execute query using specified strategies
  - Display results in readable format with relevance scores
  - Show timing information for each strategy
  - Return top-k results (default 5, configurable)
  - Handle missing index with clear error message

### AC-3: Strategy Listing
- **Given** the library has strategies installed
- **When** I run `rag-factory strategies --list`
- **Then** the CLI should:
  - Display all available strategies grouped by type
  - Show strategy names, types, and descriptions
  - Use formatted, colorized output
  - Support filtering by type

### AC-4: Configuration Validation
- **Given** a configuration file
- **When** I run `rag-factory config --validate config.yaml`
- **Then** the CLI should:
  - Parse and validate the configuration
  - Report all validation errors with line numbers
  - Suggest corrections for common mistakes
  - Return exit code 0 for valid config, non-zero for invalid

### AC-5: Benchmark Execution
- **Given** a benchmark dataset file
- **When** I run `rag-factory benchmark --dataset test.json`
- **Then** the CLI should:
  - Execute all queries in the dataset
  - Measure latency and accuracy metrics
  - Generate comparison table across strategies
  - Support outputting results to file (JSON/CSV)
  - Display summary statistics

### AC-6: Interactive REPL Mode
- **Given** the CLI is started in REPL mode
- **When** I run `rag-factory repl`
- **Then** the CLI should:
  - Provide interactive prompt
  - Support command history (up/down arrows)
  - Provide auto-completion for commands
  - Allow switching configurations interactively
  - Support exit/quit commands

### AC-7: Help and Documentation
- **Given** any command
- **When** I run with `--help` flag
- **Then** the CLI should:
  - Display comprehensive help text
  - Show usage examples
  - List all available options
  - Provide clear descriptions

### AC-8: Error Handling
- **Given** invalid input
- **When** I run any command with invalid parameters
- **Then** the CLI should:
  - Display clear error message
  - Suggest correct usage
  - Return appropriate exit code
  - Not crash or show stack traces (unless --debug flag)

---

## Technical Implementation

### Technology Stack

- **CLI Framework:** Typer (or Click) - for command parsing and routing
- **Output Formatting:** Rich - for colorized, formatted output
- **Progress Indicators:** Rich Progress - for progress bars
- **Configuration:** Pydantic - for configuration validation
- **Testing:** Pytest - for unit and integration tests

### Architecture

```
rag_factory_cli/
├── __init__.py
├── main.py              # Entry point, Typer app setup
├── commands/
│   ├── __init__.py
│   ├── index.py         # Index command implementation
│   ├── query.py         # Query command implementation
│   ├── strategies.py    # Strategy listing command
│   ├── config.py        # Configuration validation
│   ├── benchmark.py     # Benchmark command
│   └── repl.py          # REPL mode implementation
├── formatters/
│   ├── __init__.py
│   ├── output.py        # Rich output formatting utilities
│   └── results.py       # Result display formatting
└── utils/
    ├── __init__.py
    ├── progress.py      # Progress bar utilities
    └── validation.py    # Input validation utilities
```

### Module Specifications

#### main.py
```python
import typer
from rag_factory_cli.commands import index, query, strategies, config, benchmark, repl

app = typer.Typer(
    name="rag-factory",
    help="RAG Factory CLI - Development tool for testing strategies",
    add_completion=True,
)

app.command()(index.index_command)
app.command()(query.query_command)
app.command()(strategies.list_strategies)
app.command()(config.validate_config)
app.command()(benchmark.run_benchmark)
app.command()(repl.start_repl)

if __name__ == "__main__":
    app()
```

#### commands/index.py
- Function: `index_command(path: str, strategy: str, config: Optional[str])`
- Responsibilities:
  - Validate input path exists
  - Load configuration if provided
  - Initialize strategy from library
  - Process documents with progress bar
  - Display statistics on completion

#### commands/query.py
- Function: `query_command(query: str, strategies: str, top_k: int, config: Optional[str])`
- Responsibilities:
  - Parse comma-separated strategies
  - Load configuration if provided
  - Execute query using library
  - Format and display results
  - Show timing information

#### commands/repl.py
- Function: `start_repl(config: Optional[str])`
- Responsibilities:
  - Initialize interactive session
  - Setup command history and auto-completion
  - Parse and execute commands
  - Maintain session state

---

## Testing Requirements

### Unit Tests

#### UT-1: Command Parsing Tests
**Test File:** `tests/unit/cli/test_command_parsing.py`

```python
def test_index_command_parses_arguments():
    """Test that index command correctly parses arguments"""
    # Given: Valid command arguments
    # When: Arguments are parsed
    # Then: All parameters are correctly extracted
    pass

def test_query_command_parses_strategies():
    """Test that query command parses comma-separated strategies"""
    # Given: Strategies string "reranking,query_expansion"
    # When: String is parsed
    # Then: List contains ["reranking", "query_expansion"]
    pass

def test_invalid_strategy_raises_error():
    """Test that invalid strategy name raises clear error"""
    # Given: Invalid strategy name
    # When: Command is executed
    # Then: Error message suggests valid strategies
    pass
```

#### UT-2: Output Formatting Tests
**Test File:** `tests/unit/cli/test_formatters.py`

```python
def test_format_query_results():
    """Test query results are formatted correctly"""
    # Given: Mock query results
    # When: Results are formatted
    # Then: Output includes scores, metadata, and is readable
    pass

def test_format_strategy_list():
    """Test strategy list is formatted with colors and grouping"""
    # Given: List of strategies
    # When: List is formatted
    # Then: Strategies are grouped by type with descriptions
    pass

def test_format_benchmark_results():
    """Test benchmark results display comparison table"""
    # Given: Benchmark results for multiple strategies
    # When: Results are formatted
    # Then: Table shows metrics comparison
    pass
```

#### UT-3: Validation Tests
**Test File:** `tests/unit/cli/test_validation.py`

```python
def test_validate_path_exists():
    """Test path validation for index command"""
    # Given: Non-existent path
    # When: Path is validated
    # Then: ValidationError is raised with helpful message
    pass

def test_validate_config_file():
    """Test configuration file validation"""
    # Given: Invalid YAML config
    # When: Config is validated
    # Then: Specific errors are returned with line numbers
    pass

def test_validate_strategy_names():
    """Test strategy name validation"""
    # Given: Unknown strategy name
    # When: Strategy is validated
    # Then: Error suggests similar valid strategies
    pass
```

#### UT-4: Progress Indicator Tests
**Test File:** `tests/unit/cli/test_progress.py`

```python
def test_progress_bar_updates():
    """Test progress bar updates correctly"""
    # Given: Operation with 100 items
    # When: Progress is updated 10 times
    # Then: Progress bar shows correct percentage
    pass

def test_progress_bar_completion():
    """Test progress bar completes at 100%"""
    # Given: Operation in progress
    # When: Final item is processed
    # Then: Progress shows 100% complete
    pass
```

### Integration Tests

#### IT-1: End-to-End Index and Query
**Test File:** `tests/integration/cli/test_index_query_flow.py`

```python
def test_full_index_and_query_workflow(tmp_path):
    """Test complete workflow from indexing to querying"""
    # Given: Sample documents in temp directory
    # When: Documents are indexed and then queried
    # Then: Query returns relevant results
    # Setup:
    #   1. Create test documents
    #   2. Run index command
    #   3. Verify index created
    #   4. Run query command
    #   5. Verify results returned
    pass

def test_multiple_strategies_in_query(tmp_path):
    """Test querying with multiple strategies"""
    # Given: Indexed documents
    # When: Query is run with multiple strategies
    # Then: Each strategy returns results and timing info
    pass
```

#### IT-2: REPL Mode Integration
**Test File:** `tests/integration/cli/test_repl_mode.py`

```python
def test_repl_command_execution():
    """Test commands can be executed in REPL mode"""
    # Given: REPL session started
    # When: Commands are entered interactively (simulated)
    # Then: Commands execute successfully
    # Test with pexpect or similar for interactive testing
    pass

def test_repl_maintains_state():
    """Test REPL maintains state between commands"""
    # Given: REPL session with loaded config
    # When: Multiple commands are executed
    # Then: Configuration persists across commands
    pass
```

#### IT-3: Benchmark Integration
**Test File:** `tests/integration/cli/test_benchmark.py`

```python
def test_benchmark_with_dataset(tmp_path):
    """Test benchmark runs with test dataset"""
    # Given: Benchmark dataset file
    # When: Benchmark command is executed
    # Then: Results include all queries and metrics
    # Verify:
    #   - All queries processed
    #   - Metrics calculated
    #   - Comparison table generated
    pass

def test_benchmark_output_to_file(tmp_path):
    """Test benchmark results export to file"""
    # Given: Benchmark completed
    # When: Results are exported to JSON
    # Then: File contains all benchmark data
    pass
```

#### IT-4: Error Handling Integration
**Test File:** `tests/integration/cli/test_error_handling.py`

```python
def test_missing_index_error():
    """Test clear error when querying without index"""
    # Given: No indexed data
    # When: Query command is executed
    # Then: Error message suggests running index first
    pass

def test_invalid_config_file_error():
    """Test validation of invalid config file"""
    # Given: Malformed YAML config
    # When: Config validation runs
    # Then: Specific YAML errors are reported
    pass

def test_keyboard_interrupt_handling():
    """Test graceful handling of Ctrl+C"""
    # Given: Long-running index operation
    # When: User presses Ctrl+C
    # Then: Operation stops gracefully with message
    pass
```

#### IT-5: Library Integration
**Test File:** `tests/integration/cli/test_library_integration.py`

```python
def test_cli_uses_public_api_only():
    """Verify CLI uses only public library API"""
    # Given: CLI commands
    # When: Commands are executed
    # Then: Only public API methods are called
    # Use mocking to verify API calls
    pass

def test_all_strategies_accessible():
    """Test CLI can access all library strategies"""
    # Given: Library with multiple strategies
    # When: Strategies are listed
    # Then: All strategies from Epic 4 are available
    pass
```

---

## Code Quality Requirements

### CQ-1: Code Style
- Follow PEP 8 style guide
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use Black for code formatting
- Use isort for import sorting

### CQ-2: Documentation
- Docstrings for all public functions (Google style)
- README with installation and usage examples
- Help text for all CLI commands
- Inline comments for complex logic only

### CQ-3: Test Coverage
- Minimum 80% code coverage
- All commands must have unit tests
- All user workflows must have integration tests
- Error paths must be tested

### CQ-4: Error Handling
- All exceptions caught and handled gracefully
- User-friendly error messages (no stack traces by default)
- Proper exit codes (0 for success, non-zero for errors)
- Debug flag for verbose output

### CQ-5: Type Safety
- All functions have type hints
- Mypy type checking passes with no errors
- Use Pydantic for data validation

### CQ-6: Performance
- No memory leaks in long-running operations
- Efficient streaming for large datasets
- Progress indicators for operations > 1 second

---

## Definition of Done

- [ ] All functional requirements implemented
- [ ] All acceptance criteria met
- [ ] All unit tests written and passing
- [ ] All integration tests written and passing
- [ ] Code coverage ≥ 80%
- [ ] Type checking (mypy) passes
- [ ] Code style checks (Black, isort, flake8) pass
- [ ] Documentation complete (README, docstrings, help text)
- [ ] Tested on Linux, macOS, and Windows
- [ ] CLI available via `pip install` or `python -m rag_factory_cli`
- [ ] Examples documented and tested
- [ ] Peer review completed
- [ ] Integration with library's public API verified

---

## Testing Checklist

### Before Development
- [ ] Review Epic 4 strategies to understand API
- [ ] Set up CLI framework (Typer + Rich)
- [ ] Define command structure and arguments
- [ ] Write test stubs for all commands

### During Development
- [ ] Write unit test before implementing each command
- [ ] Test each command manually
- [ ] Verify colorized output renders correctly
- [ ] Test progress bars with real operations
- [ ] Verify error messages are user-friendly

### After Implementation
- [ ] Run full test suite and achieve 80%+ coverage
- [ ] Test on different operating systems
- [ ] Test with different terminal emulators
- [ ] Verify --help text is comprehensive
- [ ] Test REPL mode interactively
- [ ] Run benchmark with large datasets
- [ ] Performance test with 1000+ documents

---

## Example Usage Scenarios

### Scenario 1: Quick Strategy Test
```bash
# Index documents
rag-factory index --path ./docs --strategy context_aware_chunking

# Query with multiple strategies
rag-factory query "What are the action items?" --strategies reranking,query_expansion
```

### Scenario 2: Configuration Testing
```bash
# Validate config before use
rag-factory config --validate my_config.yaml

# Index with config
rag-factory index --path ./docs --config my_config.yaml
```

### Scenario 3: Benchmarking
```bash
# Run benchmark with multiple strategies
rag-factory benchmark --dataset benchmark_queries.json --strategies reranking,query_expansion,semantic_chunking

# Export results
rag-factory benchmark --dataset benchmark_queries.json --output results.json
```

### Scenario 4: Interactive Exploration
```bash
# Start REPL
rag-factory repl

# Inside REPL:
>>> list strategies
>>> index --path ./docs --strategy context_aware_chunking
>>> query "test query" --strategies reranking
>>> exit
```

---

## Risk Management

### Risk 1: CLI Framework Complexity
- **Risk:** Typer/Click may be too complex for simple commands
- **Mitigation:** Start with simple commands, add features incrementally
- **Contingency:** Use argparse if Typer proves too heavy

### Risk 2: Cross-Platform Compatibility
- **Risk:** Progress bars or colors may not work on all terminals
- **Mitigation:** Test on Windows, macOS, Linux early
- **Contingency:** Provide --plain flag for basic output

### Risk 3: Library API Changes
- **Risk:** Library API may change during development
- **Mitigation:** Use only stable public API, coordinate with Epic 4
- **Contingency:** Version pin library dependencies

---

## Success Metrics

- CLI successfully indexes and queries documents
- All unit tests pass with ≥80% coverage
- All integration tests pass
- Developer feedback positive (manual testing)
- CLI validates library's public API design
- Documentation complete and clear
- Works on all target platforms
