# Story 14.2: Add Consistency Checking Command

**Story ID:** 14.2
**Epic:** Epic 14 - CLI Enhancements for Pipeline Validation & Consistency Checking
**Story Points:** 5
**Priority:** Medium
**Dependencies:** Epic 14.1 (Pipeline Validation - shares infrastructure)

---

## User Story

**As a** developer
**I want** to check all registered strategies for consistency issues
**So that** I can identify suspicious patterns or misconfigurations before deployment

---

## Detailed Requirements

### Functional Requirements

1.  **CLI Command**
    -   Add `check-consistency` command to the `rag-factory` CLI.
    -   Accept optional `--strategies` flag to check specific strategies (comma-separated).
    -   Accept optional `--type` flag to filter by strategy type (`indexing`, `retrieval`, `all`).
    -   Accept optional `--verbose` flag for detailed output.
    -   Accept `--config` flag for custom configuration.

2.  **Consistency Checks**
    -   **Iterate Strategies:** Iterate through all registered strategies (or selected ones) in the factory.
    -   **Capability/Service Mismatch:** Check if a strategy produces capabilities that require services which are not configured (e.g., producing vectors without an embedding service).
    -   **Suspicious Patterns:** Identify unusual combinations (e.g., a strategy that claims to do nothing, or one that requires heavy services but produces simple outputs).
    -   **Warning Generation:** Generate warning messages for any identified issues. These are warnings, not errors, as some advanced configurations might intentionally violate standard patterns.

3.  **Output & Reporting**
    -   Group results by strategy type (Indexing vs. Retrieval).
    -   Display a list of strategies with their status (Consistent vs. Warning).
    -   For strategies with warnings, display the warning messages clearly.
    -   Provide a summary at the end (Total checked, Consistent count, Warning count).
    -   Use colorized output (green for consistent, yellow for warnings).

4.  **Exit Codes**
    -   Return exit code 0 even if warnings are found (warnings do not block usage).
    -   Return non-zero exit code only if the command execution fails (e.g., config error).

### Non-Functional Requirements

1.  **Performance**
    -   Checks should be instantaneous.

2.  **Usability**
    -   Output should be easy to scan.
    -   Verbose mode should provide more context on what was checked.

---

## Acceptance Criteria

### AC1: Command Interface
- [ ] `check-consistency` command is available.
- [ ] `--strategies` filters the list of strategies to check.
- [ ] `--type` filters by indexing or retrieval.
- [ ] `--verbose` enables detailed logging.

### AC2: Check Execution
- [ ] Command correctly loads the factory and retrieves registered strategies.
- [ ] Command invokes `factory.check_all_strategies()` (or equivalent).
- [ ] Checks are applied to the selected strategies.

### AC3: Output Formatting
- [ ] Strategies are grouped by type.
- [ ] Consistent strategies are shown with a green checkmark.
- [ ] Strategies with issues are shown with a yellow warning icon.
- [ ] Warning details are indented and clearly visible.
- [ ] Summary statistics are correct.

### AC4: Exit Codes
- [ ] Exit code is 0 when warnings are present.
- [ ] Exit code is 0 when no warnings are present.
- [ ] Exit code is 1 on system error.

### AC5: Testing
- [ ] Unit tests for command filtering logic.
- [ ] Integration tests with strategies that trigger warnings.
- [ ] Integration tests with consistent strategies.

---

## Technical Specifications

### Implementation Details

```python
import click
import sys
from rich.console import Console
from rag_factory.factory import RAGFactory

console = Console()

@click.command()
@click.option('--strategies', help='Comma-separated strategy names to check (default: all)')
@click.option('--type', type=click.Choice(['indexing', 'retrieval', 'all']), default='all')
@click.option('--verbose', is_flag=True, help='Show detailed checking information')
@click.option('--config', type=click.Path(), help='Configuration file path')
def check_consistency(strategies: str, type: str, verbose: bool, config: str):
    """
    Check strategies for consistency.
    """
    try:
        # 1. Load Config & Factory
        factory = ...

        console.print("\n[bold]Checking strategy consistency...[/bold]\n")

        # 2. Determine strategies to check
        strategy_list = [s.strip() for s in strategies.split(',')] if strategies else None

        # 3. Run Checks
        # factory.check_all_strategies() should return a dict of {strategy_name: [warnings]}
        results = factory.check_all_strategies()

        # 4. Filter Results
        # (Filter by type and strategy_list)

        # 5. Display Results
        display_consistency_results(results, verbose)

        # 6. Exit
        sys.exit(0)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

def display_consistency_results(results, verbose):
    # Group by type
    # Print Indexing strategies
    # Print Retrieval strategies
    # Print Summary
    pass
```

### Dependencies

-   Same as Story 14.1 (`click`, `rich`, `rag_factory`).

### Testing Strategy

-   **Unit Tests:**
    -   Mock `factory.check_all_strategies` to return various combinations of warnings and clean results.
    -   Verify that the output correctly categorizes and displays these results.
    -   Verify filtering logic (e.g., if I pass `--type indexing`, retrieval results shouldn't show).

-   **Integration Tests:**
    -   Register a "bad" strategy (e.g., one that requires an embedding service but the factory is initialized without one) and verify the warning appears.
    -   Register a standard valid strategy and verify it shows as consistent.
