# Story 14.1: Add Pipeline Validation Command

**Story ID:** 14.1
**Epic:** Epic 14 - CLI Enhancements for Pipeline Validation & Consistency Checking
**Story Points:** 8
**Priority:** High
**Dependencies:** Epic 8.5 (CLI), Epic 11 (Dependency Injection), Epic 12 (Pipeline Separation)

---

## User Story

**As a** developer
**I want** to validate pipeline compatibility via CLI
**So that** I can check if my indexing and retrieval strategies work together before deployment

---

## Detailed Requirements

### Functional Requirements

1.  **CLI Command**
    -   Add `validate-pipeline` command to the `rag-factory` CLI.
    -   Accept comma-separated indexing strategy names via `--indexing` flag.
    -   Accept comma-separated retrieval strategy names via `--retrieval` flag.
    -   Accept configuration file path via `--config` flag.

2.  **Validation Logic**
    -   **Capability Compatibility:** Verify that the capabilities produced by the indexing pipeline meet the requirements of the retrieval pipeline.
    -   **Service Availability:** Check if the required services (e.g., Embedding, Database, LLM) are configured and available in the factory.
    -   **Strategy Consistency:** Perform consistency checks on the selected strategies to identify potential issues or suspicious patterns.

3.  **Output & Reporting**
    -   Display a clear summary of the validation results.
    -   Show detailed breakdown of capabilities (produced vs. required).
    -   Show status of service requirements.
    -   Provide clear error messages for incompatibilities (missing capabilities or services).
    -   Offer actionable suggestions for fixing identified issues (e.g., "Add VectorEmbeddingIndexing to indexing pipeline").
    -   Use colorized output for better readability (green for valid, red for invalid, yellow for warnings).

4.  **Exit Codes**
    -   Return exit code 0 if the pipeline is valid.
    -   Return a non-zero exit code if the pipeline is invalid or if an error occurs.

### Non-Functional Requirements

1.  **Performance**
    -   Validation should be fast (sub-second for typical pipelines).
    -   CLI startup time should remain low.

2.  **Usability**
    -   Command help and usage examples should be clear.
    -   Error messages should be user-friendly and guide the user towards a solution.

3.  **Maintainability**
    -   The validation logic should reuse the existing factory validation methods to ensure consistency between CLI and programmatic usage.
    -   The command implementation should be modular and easy to extend.

---

## Acceptance Criteria

### AC1: Command Interface
- [ ] `validate-pipeline` command is available in the CLI.
- [ ] `--indexing` argument accepts a list of strategy names.
- [ ] `--retrieval` argument accepts a list of strategy names.
- [ ] `--config` argument allows specifying a custom configuration file.
- [ ] Help text (`--help`) accurately describes the command and its options.

### AC2: Validation Execution
- [ ] Command successfully creates a factory instance from the config.
- [ ] Command creates indexing and retrieval pipelines with the specified strategies.
- [ ] Command invokes `factory.validate_pipeline` to perform the checks.
- [ ] Validation correctly identifies valid and invalid pipeline combinations.

### AC3: Output Formatting
- [ ] Output clearly shows "Indexing Capabilities" with checkmarks.
- [ ] Output clearly shows "Retrieval Requirements" with status (met/unmet).
- [ ] Output clearly shows "Service Requirements" with availability status.
- [ ] Valid pipelines result in a "Pipeline is VALID" message (green).
- [ ] Invalid pipelines result in a "Pipeline is INVALID" message (red).
- [ ] Missing capabilities and services are listed explicitly.
- [ ] Suggestions for fixes are displayed when applicable.

### AC4: Error Handling & Exit Codes
- [ ] Exit code is 0 for valid pipelines.
- [ ] Exit code is non-zero for invalid pipelines.
- [ ] Graceful handling of invalid strategy names (user informed).
- [ ] Graceful handling of configuration errors.

### AC5: Testing
- [ ] Unit tests for command argument parsing.
- [ ] Unit tests for output formatting (mocking console).
- [ ] Integration tests with valid and invalid pipeline configurations.
- [ ] Verification of correct exit codes in various scenarios.

---

## Technical Specifications

### Implementation Details

The implementation will leverage the `click` library for CLI command definition and `rich` for formatted console output. It will interact with the `RAGFactory` to create pipelines and perform validation.

```python
import click
import sys
from rich.console import Console
from rag_factory.factory import RAGFactory
# ... other imports

console = Console()

@click.command()
@click.option('--indexing', required=True, help='Comma-separated indexing strategy names')
@click.option('--retrieval', required=True, help='Comma-separated retrieval strategy names')
@click.option('--config', type=click.Path(), help='Configuration file path')
def validate_pipeline(indexing: str, retrieval: str, config: str):
    """
    Validate pipeline compatibility.
    """
    try:
        # 1. Parse arguments
        indexing_strategies = [s.strip() for s in indexing.split(',')]
        retrieval_strategies = [s.strip() for s in retrieval.split(',')]

        # 2. Load config & Create Factory
        # (Implementation details for loading config)
        factory = ... 

        # 3. Create Pipelines
        console.print("\n[bold]Creating pipelines...[/bold]")
        indexing_pipeline = factory.create_indexing_pipeline(indexing_strategies, ...)
        retrieval_pipeline = factory.create_retrieval_pipeline(retrieval_strategies, ...)

        # 4. Validate
        console.print("\n[bold]Running validation...[/bold]\n")
        validation = factory.validate_pipeline(indexing_pipeline, retrieval_pipeline)

        # 5. Display Results
        display_validation_results(validation, indexing_pipeline, retrieval_pipeline)

        # 6. Exit
        sys.exit(0 if validation.is_valid else 1)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

def display_validation_results(validation, indexing_pipeline, retrieval_pipeline):
    # Use rich console to print formatted results
    # Show capabilities, requirements, service status
    # Show final VALID/INVALID status
    # Show suggestions
    pass
```

### Dependencies

-   `click`: For CLI command structure.
-   `rich`: For terminal formatting (colors, tables, bold text).
-   `rag_factory`: Core library for factory and pipeline logic.

### Testing Strategy

-   **Unit Tests:**
    -   Use `click.testing.CliRunner` to invoke the command.
    -   Mock `RAGFactory` and its methods (`create_indexing_pipeline`, `create_retrieval_pipeline`, `validate_pipeline`) to test different validation outcomes without relying on the actual factory logic.
    -   Verify output strings contain expected success/failure messages and formatting.
    -   Verify exit codes.

-   **Integration Tests:**
    -   Run the command against the real `RAGFactory` with a sample configuration.
    -   Test with a known valid combination (e.g., `context_aware_chunking` + `vector_embedding` -> `reranking`).
    -   Test with a known invalid combination (e.g., `keyword_extraction` -> `reranking` which might need vectors).
