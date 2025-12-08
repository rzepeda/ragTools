# Epic 14: CLI Enhancements for Pipeline Validation & Consistency Checking

**Epic Goal:** Enhance the existing CLI tool with pipeline validation and consistency checking capabilities, enabling developers to validate RAG pipelines and detect strategy inconsistencies before deployment.

**Epic Story Points Total:** 13

**Dependencies:** 
- Epic 8.5 (CLI/Dev Server - COMPLETED ✅)
- Epic 11 (Dependency Injection with ConsistencyChecker - must be complete)
- Epic 12 (Pipeline Separation - must be complete)

**Status:** Ready for implementation after Epics 11 and 12

---

## Background

Epic 8.5 delivered a basic CLI tool for testing strategies and running POCs. With the introduction of:
- Dependency injection (Epic 11) with consistency checking
- Pipeline separation (Epic 12) with capability validation

The CLI needs enhancements to help developers:
1. Validate pipeline compatibility before indexing
2. Check strategies for consistency issues
3. Debug capability and service mismatches
4. Understand why strategies aren't compatible

This epic extends the CLI without requiring a full server deployment, keeping it as a lightweight development tool.

---

## Story 14.1: Add Pipeline Validation Command

**As a** developer  
**I want** to validate pipeline compatibility via CLI  
**So that** I can check if my indexing and retrieval strategies work together

**Acceptance Criteria:**
- Add `validate-pipeline` command to CLI
- Accept indexing strategy names via `--indexing` flag
- Accept retrieval strategy names via `--retrieval` flag
- Show capability compatibility check results
- Show service availability check results
- Display clear error messages for incompatibilities
- Show suggestions for fixing issues
- Colorized output (green for valid, red for invalid, yellow for warnings)
- Exit code 0 for valid, non-zero for invalid
- Unit tests for validation command
- Documentation with examples

**Command Specification:**

```bash
rag-factory validate-pipeline \
  --indexing context_aware_chunking,vector_embedding \
  --retrieval reranking,query_expansion \
  --config config.yaml
```

**Implementation:**

```python
import click
from rich.console import Console
from rich.table import Table

console = Console()

@click.command()
@click.option('--indexing', required=True, help='Comma-separated indexing strategy names')
@click.option('--retrieval', required=True, help='Comma-separated retrieval strategy names')
@click.option('--config', type=click.Path(), help='Configuration file path')
def validate_pipeline(indexing: str, retrieval: str, config: str):
    """
    Validate pipeline compatibility.
    
    Checks:
    1. Capability compatibility (does indexing produce what retrieval needs?)
    2. Service availability (are required services configured?)
    3. Strategy consistency (any suspicious patterns?)
    """
    try:
        # Parse strategy names
        indexing_strategies = [s.strip() for s in indexing.split(',')]
        retrieval_strategies = [s.strip() for s in retrieval.split(',')]
        
        # Load config
        factory_config = load_config(config) if config else {}
        
        # Create factory
        factory = create_factory_from_config(factory_config)
        
        # Create pipelines
        console.print("\n[bold]Creating pipelines...[/bold]")
        
        indexing_pipeline = factory.create_indexing_pipeline(
            indexing_strategies,
            [{}] * len(indexing_strategies)
        )
        
        retrieval_pipeline = factory.create_retrieval_pipeline(
            retrieval_strategies,
            [{}] * len(retrieval_strategies)
        )
        
        # Run validation (includes consistency checks)
        console.print("\n[bold]Running validation...[/bold]\n")
        
        validation = factory.validate_pipeline(indexing_pipeline, retrieval_pipeline)
        
        # Display results
        display_validation_results(validation, indexing_pipeline, retrieval_pipeline)
        
        # Exit with appropriate code
        sys.exit(0 if validation.is_valid else 1)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

def display_validation_results(
    validation: ValidationResult,
    indexing_pipeline: IndexingPipeline,
    retrieval_pipeline: RetrievalPipeline
):
    """Display validation results in a formatted way"""
    
    # Show capabilities
    console.print("[bold cyan]Indexing Capabilities:[/bold cyan]")
    caps = indexing_pipeline.get_capabilities()
    for cap in sorted(caps, key=lambda c: c.name):
        console.print(f"  ✓ {cap.name}")
    
    console.print("\n[bold cyan]Retrieval Requirements:[/bold cyan]")
    reqs = retrieval_pipeline.get_requirements()
    for req in sorted(reqs, key=lambda r: r.name):
        has_it = req in caps
        icon = "✓" if has_it else "✗"
        color = "green" if has_it else "red"
        console.print(f"  [{color}]{icon} {req.name}[/{color}]")
    
    # Show service requirements
    console.print("\n[bold cyan]Service Requirements:[/bold cyan]")
    service_reqs = retrieval_pipeline.get_service_requirements()
    # Check availability from factory dependencies
    # ... (implementation details)
    
    # Show validation result
    console.print("\n" + "="*50)
    
    if validation.is_valid:
        console.print("[bold green]✓ Pipeline is VALID[/bold green]")
    else:
        console.print("[bold red]✗ Pipeline is INVALID[/bold red]")
        
        if validation.missing_capabilities:
            console.print("\n[yellow]Missing capabilities:[/yellow]")
            for cap in validation.missing_capabilities:
                console.print(f"  • {cap.name}")
        
        if validation.missing_services:
            console.print("\n[yellow]Missing services:[/yellow]")
            for svc in validation.missing_services:
                console.print(f"  • {svc.name}")
        
        if validation.suggestions:
            console.print("\n[bold cyan]Suggestions:[/bold cyan]")
            for suggestion in validation.suggestions:
                console.print(f"  → {suggestion}")
```

**Example Output:**

```bash
$ rag-factory validate-pipeline --indexing context_aware_chunking,vector_embedding --retrieval reranking

Creating pipelines...
  ✓ Created indexing pipeline (2 strategies)
  ✓ Created retrieval pipeline (1 strategy)

Running validation...

Indexing Capabilities:
  ✓ CHUNKS
  ✓ DATABASE
  ✓ VECTORS

Retrieval Requirements:
  ✓ CHUNKS
  ✓ VECTORS

Service Requirements:
  ✓ EMBEDDING (available)
  ✓ DATABASE (available)
  ✓ RERANKER (available)

==================================================
✓ Pipeline is VALID

$ rag-factory validate-pipeline --indexing keyword_extraction --retrieval reranking

Creating pipelines...
  ✓ Created indexing pipeline (1 strategy)
  ✓ Created retrieval pipeline (1 strategy)

Running validation...

Indexing Capabilities:
  ✓ DATABASE
  ✓ KEYWORDS

Retrieval Requirements:
  ✗ CHUNKS
  ✗ VECTORS

==================================================
✗ Pipeline is INVALID

Missing capabilities:
  • CHUNKS
  • VECTORS

Suggestions:
  → Add ContextAwareChunking to indexing pipeline
  → Add VectorEmbeddingIndexing to indexing pipeline
```

**Story Points:** 8

---

## Story 14.2: Add Consistency Checking Command

**As a** developer  
**I want** to check all registered strategies for consistency issues  
**So that** I can identify suspicious patterns before deployment

**Acceptance Criteria:**
- Add `check-consistency` command to CLI
- Check all registered indexing strategies
- Check all registered retrieval strategies
- Display warnings for inconsistent strategies
- Show green checkmarks for consistent strategies
- Summary statistics at the end
- Option to check specific strategies only
- Colorized output
- Exit code 0 (warnings don't fail)
- Unit tests for consistency command
- Documentation with examples

**Command Specification:**

```bash
# Check all strategies
rag-factory check-consistency

# Check specific strategies
rag-factory check-consistency --strategies context_aware_chunking,reranking

# Check only indexing or retrieval
rag-factory check-consistency --type indexing
rag-factory check-consistency --type retrieval

# Verbose mode (show what's being checked)
rag-factory check-consistency --verbose
```

**Implementation:**

```python
@click.command()
@click.option('--strategies', help='Comma-separated strategy names to check (default: all)')
@click.option('--type', type=click.Choice(['indexing', 'retrieval', 'all']), default='all')
@click.option('--verbose', is_flag=True, help='Show detailed checking information')
@click.option('--config', type=click.Path(), help='Configuration file path')
def check_consistency(strategies: str, type: str, verbose: bool, config: str):
    """
    Check strategies for consistency between capabilities and services.
    
    Warnings indicate suspicious patterns but don't prevent strategy usage.
    """
    try:
        # Load config and create factory
        factory_config = load_config(config) if config else {}
        factory = create_factory_from_config(factory_config)
        
        console.print("\n[bold]Checking strategy consistency...[/bold]\n")
        
        # Determine which strategies to check
        if strategies:
            strategy_list = [s.strip() for s in strategies.split(',')]
        else:
            strategy_list = None  # Check all
        
        # Run consistency checks
        results = factory.check_all_strategies()
        
        # Filter by type and strategy list if specified
        filtered_results = filter_results(results, type, strategy_list)
        
        # Display results
        display_consistency_results(filtered_results, verbose)
        
        # Always exit 0 (warnings don't fail)
        sys.exit(0)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

def display_consistency_results(results: dict, verbose: bool):
    """Display consistency check results"""
    
    # Group by type
    indexing_results = {k: v for k, v in results.items() if k.startswith('indexing:')}
    retrieval_results = {k: v for k, v in results.items() if k.startswith('retrieval:')}
    
    # Show indexing strategies
    if indexing_results or not results:
        console.print("[bold cyan]Indexing Strategies:[/bold cyan]")
        
        for strategy_name, warnings in indexing_results.items():
            name = strategy_name.replace('indexing:', '')
            
            if warnings:
                console.print(f"  [yellow]⚠️  {name}[/yellow]")
                for warning in warnings:
                    # Extract just the message part
                    msg = warning.split(':', 1)[1].strip() if ':' in warning else warning
                    console.print(f"     → {msg}")
            else:
                console.print(f"  [green]✓ {name}[/green]")
        
        if not indexing_results:
            console.print("  [dim]No indexing strategies registered[/dim]")
    
    console.print()
    
    # Show retrieval strategies
    if retrieval_results or not results:
        console.print("[bold cyan]Retrieval Strategies:[/bold cyan]")
        
        for strategy_name, warnings in retrieval_results.items():
            name = strategy_name.replace('retrieval:', '')
            
            if warnings:
                console.print(f"  [yellow]⚠️  {name}[/yellow]")
                for warning in warnings:
                    msg = warning.split(':', 1)[1].strip() if ':' in warning else warning
                    console.print(f"     → {msg}")
            else:
                console.print(f"  [green]✓ {name}[/green]")
        
        if not retrieval_results:
            console.print("  [dim]No retrieval strategies registered[/dim]")
    
    # Summary
    total = len(results)
    warnings_count = sum(1 for v in results.values() if v)
    consistent_count = total - warnings_count
    
    console.print("\n" + "="*50)
    console.print(f"[bold]Summary:[/bold] {total} strategies checked")
    console.print(f"  [green]✓ {consistent_count} consistent[/green]")
    console.print(f"  [yellow]⚠️  {warnings_count} warnings[/yellow]")
    
    if warnings_count > 0:
        console.print("\n[dim]Note: Warnings don't prevent strategy usage,[/dim]")
        console.print("[dim]but indicate patterns worth reviewing.[/dim]")
```

**Example Output:**

```bash
$ rag-factory check-consistency

Checking strategy consistency...

Indexing Strategies:
  ✓ context_aware_chunking
  ✓ vector_embedding
  ✓ keyword_extraction
  ⚠️  experimental_strategy
     → Produces VECTORS but doesn't require EMBEDDING service
     → This is unusual unless loading pre-computed embeddings

Retrieval Strategies:
  ✓ reranking
  ✓ query_expansion
  ✓ multi_query
  ✓ hierarchical_rag

==================================================
Summary: 8 strategies checked
  ✓ 7 consistent
  ⚠️  1 warnings

Note: Warnings don't prevent strategy usage,
but indicate patterns worth reviewing.

$ rag-factory check-consistency --type indexing --strategies experimental_strategy

Checking strategy consistency...

Indexing Strategies:
  ⚠️  experimental_strategy
     → Produces VECTORS but doesn't require EMBEDDING service
     → This is unusual unless loading pre-computed embeddings

==================================================
Summary: 1 strategy checked
  ✓ 0 consistent
  ⚠️  1 warnings
```

**Story Points:** 5

---

## Sprint Planning

**Sprint 18:** Stories 14.1, 14.2 (13 points)

---

## CLI Command Reference (Updated)

After this epic, the CLI will have:

### Existing Commands (Epic 8.5):
```bash
# Index documents
rag-factory index --path ./docs --strategies chunking,embedding

# Query
rag-factory query "question" --strategies reranking

# List strategies
rag-factory strategies --list

# Validate config
rag-factory config --validate config.yaml

# Benchmark
rag-factory benchmark --dataset test.json

# Dev server
rag-factory serve --port 8000
```

### New Commands (Epic 14):
```bash
# Validate pipeline compatibility
rag-factory validate-pipeline \
  --indexing chunking,embedding \
  --retrieval reranking

# Check consistency
rag-factory check-consistency
rag-factory check-consistency --type indexing
rag-factory check-consistency --strategies my_strategy
```

---

## Integration with Factory

The CLI commands use the factory's validation methods directly:

```python
# In CLI implementation
def create_factory_from_config(config: dict):
    """Create factory with services from config"""
    
    # Create services based on config
    if config.get('use_onnx', True):
        # ONNX for CLI/testing
        embedding_svc = ONNXEmbeddingService(
            model_path=config.get('embedding_model', './models/embedding.onnx')
        )
        llm_svc = ONNXLLMService(
            model_path=config.get('llm_model', './models/llm.onnx')
        )
    else:
        # API services for production testing
        embedding_svc = OpenAIEmbeddingService(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        llm_svc = AnthropicLLMService(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
    
    # Database service
    db_svc = PostgresqlDatabaseService(
        connection_string=config.get('database_url', 'postgresql://localhost/rag')
    )
    
    # Create factory with services
    return RAGFactory(
        embedding_service=embedding_svc,
        llm_service=llm_svc,
        database_service=db_svc
    )
```

---

## Testing Strategy

### Unit Tests
- Test command parsing
- Test output formatting
- Mock factory methods
- Test error handling

### Integration Tests
- Test with real factory
- Test with valid pipelines
- Test with invalid pipelines
- Test with inconsistent strategies
- Test exit codes

### Example Test:

```python
def test_validate_pipeline_command():
    runner = CliRunner()
    
    # Valid pipeline
    result = runner.invoke(
        validate_pipeline,
        ['--indexing', 'chunking,embedding', '--retrieval', 'reranking']
    )
    assert result.exit_code == 0
    assert '✓ Pipeline is VALID' in result.output
    
    # Invalid pipeline
    result = runner.invoke(
        validate_pipeline,
        ['--indexing', 'keyword_extraction', '--retrieval', 'reranking']
    )
    assert result.exit_code != 0
    assert '✗ Pipeline is INVALID' in result.output
    assert 'VECTORS' in result.output  # Missing capability

def test_check_consistency_command():
    runner = CliRunner()
    
    result = runner.invoke(check_consistency, [])
    assert result.exit_code == 0  # Always 0 (warnings don't fail)
    assert 'Summary:' in result.output
```

---

## Documentation Updates

- [ ] Update CLI documentation with new commands
- [ ] Add validation workflow guide
- [ ] Add consistency checking best practices
- [ ] Add troubleshooting guide using CLI
- [ ] Update examples with validation steps

---

## Success Criteria

- [ ] `validate-pipeline` command implemented
- [ ] `check-consistency` command implemented
- [ ] Colorized output working
- [ ] Correct exit codes
- [ ] Clear error messages
- [ ] Helpful suggestions
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] CLI validates same as factory methods
- [ ] Examples demonstrate common workflows

---

## Benefits Achieved

**Developer Experience:**
- ✅ Validate pipelines before writing code
- ✅ Catch inconsistencies early
- ✅ Understand compatibility issues quickly
- ✅ Get suggestions for fixes

**Quality:**
- ✅ Fewer runtime errors
- ✅ Better strategy design
- ✅ Consistent patterns encouraged

**Debugging:**
- ✅ Easy to identify issues
- ✅ Clear error messages
- ✅ Actionable suggestions

---

## Future Enhancements (Post-Epic 14)

**Possible additions in future epics:**
- Interactive pipeline builder (wizard-style)
- Capability graph visualization
- Strategy recommendation engine
- Performance profiling commands
- Migration helper (old strategies → new interfaces)

---

## Example Workflows

### Workflow 1: Validate Before Implementation

```bash
# 1. Check what capabilities you need
rag-factory validate-pipeline \
  --indexing chunking,embedding \
  --retrieval reranking
  
# Output shows: ✓ Valid

# 2. Proceed with implementation confidently
python my_rag_app.py
```

### Workflow 2: Debug Incompatibility

```bash
# 1. Try to validate
rag-factory validate-pipeline \
  --indexing keyword_extraction \
  --retrieval reranking

# Output shows: Missing VECTORS
# Suggestion: Add VectorEmbeddingIndexing

# 2. Add suggested strategy
rag-factory validate-pipeline \
  --indexing keyword_extraction,vector_embedding \
  --retrieval reranking

# Output shows: ✓ Valid
```

### Workflow 3: Check Custom Strategy

```bash
# 1. Check consistency after creating custom strategy
rag-factory check-consistency --strategies my_custom_strategy

# Output shows warnings if any

# 2. Fix issues based on warnings

# 3. Verify fix
rag-factory check-consistency --strategies my_custom_strategy

# Output shows: ✓ consistent
```

---

## Notes

- This epic extends Epic 8.5 (CLI) without replacing it
- All new commands are non-breaking additions
- CLI remains a development tool (not production-ready)
- Uses same validation logic as programmatic API
- Helps developers before they write code
- Warnings system encourages good patterns without blocking
