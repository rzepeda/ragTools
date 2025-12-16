"""Validation result formatters for CLI."""

from rich.console import Console
from rich.panel import Panel

from rag_factory.core.capabilities import ValidationResult
from rag_factory.core.pipeline import IndexingPipeline, RetrievalPipeline


def format_validation_results(
    validation: ValidationResult,
    indexing_pipeline: IndexingPipeline,
    retrieval_pipeline: RetrievalPipeline,
    console: Console
) -> None:
    """
    Format and display validation results with rich formatting.
    
    Args:
        validation: ValidationResult from factory.validate_pipeline()
        indexing_pipeline: The indexing pipeline being validated
        retrieval_pipeline: The retrieval pipeline being validated
        console: Rich console for output
    """
    console.print()
    console.print("[bold]Pipeline Validation Results[/bold]")
    console.print()
    
    # Display indexing capabilities
    _display_capabilities(indexing_pipeline, console)
    
    # Display retrieval requirements
    _display_requirements(retrieval_pipeline, indexing_pipeline, console)
    
    # Display service requirements
    _display_service_requirements(retrieval_pipeline, validation, console)
    
    # Display final validation status
    _display_validation_status(validation, console)
    
    # Display suggestions if any
    if validation.suggestions:
        _display_suggestions(validation.suggestions, console)


def _display_capabilities(pipeline: IndexingPipeline, console: Console) -> None:
    """Display indexing capabilities."""
    capabilities = pipeline.get_capabilities()
    
    console.print("[bold cyan]Indexing Capabilities:[/bold cyan]")
    if capabilities:
        for cap in sorted(capabilities, key=lambda c: c.name):
            console.print(f"  [green]✓[/green] {cap.name}")
    else:
        console.print("  [yellow]⚠[/yellow] No capabilities produced")
    console.print()


def _display_requirements(
    retrieval_pipeline: RetrievalPipeline,
    indexing_pipeline: IndexingPipeline,
    console: Console
) -> None:
    """Display retrieval requirements and their status."""
    requirements = retrieval_pipeline.get_requirements()
    capabilities = indexing_pipeline.get_capabilities()
    
    console.print("[bold cyan]Retrieval Requirements:[/bold cyan]")
    if requirements:
        for req in sorted(requirements, key=lambda r: r.name):
            if req in capabilities:
                console.print(f"  [green]✓[/green] {req.name} [dim](met)[/dim]")
            else:
                console.print(f"  [red]✗[/red] {req.name} [dim](unmet)[/dim]")
    else:
        console.print("  [dim]No requirements[/dim]")
    console.print()


def _display_service_requirements(
    retrieval_pipeline: RetrievalPipeline,
    validation: ValidationResult,
    console: Console
) -> None:
    """Display service requirements and their availability."""
    service_reqs = retrieval_pipeline.get_service_requirements()
    missing_services = validation.missing_services
    
    console.print("[bold cyan]Service Requirements:[/bold cyan]")
    if service_reqs:
        for svc in sorted(service_reqs, key=lambda s: s.name):
            if svc in missing_services:
                console.print(f"  [red]✗[/red] {svc.name} [dim](unavailable)[/dim]")
            else:
                console.print(f"  [green]✓[/green] {svc.name} [dim](available)[/dim]")
    else:
        console.print("  [dim]No service requirements[/dim]")
    console.print()


def _display_validation_status(validation: ValidationResult, console: Console) -> None:
    """Display final validation status."""
    if validation.is_valid:
        panel = Panel(
            "[bold green]Pipeline is VALID ✓[/bold green]\n\n"
            "All capability and service requirements are met.",
            border_style="green",
            title="Validation Result"
        )
    else:
        # Build error message
        error_parts = []
        if validation.missing_capabilities:
            caps = sorted([c.name for c in validation.missing_capabilities])
            error_parts.append(f"Missing capabilities: {', '.join(caps)}")
        if validation.missing_services:
            svcs = sorted([s.name for s in validation.missing_services])
            error_parts.append(f"Missing services: {', '.join(svcs)}")

        error_msg = "\n".join(error_parts) if error_parts else validation.message

        panel = Panel(
            f"[bold red]Pipeline is INVALID ✗[/bold red]\n\n{error_msg}",
            border_style="red",
            title="Validation Result"
        )
    
    console.print(panel)
    console.print()


def _display_suggestions(suggestions: list, console: Console) -> None:
    """Display actionable suggestions for fixing issues."""
    console.print("[bold yellow]Suggestions:[/bold yellow]")
    for i, suggestion in enumerate(suggestions, 1):
        console.print(f"  {i}. {suggestion}")
    console.print()
