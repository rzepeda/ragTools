"""Pipeline validation command."""

import sys
from typing import Optional

import typer
from rich.console import Console

from rag_factory.cli.formatters.validation import format_validation_results
from rag_factory.cli.formatters import print_error
from rag_factory.cli.utils.validation import parse_strategy_list, validate_config_file
from rag_factory.factory import RAGFactory
from rag_factory.core.indexing_interface import IndexingContext
from rag_factory.core.retrieval_interface import RetrievalContext
from rag_factory.core.pipeline import IndexingPipeline, RetrievalPipeline

console = Console()


def validate_pipeline(
    indexing: str = typer.Option(
        ...,
        "--indexing",
        "-i",
        help="Comma-separated indexing strategy names"
    ),
    retrieval: str = typer.Option(
        ...,
        "--retrieval",
        "-r",
        help="Comma-separated retrieval strategy names"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file path (YAML or JSON)"
    ),
) -> None:
    """
    Validate pipeline compatibility.
    
    This command checks if your indexing and retrieval strategies work together
    by validating capability compatibility and service availability.
    
    Examples:
    
        Validate a simple pipeline:
        $ rag-factory validate-pipeline --indexing vector_embedding --retrieval reranking
        
        Validate with multiple strategies:
        $ rag-factory validate-pipeline --indexing "context_aware,vector_embedding" --retrieval "reranking"
        
        Validate with config file:
        $ rag-factory validate-pipeline --indexing vector_embedding --retrieval reranking --config config.yaml
    """
    try:
        console.print("\n[bold]Validating pipeline compatibility...[/bold]\n")
        
        # 1. Parse strategy names
        indexing_strategies = parse_strategy_list(indexing)
        retrieval_strategies = parse_strategy_list(retrieval)
        
        if not indexing_strategies:
            print_error("No indexing strategies provided")
            raise typer.Exit(1)
        
        if not retrieval_strategies:
            print_error("No retrieval strategies provided")
            raise typer.Exit(1)
        
        console.print(f"[dim]Indexing strategies: {', '.join(indexing_strategies)}[/dim]")
        console.print(f"[dim]Retrieval strategies: {', '.join(retrieval_strategies)}[/dim]")
        console.print()
        
        # 2. Load config if provided
        factory_config = {}
        if config:
            console.print(f"[dim]Loading configuration from: {config}[/dim]")
            factory_config = validate_config_file(config)
            console.print("[green]✓[/green] Configuration loaded\n")
        
        # 3. Create factory instance
        # Note: For validation purposes, we create a factory without services
        # The validation will check what services are required vs available
        factory = RAGFactory()
        
        # 4. Create indexing pipeline
        console.print("[dim]Creating indexing pipeline...[/dim]")
        indexing_strategy_instances = []
        for strategy_name in indexing_strategies:
            try:
                # Create strategy with empty config for validation purposes
                strategy = factory.create_strategy(strategy_name, config={})
                indexing_strategy_instances.append(strategy)
            except Exception as e:
                print_error(f"Failed to create indexing strategy '{strategy_name}': {e}")
                raise typer.Exit(1)
        
        indexing_context = IndexingContext(
            database_service=factory.dependencies.database_service  # type: ignore[arg-type]
        )
        indexing_pipeline = IndexingPipeline(
            strategies=indexing_strategy_instances,  # type: ignore[arg-type]
            context=indexing_context
        )
        console.print("[green]✓[/green] Indexing pipeline created\n")
        
        # 5. Create retrieval pipeline
        console.print("[dim]Creating retrieval pipeline...[/dim]")
        retrieval_strategy_instances = []
        for strategy_name in retrieval_strategies:
            try:
                # Create strategy with empty config for validation purposes
                strategy = factory.create_strategy(strategy_name, config={})
                retrieval_strategy_instances.append(strategy)
            except Exception as e:
                print_error(f"Failed to create retrieval strategy '{strategy_name}': {e}")
                raise typer.Exit(1)
        
        
        retrieval_context = RetrievalContext(
            database_service=factory.dependencies.database_service  # type: ignore[arg-type]
        )
        retrieval_pipeline = RetrievalPipeline(
            strategies=retrieval_strategy_instances,  # type: ignore[arg-type]
            context=retrieval_context
        )
        console.print("[green]✓[/green] Retrieval pipeline created\n")
        
        # 6. Validate pipeline
        console.print("[dim]Running validation...[/dim]\n")
        validation = factory.validate_pipeline(indexing_pipeline, retrieval_pipeline)
        
        # 7. Display results
        format_validation_results(
            validation,
            indexing_pipeline,
            retrieval_pipeline,
            console
        )
        
        # 8. Exit with appropriate code
        if validation.is_valid:
            console.print("[dim]Use these strategies in your indexing and retrieval pipelines.[/dim]")
            sys.exit(0)
        else:
            sys.exit(1)
            
    except typer.Exit:
        raise
    except Exception as e:
        if "--debug" in sys.argv:
            raise
        print_error(f"Validation failed: {e}")
        raise typer.Exit(1)
