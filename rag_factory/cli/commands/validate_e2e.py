"""End-to-end validation command."""

import asyncio
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import yaml
import typer
from rich.console import Console

from rag_factory.config.strategy_pair_manager import StrategyPairManager
from rag_factory.registry.service_registry import ServiceRegistry
from rag_factory.core.indexing_interface import IndexingContext
from rag_factory.core.retrieval_interface import RetrievalContext
from rag_factory.strategies.base import Chunk
import rag_factory.strategies.indexing.vector_embedding
import rag_factory.strategies.retrieval.semantic_retriever

console = Console()

def validate_e2e(
    config_path: str = typer.Option(
        "cli-config.yaml",
        "--config",
        "-c",
        help="Path to CLI configuration file"
    )
):
    """Run end-to-end validation using a strategy pair."""
    load_dotenv()
    asyncio.run(_run_validation(config_path))

async def _run_validation(config_path: str):
    # Load CLI Config
    path = Path(config_path)
    if not path.exists():
        console.print(f"[red]Config file {path} not found![/red]")
        sys.exit(1)
        
    with open(path) as f:
        cli_config = yaml.safe_load(f)
        
    strategy_pair_name = cli_config.get("strategy_pair")
    service_registry_path = cli_config.get("service_registry", "config/services.yaml")
    alembic_config = cli_config.get("alembic_config", "alembic.ini")
    
    console.print(f"[bold]Validation via {path}[/bold]")
    console.print(f"Strategy Pair: {strategy_pair_name}")
    console.print(f"Service Registry: {service_registry_path}")
    
    # Initialize Registry
    # ServiceRegistry automatically uses EnvResolver to handle .env if config has ${VAR}
    registry = ServiceRegistry(service_registry_path)
    
    # Manager
    manager = StrategyPairManager(
        service_registry=registry,
        config_dir="strategies/", 
        alembic_config=alembic_config
    )
    
    # Load Pair
    try:
        indexer, retriever = manager.load_pair(strategy_pair_name)
    except Exception as e:
        console.print(f"[red]Failed to load strategy pair: {e}[/red]")
        if "Missing migrations" in str(e):
             console.print("[yellow]Tip: Run 'rag-factory check-consistency' or 'alembic upgrade head'[/yellow]")
        sys.exit(1)
        
    # Index Documents
    documents = _load_documents("sample-docs/")
    if not documents:
        console.print("[red]No sample documents found in sample-docs/[/red]")
        sys.exit(1)
        
    logger.info(f"Loaded {len(documents)} sample documents")
    console.print(f"Loaded {len(documents)} sample documents")
    
    # Indexing Context
    db_service = indexer.deps.database_service
    indexing_context = IndexingContext(database_service=db_service, config=indexer.config)
    
    console.print("[bold]Indexing...[/bold]")
    try:
        logger.info(f"Adding documents to indexer: {len(documents)} documents")
        logger.debug(f"Document IDs: {[d.get('id') for d in documents]}")
        result = await indexer.process(documents, indexing_context)
        logger.info("Indexing completed successfully")
        console.print(f"Indexing result: Produced {result.capabilities}")
        console.print(f"Stats: {result.document_count} docs, {result.chunk_count} chunks")
    except Exception as e:
         logger.exception("Indexing failed with exception")
         console.print(f"[red]Indexing failed: {e}[/red]")
         sys.exit(1)
    
    # Retrieval Context
    retrieval_context = RetrievalContext(database_service=db_service, config=retriever.config)
    
    # Query
    queries = [
        "What is Python?",
        "Explain supervised learning",
        "What are word embeddings?",
        "Who created Python?",
        "Name some embedding models"
    ]
    
    top_k = cli_config.get("defaults", {}).get("top_k", 5)
    
    console.print("[bold]Querying...[/bold]")
    try:
        for q in queries:
            console.print(f"\n[cyan]Q: {q}[/cyan]")
            results = await retriever.retrieve(q, retrieval_context, top_k=top_k) 
            if not results:
                console.print("  [dim]No results found[/dim]")
                continue
                
            for i, res in enumerate(results):
                 # Result is likely a Chunk object with fields
                 # Wait, Chunk object might be what's strictly typed or just an object with text
                 # Check Chunk definition in core/types.py or similar if needed.
                 # Assuming it has 'text' or 'content' attribute.
                 # Actually, let's look at `retrieve` return type hint `List['Chunk']`.
                 
                 text = getattr(res, 'text', str(res))[:100]
                 score = getattr(res, 'score', getattr(res, 'similarity', 0.0))
                 console.print(f"  {i+1}. {text}... (Score: {score:.4f})")
                 
    except Exception as e:
         console.print(f"[red]Retrieval failed: {e}[/red]")
         sys.exit(1)
             
    console.print("\n[green]Validation Complete[/green]")

def _load_documents(dir_path: str) -> List[Dict[str, Any]]:
    p = Path(dir_path)
    docs = []
    if not p.exists():
        return []
        
    for f in p.glob("*.txt"):
        docs.append({
            "text": f.read_text(),
            "id": f.name,
            "metadata": {"source": f.name}
        })
    return docs
