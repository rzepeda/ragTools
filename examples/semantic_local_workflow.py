"""
Example workflow using Semantic Local Pair strategy.
"""
import asyncio
import os
import logging
from pathlib import Path

# Adjust path to include project root
import sys
sys.path.append(str(Path(__file__).parent.parent))

from rag_factory.registry.service_registry import ServiceRegistry
from rag_factory.config.strategy_pair_manager import StrategyPairManager
from rag_factory.core.indexing_interface import IndexingContext
from rag_factory.core.retrieval_interface import RetrievalContext

# Import strategies to ensure registration
import rag_factory.strategies.indexing.vector_embedding
import rag_factory.strategies.retrieval.semantic_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # 1. Setup Registry
    # Ensure config/services.yaml exists
    services_path = Path("config/services.yaml")
    if not services_path.exists():
        logger.error(f"Config not found at {services_path}")
        return

    try:
        registry = ServiceRegistry(str(services_path))
    except Exception as e:
        logger.error(f"Failed to load registry: {e}")
        return

    # 2. Initialize Manager
    manager = StrategyPairManager(
        service_registry=registry,
        config_dir="strategies"
    )

    try:
        # 3. Load Strategy Pair
        logger.info("Loading strategy pair...")
        indexing, retrieval = manager.load_pair("semantic-local-pair")
        logger.info(f"Loaded: {indexing.__class__.__name__} and {retrieval.__class__.__name__}")

        # 4. Mock execution (unless DB is available)
        if "DATABASE_URL" in os.environ:
             logger.info("Starting actual processing...")
             
             # Example Document
             docs = [
                 {"id": "test_doc_1", "text": "RAG enhances LLMs by providing external context."}
             ]
             
             # Setup Contexts
             idx_context = IndexingContext(
                 database_service=indexing.deps.database_service,
                 config={}
             )
             
             # Index
             # Note: Actual execution might require chunks to be present if Indexer assumes them.
             # VectorEmbeddingIndexer typically expects chunks. 
             # For this example, we might need to inject chunks or assume strategy does chunking.
             # Assuming VectorEmbeddingIndexer handles text -> embedding.
             await indexing.process(docs, idx_context)
             
             # Retrieve
             ret_context = RetrievalContext(
                 database_service=retrieval.deps.database_service,
                 config={}
             )
             results = await retrieval.retrieve("What does RAG do?", ret_context)
             
             logger.info(f"Retrieved {len(results)} chunks:")
             for r in results:
                 logger.info(f" - {r.text} (Score: {r.score})")
        else:
            logger.info("DATABASE_URL not set, skipping execution.")

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
    finally:
        registry.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
