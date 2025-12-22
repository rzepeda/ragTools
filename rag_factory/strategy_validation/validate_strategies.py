"""Strategy Validation Tool

This module validates all RAG strategies by running indexing and retrieval
operations using standardized test data from basetext.json.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import traceback

from rag_factory.config.strategy_pair_manager import StrategyPairManager
from rag_factory.registry.service_registry import ServiceRegistry
from rag_factory.core.indexing_interface import IIndexingStrategy, IndexingContext
from rag_factory.core.retrieval_interface import IRetrievalStrategy, RetrievalContext

logger = logging.getLogger(__name__)


class StrategyValidator:
    """Validates RAG strategies by running indexing and retrieval operations."""
    
    def __init__(
        self,
        config_path: str = "config/services.yaml",
        strategies_dir: str = "strategies",
        test_data_path: str = "rag_factory/strategy_validation/basetext.json",
        alembic_config: str = "alembic.ini"
    ):
        """Initialize the strategy validator.
        
        Args:
            config_path: Path to services configuration file
            strategies_dir: Directory containing strategy YAML files
            test_data_path: Path to test data JSON file
            alembic_config: Path to Alembic configuration file
        """
        self.config_path = Path(config_path)
        self.strategies_dir = Path(strategies_dir)
        self.test_data_path = Path(test_data_path)
        self.alembic_config = alembic_config
        
        # Initialize backend components
        self.service_registry: Optional[ServiceRegistry] = None
        self.strategy_manager: Optional[StrategyPairManager] = None
        self.test_data: List[Dict[str, Any]] = []
        
    def initialize(self) -> None:
        """Initialize service registry and strategy manager."""
        logger.info("Initializing StrategyValidator...")
        
        # Load test data
        if not self.test_data_path.exists():
            raise FileNotFoundError(f"Test data file not found: {self.test_data_path}")
        
        with open(self.test_data_path, 'r') as f:
            self.test_data = json.load(f)
        
        logger.info(f"Loaded {len(self.test_data)} test cases from {self.test_data_path}")
        
        # Initialize ServiceRegistry
        if not self.config_path.exists():
            raise FileNotFoundError(f"Service configuration not found: {self.config_path}")
        
        logger.info(f"Initializing ServiceRegistry from {self.config_path}")
        self.service_registry = ServiceRegistry(str(self.config_path))
        
        # Initialize StrategyPairManager
        logger.info("Initializing StrategyPairManager")
        self.strategy_manager = StrategyPairManager(
            service_registry=self.service_registry,
            config_dir=str(self.strategies_dir),
            alembic_config=self.alembic_config
        )
        
        logger.info("Initialization complete")
    
    def get_strategy_list(self) -> List[str]:
        """Get list of available strategy pairs.
        
        Returns:
            List of strategy names
        """
        if not self.strategies_dir.exists():
            raise FileNotFoundError(f"Strategies directory not found: {self.strategies_dir}")
        
        yaml_files = sorted(self.strategies_dir.glob("*.yaml"))
        strategy_names = [f.stem for f in yaml_files]
        
        logger.info(f"Found {len(strategy_names)} strategies: {strategy_names}")
        return strategy_names
    
    async def validate_strategy(
        self,
        strategy_name: str,
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a single strategy with the given test case.
        
        Args:
            strategy_name: Name of the strategy to validate
            test_case: Test case data with 'text' and 'query' fields
            
        Returns:
            Dictionary with validation results
        """
        result = {
            "strategy_name": strategy_name,
            "test_case_id": test_case.get("id", "unknown"),
            "indexer": None,
            "retriever": None,
            "query": test_case.get("query", ""),
            "retrieved_chunks": [],
            "error": None,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            logger.info(f"Validating strategy: {strategy_name}")
            
            # Load strategy pair
            indexing_strategy, retrieval_strategy = self.strategy_manager.load_pair(strategy_name)
            
            # Record strategy types
            result["indexer"] = type(indexing_strategy).__name__
            result["retriever"] = type(retrieval_strategy).__name__ if retrieval_strategy else None
            
            # Prepare document for indexing
            doc = {
                "id": f"test_{test_case.get('id', 'doc')}",
                "text": test_case["text"]
            }
            
            # Create indexing context
            indexing_context = IndexingContext(
                database_service=indexing_strategy.deps.database_service,
                config={}
            )
            
            # Run indexing
            logger.info(f"  Indexing document for {strategy_name}...")
            indexing_result = await indexing_strategy.process([doc], indexing_context)
            logger.info(f"  Indexing complete. Capabilities: {indexing_result.capabilities}")
            
            # Run retrieval if retriever is available
            if retrieval_strategy:
                # Create retrieval context
                retrieval_context = RetrievalContext(
                    database_service=retrieval_strategy.deps.database_service,
                    config={"top_k": 5}
                )
                
                # Run retrieval
                logger.info(f"  Retrieving for query: '{test_case['query']}'")
                chunks = await retrieval_strategy.retrieve(test_case["query"], retrieval_context)
                
                # Extract text from chunks
                retrieved_texts = []
                for chunk in chunks:
                    # Handle both Chunk objects and dictionaries
                    if hasattr(chunk, 'text'):
                        retrieved_texts.append(chunk.text)
                    elif isinstance(chunk, dict) and 'text' in chunk:
                        retrieved_texts.append(chunk['text'])
                    else:
                        retrieved_texts.append(str(chunk))
                
                result["retrieved_chunks"] = retrieved_texts
                logger.info(f"  Retrieved {len(retrieved_texts)} chunks")
            else:
                logger.info(f"  No retriever available for {strategy_name}")
            
            logger.info(f"✓ Successfully validated {strategy_name}")
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            result["error"] = error_msg
            result["traceback"] = traceback.format_exc()
            logger.error(f"✗ Error validating {strategy_name}: {error_msg}")
            logger.debug(traceback.format_exc())
        
        return result
    
    async def validate_all(
        self,
        strategies: Optional[List[str]] = None,
        test_case_index: int = 0
    ) -> List[Dict[str, Any]]:
        """Validate all or specified strategies.
        
        Args:
            strategies: List of strategy names to validate (None = all)
            test_case_index: Index of test case to use (default: 0 = first)
            
        Returns:
            List of validation results
        """
        # Get strategy list
        if strategies is None:
            strategies = self.get_strategy_list()
        
        # Get test case
        if test_case_index >= len(self.test_data):
            raise ValueError(f"Test case index {test_case_index} out of range (max: {len(self.test_data) - 1})")
        
        test_case = self.test_data[test_case_index]
        logger.info(f"Using test case: {test_case.get('id', 'unknown')} - {test_case.get('type', 'unknown')}")
        
        # Validate each strategy
        results = []
        for i, strategy_name in enumerate(strategies, 1):
            logger.info(f"\n[{i}/{len(strategies)}] Validating {strategy_name}...")
            result = await self.validate_strategy(strategy_name, test_case)
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """Save validation results to JSON file.
        
        Args:
            results: List of validation results
            output_path: Path to output JSON file
        """
        output_file = Path(output_path)
        
        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare summary
        total = len(results)
        successful = sum(1 for r in results if r["error"] is None)
        failed = total - successful
        
        output_data = {
            "summary": {
                "total_strategies": total,
                "successful": successful,
                "failed": failed,
                "timestamp": datetime.now().isoformat()
            },
            "results": results
        }
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"\nResults saved to {output_file}")
        logger.info(f"Summary: {successful}/{total} strategies validated successfully")


async def run_validation(
    config_path: str = "config/services.yaml",
    strategies_dir: str = "strategies",
    test_data_path: str = "rag_factory/strategy_validation/basetext.json",
    output_path: str = "validation_results.json",
    strategies: Optional[List[str]] = None,
    test_case_index: int = 0,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """Run strategy validation.
    
    Args:
        config_path: Path to services configuration file
        strategies_dir: Directory containing strategy YAML files
        test_data_path: Path to test data JSON file
        output_path: Path to output JSON file
        strategies: List of specific strategies to validate (None = all)
        test_case_index: Index of test case to use
        verbose: Enable verbose logging
        
    Returns:
        List of validation results
    """
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Create validator
    validator = StrategyValidator(
        config_path=config_path,
        strategies_dir=strategies_dir,
        test_data_path=test_data_path
    )
    
    # Initialize
    validator.initialize()
    
    # Run validation
    results = await validator.validate_all(
        strategies=strategies,
        test_case_index=test_case_index
    )
    
    # Save results
    validator.save_results(results, output_path)
    
    return results
