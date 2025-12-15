import logging
import importlib
from typing import Dict, Any, Tuple, Optional, Type
from pathlib import Path

from rag_factory.registry.service_registry import ServiceRegistry
from rag_factory.services.database.migration_validator import MigrationValidator
from rag_factory.services.dependencies import StrategyDependencies
from rag_factory.core.indexing_interface import IIndexingStrategy
from rag_factory.core.retrieval_interface import IRetrievalStrategy
from rag_factory.core.capabilities import IndexCapability
from rag_factory.config.strategy_loader import StrategyPairLoader, ConfigurationError

logger = logging.getLogger(__name__)

class CompatibilityError(Exception):
    """Raised when strategies are incompatible."""
    pass

class StrategyPairManager:
    """
    Manages loading and instantiation of strategy pairs.
    """
    
    def __init__(
        self,
        service_registry: ServiceRegistry,
        config_dir: str = "strategies/",
        alembic_config: str = "alembic.ini"
    ):
        """
        Initialize the manager.
        
        Args:
            service_registry: Registry to resolve services from.
            config_dir: Directory containing strategy pair YAML configurations.
            alembic_config: Path to alembic configuration file.
        """
        self.registry = service_registry
        self.config_dir = Path(config_dir)
        self.loader = StrategyPairLoader()
        
        # We assume the primary database is named "db1" as per story example,
        # but in practice we might need to be more flexible.
        # For now, we follow the story or try to find a database service.
        # If 'db1' is not in registry, we postpone initialization/validation or let it fail?
        # The story says: service_registry.get("db1")
        try:
            db_service = service_registry.get("db1")
            self.migration_validator = MigrationValidator(
                db_service, 
                alembic_config
            )
        except Exception as e:
            logger.warning(f"Could not initialize MigrationValidator: {e}. Migration checks will be skipped.")
            self.migration_validator = None
            
        self._loaded_pairs: Dict[str, Tuple[IIndexingStrategy, IRetrievalStrategy]] = {}

    def load_pair(
        self,
        pair_name: str
    ) -> Tuple[IIndexingStrategy, IRetrievalStrategy]:
        """
        Load complete strategy pair with all dependencies.
        
        Args:
            pair_name: Name of the strategy pair (without .yaml extension).
            
        Returns:
            Tuple of (indexing_strategy, retrieval_strategy).
            
        Raises:
            ConfigurationError: If loading fails.
            CompatibilityError: If strategies are incompatible.
            ValueError: If services are missing.
        """
        if pair_name in self._loaded_pairs:
            return self._loaded_pairs[pair_name]
            
        config_path = self.config_dir / f"{pair_name}.yaml"
        logger.info(f"Loading strategy pair from {config_path}")
        
        config = self.loader.load_config(config_path)
        
        # Validate migrations
        if config.get('migrations') and self.migration_validator:
            required_revisions = config['migrations'].get('required_revisions', [])
            # Fix: calling validate with correct argument type if it expects a list
            # Checking migration_validator signature might be needed.
            # Assuming validate takes a list of revisions.
            is_valid, missing = self.migration_validator.validate(required_revisions)
            
            if not is_valid:
                raise ConfigurationError(
                    f"Missing migrations for '{pair_name}': {missing}\n"
                    f"Run: alembic upgrade head"
                )
        elif config.get('migrations') and not self.migration_validator:
            logger.warning(f"Strategy pair '{pair_name}' requires migrations {config['migrations']} but validator is not available.")

        # Create indexing strategy
        indexing = self._create_strategy(
            config['indexer'],
            is_indexing=True
        )
        
        # Create retrieval strategy
        retrieval = self._create_strategy(
            config['retriever'],
            is_indexing=False
        )
        
        # Validate capability compatibility
        self._validate_compatibility(indexing, retrieval)
        
        self._loaded_pairs[pair_name] = (indexing, retrieval)
        return indexing, retrieval
    
    def _create_strategy(self, config: Dict[str, Any], is_indexing: bool) -> Any:
        """
        Instantiate strategy with all dependencies.
        
        Args:
            config: Strategy configuration dictionary.
            is_indexing: True if creating indexing strategy, False for retrieval.
            
        Returns:
            Instantiated strategy object.
        """
        # Resolve service references
        # Config['services'] maps abstract service names (llm, db) to registry keys (openai_gpt4, postgres_main)
        services_map = config.get('services', {})
        resolved_services = {}
        
        for service_type, service_ref in services_map.items():
            try:
                resolved_services[service_type] = self.registry.get(service_ref)
            except Exception as e:
                 raise ConfigurationError(f"Failed to resolve service '{service_ref}' for type '{service_type}': {e}")
        
        # Handle DatabaseContext if db_config present
        # This allows mapping specific tables/fields for this strategy
        if 'db_config' in config and 'db' in resolved_services:
            db_service = resolved_services['db']
            # Check if db_service supports get_context
            if hasattr(db_service, 'get_context'):
                resolved_services['db'] = db_service.get_context(
                    table_mapping=config['db_config'].get('tables', {}),
                    field_mapping=config['db_config'].get('fields', {})
                )
        
        # Import strategy class
        strategy_class_path = config['strategy']
        strategy_class = self._import_strategy_class(strategy_class_path)
        
        # Construct StrategyDependencies
        deps = self._build_dependencies(resolved_services)
        
        # Instantiate strategy
        # Expecting strategy(config, dependencies) signature
        try:
            strategy = strategy_class(
                config=config.get('config', {}),
                dependencies=deps
            )
        except TypeError as e:
            # Fallback or different signature handling?
            # Standard IRAGStrategy signature is (config, dependencies)
            raise ConfigurationError(f"Failed to instantiate strategy {strategy_class_path}: {e}")
            
        return strategy

    def _import_strategy_class(self, class_path: str) -> Type:
        """
        Import a class dynamically.
        
        Args:
            class_path: Dot-separated path to the class (module.ClassName).
            
        Returns:
            The class object.
        """
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
            raise ConfigurationError(f"Could not import strategy class '{class_path}': {e}")

    def _build_dependencies(self, resolved_services: Dict[str, Any]) -> StrategyDependencies:
        """
        Convert resolved services dict to StrategyDependencies object.
        
        Args:
            resolved_services: Dict mapping service types (llm, db, etc.) to service instances.
            
        Returns:
            StrategyDependencies object.
        """
        # Map generic keys to StrategyDependencies fields
        # supported keys in yaml: llm, embedding, db, graph, reranker
        return StrategyDependencies(
            llm_service=resolved_services.get('llm'),
            embedding_service=resolved_services.get('embedding'),
            database_service=resolved_services.get('db'),
            graph_service=resolved_services.get('graph'),
            reranker_service=resolved_services.get('reranker')
        )

    def _validate_compatibility(self, indexing: Any, retrieval: Any) -> None:
        """
        Validate that retrieval requirements are met by indexing capabilities.
        
        Args:
            indexing: Indexing strategy instance.
            retrieval: Retrieval strategy instance.
            
        Raises:
            CompatibilityError: If incompatible.
        """
        # Check if they implement producers/requirements
        # Indexing strategies should implement promises/produces
        # Retrieval strategies should implement requires/requirements
        
        # Per story example:
        # if not retrieval.requires().issubset(indexing.produces()):
        
        # Check if methods exist, otherwise skip (or strictly require them?)
        # IRAGStrategy doesn't enforce these, but IIndexingStrategy/IRetrievalStrategy likely do if they are well defined.
        # Based on vector_embedding.py seen earlier: produces() -> Set[IndexCapability]
        
        if hasattr(indexing, 'produces') and hasattr(retrieval, 'requires_capabilities'):
             # NOTE: vector_embedding has produces(), retrieval might have requires_capabilities() or just requires()?
             # vector_embedding.requires_services() is for services.
             # Story uses 'requires()'. I'll check for both 'requires' and 'requires_capabilities' or similar.
             # Wait, capabilities.py might define this? 
             pass # Logic continues below
             
        # Let's try to follow story convention first
        produced = set()
        if hasattr(indexing, 'produces'):
            produced = indexing.produces()
            
        required = set()
        if hasattr(retrieval, 'requires'):
            required = retrieval.requires() # Story usage
        elif hasattr(retrieval, 'requires_capabilities'): # Potential alternative name
             required = retrieval.requires_capabilities()

        # If retrieval requires nothing specific in terms of capabilities (beyond what services provide),
        # then maybe required is empty.
        
        if not required.issubset(produced):
            # Raise error
             raise CompatibilityError(
                f"Retrieval requires {required} "
                f"but indexing only produces {produced}"
            )
