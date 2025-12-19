# Story 17.5: Implement StrategyPair Loader and StrategyPairManager

**As a** user  
**I want** a high-level manager to load and initialize complete strategy pairs  
**So that** I can deploy RAG configurations with one command

## Acceptance Criteria
- Create `StrategyPairLoader` to parse strategy pair YAML files
- Create `StrategyPairManager` that orchestrates loading
- Validate migrations before instantiating strategies
- Resolve service references from ServiceRegistry
- Create DatabaseContext with table mappings for each strategy
- Instantiate indexing and retrieval strategies with all dependencies
- Validate capability compatibility between indexing and retrieval
- Provide clear error messages for all validation failures
- Cache loaded pairs for performance
- Integration tests with complete workflow

## Implementation
```python
from rag_factory.strategies import IIndexingStrategy, IRetrievalStrategy
from rag_factory.registry import ServiceRegistry
from rag_factory.validators import MigrationValidator

class StrategyPairManager:
    def __init__(
        self,
        service_registry: ServiceRegistry,
        config_dir: str = "strategies/",
        alembic_config: str = "alembic.ini"
    ):
        self.registry = service_registry
        self.config_dir = config_dir
        self.migration_validator = MigrationValidator(
            service_registry.get("db1"),  # Assume primary DB
            alembic_config
        )
    
    def load_pair(
        self,
        pair_name: str
    ) -> tuple[IIndexingStrategy, IRetrievalStrategy]:
        """
        Load complete strategy pair with all dependencies.
        
        Returns:
            (indexing_strategy, retrieval_strategy)
        """
        # Load YAML configuration
        config = self._load_config(f"{self.config_dir}/{pair_name}.yaml")
        
        # Validate migrations
        if config.get('migrations'):
            is_valid, missing = self.migration_validator.validate(
                config['migrations']['required_revisions']
            )
            if not is_valid:
                raise ValidationError(
                    f"Missing migrations for '{pair_name}': {missing}\n"
                    f"Run: alembic upgrade head"
                )
        
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
        
        # Validate capability compatibility (Epic 12)
        if not retrieval.requires().issubset(indexing.produces()):
            raise CompatibilityError(
                f"Retrieval requires {retrieval.requires()} "
                f"but indexing only produces {indexing.produces()}"
            )
        
        return indexing, retrieval
    
    def _create_strategy(self, config: dict, is_indexing: bool):
        """Instantiate strategy with all dependencies"""
        # Resolve service references
        services = {}
        for service_type, service_ref in config['services'].items():
            services[service_type] = self.registry.get(service_ref)
        
        # Create DatabaseContext if db_config present
        if 'db_config' in config and 'db' in services:
            db_service = services['db']
            services['db'] = db_service.get_context(
                table_mapping=config['db_config'].get('tables', {}),
                field_mapping=config['db_config'].get('fields', {})
            )
        
        # Import and instantiate strategy class
        strategy_class = self._import_strategy_class(config['strategy'])
        
        return strategy_class(
            config=config.get('config', {}),
            **services  # Pass all services as kwargs
        )
```

## Usage
```python
registry = ServiceRegistry("config/services.yaml")
manager = StrategyPairManager(registry, config_dir="strategies/")

# Load complete strategy pair
indexing, retrieval = manager.load_pair("semantic-pair")

# Use immediately
indexing.index(documents)
results = retrieval.retrieve(query)
```

## Story Points
13
