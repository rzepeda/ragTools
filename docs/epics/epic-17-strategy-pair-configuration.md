# Epic 17: Strategy Pair Configuration System

**Epic Goal:** Create a configuration system that bundles compatible indexing and retrieval strategies with their required services, migrations, and infrastructure requirements, enabling portable and reproducible RAG deployments.

**Epic Story Points Total:** 47

**Dependencies:** 
- Epic 11 (Pipeline Separation - capability system must exist)
- Epic 15 (Alembic Migrations - migration system must be working)

---

## Background

Currently, the RAG Factory has:
- ✅ Separate indexing and retrieval strategies
- ✅ Capability-based compatibility checking
- ✅ Alembic migration system for database schema
- ✅ Dependency injection for services

**The Problem:** While strategies are compatible at the interface level, they may require:
- Different database structures (pgvector indexes vs full-text search)
- Different service providers (OpenAI vs local embeddings)
- Different infrastructure (PostgreSQL vs Neo4j)
- Specific Alembic migrations to be applied

**The Solution:** Strategy Pairs - pre-configured bundles that specify:
1. Compatible indexing + retrieval strategies
2. Required service configurations (LLM, embeddings, tokenizer)
3. Database requirements (type, extensions, migrations)
4. Environment-specific overrides (dev vs prod)

---

## Core Concepts

### What is a Strategy Pair?

A **Strategy Pair** is a YAML configuration file that bundles:
- Compatible indexing and retrieval strategies
- Service provider specifications
- Required Alembic migration revisions
- Expected database schema elements
- Environment-specific configurations

### Example Strategy Pair

```yaml
# strategies/semantic-openai-pair.yaml
name: "semantic-openai-pair"
version: "1.0.0"
description: "Semantic search using OpenAI embeddings with PostgreSQL pgvector"

indexing:
  strategy: "VectorEmbeddingIndexing"
  config:
    chunk_size: 512
    overlap: 50

retrieval:
  strategy: "SemanticRetrieval"
  config:
    top_k: 5
    similarity_threshold: 0.7

services:
  embeddings:
    provider: "openai"
    model: "text-embedding-3-small"
    dimensions: 1536
  llm:
    provider: "openai"
    model: "gpt-4"
  database:
    type: "postgres"
    extensions: ["pgvector", "uuid-ossp"]
    min_version: "12.0"

migrations:
  type: "alembic"
  required_revisions:
    - "semantic_vectors_table"
    - "vector_indexes"

expected_schema:
  tables:
    - "document_vectors"
  indexes:
    - "idx_document_vectors_embedding"
  extensions:
    - "vector"

environments:
  development:
    services:
      embeddings:
        provider: "openai"
        model: "text-embedding-3-small"
    database:
      host: "localhost"
  production:
    services:
      embeddings:
        provider: "azure-openai"
        model: "text-embedding-3-large"
    database:
      host: "prod-db.example.com"
      replicas: 3
```

---

## Story 17.1: Design Strategy Pair Configuration Schema

**As a** developer  
**I want** a well-defined YAML schema for strategy pairs  
**So that** configurations are consistent and validatable

**Acceptance Criteria:**
- Create YAML schema definition (using JSON Schema or similar)
- Define all required and optional fields
- Document configuration options with examples
- Create schema validation utility
- Support environment-specific overrides
- Include versioning for backward compatibility
- Comprehensive documentation with multiple examples

**Configuration Schema Fields:**
```yaml
# Required fields
name: string                    # Unique identifier
version: semver                 # Configuration version
indexing:
  strategy: string              # Strategy class name
  config: dict                  # Strategy-specific config
retrieval:
  strategy: string
  config: dict
services:
  embeddings: ServiceConfig
  llm: ServiceConfig
  database: DatabaseConfig

# Optional fields
description: string
migrations:
  type: "alembic"
  required_revisions: list[string]
expected_schema:
  tables: list[string]
  indexes: list[string]
  extensions: list[string]
environments:
  [env_name]: EnvironmentConfig
tags: list[string]              # For discovery/filtering
```

**Technical Notes:**
- Use `pyyaml` for parsing
- Use `jsonschema` for validation
- Support YAML includes for shared configs

**Story Points:** 8

---

## Story 17.2: Implement StrategyPair Model and Loader

**As a** developer  
**I want** a Python class to represent and load strategy pairs  
**So that** configurations can be used programmatically

**Acceptance Criteria:**
- Create `StrategyPair` dataclass/model
- Implement `StrategyPairLoader` to parse YAML files
- Support environment selection (dev/staging/prod)
- Validate configuration against schema
- Provide clear error messages for invalid configs
- Support loading from file path or YAML string
- Cache loaded configurations for performance
- Unit tests for all loading scenarios

**Implementation:**
```python
@dataclass
class StrategyPair:
    name: str
    version: str
    description: str
    indexing_config: StrategyConfig
    retrieval_config: StrategyConfig
    service_requirements: ServiceRequirements
    migration_requirements: MigrationRequirements
    schema_requirements: SchemaRequirements
    environments: Dict[str, EnvironmentConfig]
    
    def get_config_for_env(self, env: str) -> 'StrategyPair':
        """Return config merged with environment overrides"""
        pass

class StrategyPairLoader:
    def __init__(self, config_dir: str = "strategies/"):
        self.config_dir = config_dir
        self.schema_validator = SchemaValidator()
    
    def load(self, pair_name: str, environment: str = "development") -> StrategyPair:
        """Load and validate a strategy pair configuration"""
        pass
    
    def list_available_pairs(self) -> List[str]:
        """List all available strategy pair configurations"""
        pass
    
    def validate_pair(self, config_path: str) -> ValidationResult:
        """Validate a configuration without loading"""
        pass
```

**Story Points:** 8

---

## Story 17.3: Implement Migration Validator

**As a** system  
**I want** to validate that required Alembic migrations are applied  
**So that** strategy pairs only run on properly configured databases

**Acceptance Criteria:**
- Query Alembic version table to get current revision
- Check if required revisions are in the migration history
- Provide clear error messages listing missing migrations
- Support checking multiple required revisions
- Optionally suggest upgrade commands
- Handle cases where alembic_version table doesn't exist
- Integration tests with test Alembic setup
- Documentation on migration dependency management

**Implementation:**
```python
class MigrationValidator:
    def __init__(self, db_connection, alembic_config_path: str = "alembic.ini"):
        self.db = db_connection
        self.alembic_cfg = Config(alembic_config_path)
    
    def validate_migrations(self, required_revisions: List[str]) -> ValidationResult:
        """
        Check if all required migrations are applied
        Returns ValidationResult with details about missing migrations
        """
        current_rev = self._get_current_revision()
        migration_history = self._get_migration_history()
        
        missing = [r for r in required_revisions if r not in migration_history]
        
        if missing:
            return ValidationResult(
                valid=False,
                errors=[f"Missing migration: {r}" for r in missing],
                suggested_commands=[f"alembic upgrade {r}" for r in missing]
            )
        
        return ValidationResult(valid=True)
    
    def _get_current_revision(self) -> Optional[str]:
        """Get current Alembic revision from database"""
        pass
    
    def _get_migration_history(self) -> Set[str]:
        """Get all applied migrations from Alembic history"""
        pass
    
    def get_unapplied_migrations(self, required_revisions: List[str]) -> List[str]:
        """Return list of revisions that need to be applied"""
        pass
```

**Technical Dependencies:**
- Alembic must be configured and working (Epic 15)
- Database connection must be available

**Story Points:** 8

---

## Story 17.4: Implement Schema Validator

**As a** system  
**I want** to validate that expected database schema exists  
**So that** strategies don't fail due to missing tables/indexes

**Acceptance Criteria:**
- Query database information schema for tables
- Verify required tables exist
- Verify required indexes exist
- Check for required extensions (pgvector, etc.)
- Provide detailed error messages about missing elements
- Support PostgreSQL and optionally other databases
- Handle schema namespaces/schemas
- Integration tests with test database

**Implementation:**
```python
class SchemaValidator:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def validate_schema(self, requirements: SchemaRequirements) -> ValidationResult:
        """
        Verify all required schema elements exist
        Returns ValidationResult with details about missing elements
        """
        errors = []
        
        # Check tables
        for table in requirements.tables:
            if not self._table_exists(table):
                errors.append(f"Missing table: {table}")
        
        # Check indexes
        for index in requirements.indexes:
            if not self._index_exists(index):
                errors.append(f"Missing index: {index}")
        
        # Check extensions
        for ext in requirements.extensions:
            if not self._extension_exists(ext):
                errors.append(f"Missing extension: {ext}")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors
        )
    
    def _table_exists(self, table_name: str) -> bool:
        """Check if table exists in database"""
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = :table_name
            )
        """
        result = self.db.execute(sa.text(query), {"table_name": table_name})
        return result.scalar()
    
    def _index_exists(self, index_name: str) -> bool:
        """Check if index exists in database"""
        pass
    
    def _extension_exists(self, extension_name: str) -> bool:
        """Check if PostgreSQL extension is installed"""
        pass
```

**Story Points:** 5

---

## Story 17.5: Implement StrategyPairManager

**As a** user  
**I want** a high-level manager to load and initialize strategy pairs  
**So that** I can easily deploy complete RAG configurations

**Acceptance Criteria:**
- Create `StrategyPairManager` class
- Validate migrations before loading strategies
- Validate schema before loading strategies
- Instantiate indexing and retrieval strategies with proper dependencies
- Apply service configurations (LLM, embeddings)
- Support environment selection
- Provide helpful error messages for validation failures
- Cache validated configurations
- Integration tests with full workflow

**Implementation:**
```python
class StrategyPairManager:
    def __init__(
        self, 
        db_connection,
        service_registry: ServiceRegistry,
        config_dir: str = "strategies/",
        alembic_config: str = "alembic.ini"
    ):
        self.db = db_connection
        self.service_registry = service_registry
        self.loader = StrategyPairLoader(config_dir)
        self.migration_validator = MigrationValidator(db_connection, alembic_config)
        self.schema_validator = SchemaValidator(db_connection)
        self._cache = {}
    
    def load_pair(
        self, 
        pair_name: str, 
        environment: str = "development",
        validate: bool = True
    ) -> Tuple[IIndexingStrategy, IRetrievalStrategy]:
        """
        Load and initialize a strategy pair with all dependencies
        
        Args:
            pair_name: Name of the strategy pair configuration
            environment: Environment to use (dev/staging/prod)
            validate: Whether to validate migrations and schema
        
        Returns:
            Tuple of (indexing_strategy, retrieval_strategy)
        
        Raises:
            ValidationError: If migrations or schema validation fails
            ConfigurationError: If configuration is invalid
        """
        # Load configuration
        config = self.loader.load(pair_name, environment)
        
        # Validate migrations
        if validate and config.migration_requirements:
            migration_result = self.migration_validator.validate_migrations(
                config.migration_requirements.required_revisions
            )
            if not migration_result.valid:
                raise ValidationError(
                    f"Migration validation failed for '{pair_name}':\n" +
                    "\n".join(migration_result.errors) +
                    "\n\nTo fix, run:\n" +
                    "\n".join(migration_result.suggested_commands)
                )
        
        # Validate schema
        if validate and config.schema_requirements:
            schema_result = self.schema_validator.validate_schema(
                config.schema_requirements
            )
            if not schema_result.valid:
                raise ValidationError(
                    f"Schema validation failed for '{pair_name}':\n" +
                    "\n".join(schema_result.errors)
                )
        
        # Setup services based on configuration
        services = self._configure_services(config.service_requirements)
        
        # Instantiate strategies with dependencies
        indexing_strategy = self._create_indexing_strategy(
            config.indexing_config,
            services
        )
        retrieval_strategy = self._create_retrieval_strategy(
            config.retrieval_config,
            services
        )
        
        # Validate capability compatibility
        self._validate_capability_compatibility(
            indexing_strategy,
            retrieval_strategy
        )
        
        return indexing_strategy, retrieval_strategy
    
    def _configure_services(self, requirements: ServiceRequirements) -> Dict:
        """Configure services based on strategy pair requirements"""
        pass
    
    def _create_indexing_strategy(self, config: StrategyConfig, services: Dict):
        """Instantiate indexing strategy with dependencies"""
        pass
    
    def _create_retrieval_strategy(self, config: StrategyConfig, services: Dict):
        """Instantiate retrieval strategy with dependencies"""
        pass
    
    def _validate_capability_compatibility(
        self,
        indexing: IIndexingStrategy,
        retrieval: IRetrievalStrategy
    ):
        """Ensure indexing produces what retrieval requires"""
        if not retrieval.requires().issubset(indexing.produces()):
            raise CompatibilityError(
                f"Retrieval requires {retrieval.requires()} but "
                f"indexing only produces {indexing.produces()}"
            )
    
    def list_available_pairs(self) -> List[str]:
        """List all available strategy pair configurations"""
        return self.loader.list_available_pairs()
    
    def get_pair_info(self, pair_name: str) -> StrategyPair:
        """Get configuration details for a strategy pair"""
        return self.loader.load(pair_name)
```

**Usage Example:**
```python
# Initialize manager
manager = StrategyPairManager(
    db_connection=db_engine,
    service_registry=services,
    config_dir="config/strategies/"
)

# Load a strategy pair
indexing, retrieval = manager.load_pair(
    pair_name="semantic-openai-pair",
    environment="production"
)

# Use the strategies
indexing_pipeline = IndexingPipeline(indexing)
retrieval_pipeline = RetrievalPipeline(retrieval)
```

**Story Points:** 13

---

## Story 17.6: Create Pre-Built Strategy Pair Configurations

**As a** user  
**I want** pre-built strategy pair configurations for common use cases  
**So that** I can get started quickly without writing YAML

**Acceptance Criteria:**
- Create at least 5 pre-built strategy pair configurations:
  1. **semantic-openai-pair** - Semantic search with OpenAI embeddings
  2. **semantic-local-pair** - Semantic search with local ONNX embeddings
  3. **hybrid-search-pair** - Combining vector and keyword search
  4. **hierarchical-pair** - Hierarchical indexing with context expansion
  5. **graph-enhanced-pair** - Knowledge graph + vector search
- Each configuration includes comprehensive documentation
- Example usage code for each pair
- Performance characteristics documented
- Cost estimates (API calls, compute, storage)
- Recommended use cases for each pair
- All configurations validated and tested

**Deliverables:**
```
strategies/
├── semantic-openai-pair.yaml
├── semantic-local-pair.yaml
├── hybrid-search-pair.yaml
├── hierarchical-pair.yaml
├── graph-enhanced-pair.yaml
└── README.md  # Documentation for all pairs
```

**Documentation Template for Each Pair:**
```markdown
# Semantic OpenAI Pair

## Overview
Semantic search using OpenAI's text-embedding-3-small model with PostgreSQL pgvector.

## Use Cases
- General purpose semantic search
- High accuracy requirements
- Cloud-based deployments

## Requirements
- PostgreSQL 12+ with pgvector extension
- OpenAI API key
- Alembic migrations: semantic_vectors_table, vector_indexes

## Performance
- Indexing: ~1000 chunks/minute
- Retrieval: <100ms for top-k=5
- Cost: ~$0.0001 per chunk indexed

## Configuration
[Include full YAML here]

## Example Usage
[Code example]
```

**Story Points:** 5

---

## Appendix: Integration with Existing Epics

### How Strategy Pairs Relate to Previous Work

**Epic 10 (Dependency Injection):**
- Strategy pairs specify which services to inject
- Service configurations are part of the pair definition

**Epic 11 (Pipeline Separation):**
- Pairs bundle compatible indexing + retrieval strategies
- Capability validation ensures compatibility

**Epic 12/13 (Indexing/Retrieval Strategies):**
- Strategy pairs reference these concrete implementations
- Pairs specify which strategies to use together

**Epic 15 (Alembic Migrations):**
- Strategy pairs reference required Alembic revisions
- Migration validator ensures schema is ready

### Workflow: Adding a New Strategy Pair

1. Implement indexing and/or retrieval strategies (Epics 12/13)
2. Create Alembic migrations for required schema (Epic 15)
3. Create strategy pair YAML configuration (Epic 17)
4. Test with StrategyPairManager
5. Document use cases and performance characteristics

---

## Appendix: Total Story Points

**Epic 17: Strategy Pair Configuration System:**
- Story 17.1: Configuration Schema = 8 points
- Story 17.2: Model and Loader = 8 points
- Story 17.3: Migration Validator = 8 points
- Story 17.4: Schema Validator = 5 points
- Story 17.5: StrategyPairManager = 13 points
- Story 17.6: Pre-Built Configurations = 5 points

**Total:** 47 story points (~2 sprints)

**Combined Project Total:** 
- Original project (Epics 1-9): 280 points
- Epic 10 (DI): Completed ✅
- Epic 11 (Pipelines): 73 points
- Epic 12 (Indexing): 34 points
- Epic 13 (Retrieval): 21 points
- Epic 14 (CLI): 13 points
- Epic 15 (Alembic): Already exists ✅
- Epic 16 (Advanced RAG): Variable points
- **Epic 17 (Strategy Pairs): 47 points**

---

## Benefits of This Epic

1. **Portability** - Share complete RAG configurations as YAML files
2. **Reproducibility** - Guaranteed consistent deployments
3. **Safety** - Validation prevents runtime failures
4. **Documentation** - Configurations are self-documenting
5. **Testing** - Easy to test different strategy combinations
6. **Onboarding** - New users can start with pre-built pairs
7. **Evolution** - Easy to version and migrate configurations

This epic completes the RAG Factory's transformation into a true enterprise-grade experimentation and deployment platform.
