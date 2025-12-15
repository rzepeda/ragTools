# Epic 17: Strategy Pair Configuration System

**Epic Goal:** Create a configuration system that bundles compatible indexing and retrieval strategies with their required services, database table mappings, and migrations, enabling portable and reproducible RAG deployments while leveraging shared service instances.

**Epic Story Points Total:** 55

**Dependencies:** 
- Epic 11 (Dependency Injection - service interfaces must exist) ✅
- Epic 12 (Pipeline Separation - capability system must exist) ✅
- Epic 16 (Database Consolidation - Alembic must be working) ✅

---

## Background

The RAG Factory currently has (from previous epics):
- ✅ Service interfaces: `ILLMService`, `IEmbeddingService`, `IDatabaseService` (Epic 11)
- ✅ Dependency injection with `StrategyDependencies` (Epic 11)
- ✅ Capability-based validation with `IndexCapability` enums (Epic 12)
- ✅ Separate indexing/retrieval pipelines (Epic 12)
- ✅ Alembic migration system (Epic 16)
- ✅ Database service with connection pooling (Epic 16)

**The Problem We're Solving:**

While strategies are compatible at the **interface level**, they have **implementation-level incompatibilities**:

1. **Different Database Schemas**: 
   - Semantic search needs `vector_embeddings` table with pgvector
   - Keyword search needs `inverted_index` table with GIN indexes
   - Graph search needs entities/relationships tables

2. **Different Service Configurations**:
   - Some strategies use local ONNX embeddings
   - Others use OpenAI API embeddings
   - Can't mix - embedding dimensions must match

3. **Service Sharing Required**:
   - Multiple strategies should share the same embedding service (memory/performance)
   - Same LLM instance used across strategies (cost/efficiency)
   - Same database connection pool (resource limits)

4. **Table Name Conflicts**:
   - Strategies can't all use `chunks` table - they need isolation
   - But they share the same database connection pool
   - Need strategy-specific table mappings

**The Solution:** 

**Strategy Pairs with Service Registry** - A system that:
1. Defines services once, shares them across strategies
2. Gives each strategy its own table/field mappings on shared DB
3. References existing Alembic migrations (not duplicating them)
4. Supports environment variable resolution (`${VAR}`)
5. Validates that required services and migrations exist

---

## Core Concepts

### 1. Service Registry - Share Services Across Strategies

A **Service Registry** is a centralized place where services are defined once and shared by multiple strategies.

**Problem without Service Registry:**
```python
# Each strategy loads its own embedding model - wastes memory!
semantic_indexer = VectorIndexer(embedding=EmbeddingService("model.onnx"))  # Loads model
keyword_indexer = KeywordIndexer(embedding=EmbeddingService("model.onnx"))   # Loads AGAIN
hybrid_indexer = HybridIndexer(embedding=EmbeddingService("model.onnx"))     # Loads AGAIN
```

**Solution with Service Registry:**
```python
# Load once, share everywhere
embedding_service = registry.get("embedding1")  # Loads model once
semantic_indexer = VectorIndexer(embedding=embedding_service)  # Reuses
keyword_indexer = KeywordIndexer(embedding=embedding_service)   # Reuses
hybrid_indexer = HybridIndexer(embedding=embedding_service)     # Reuses
```

**Service Registry Configuration:**
```yaml
# config/services.yaml - Define services once
services:
  llm1:
    name: "local-llama"
    url: "http://192.168.56.1:1234/v1"
    api_key: "${LM_STUDIO_API_KEY}"  # Environment variable
    model: "llama-3.2-3b"
    temperature: 0.7
  
  embedding1:
    name: "local-onnx-minilm"
    provider: "onnx"
    model: "Xenova/all-MiniLM-L6-v2"
    cache_dir: "./models/embeddings"
    batch_size: 32
    # Note: Tokenizer is bundled with embedding model (kept coupled)
  
  db1:
    name: "main-postgres"
    type: "postgres"
    connection_string: "${DATABASE_URL}"
    pool_size: 10
    max_overflow: 20
```

### 2. Strategy Pairs - Bundle Compatible Strategies

A **Strategy Pair** configuration references services and specifies table mappings:

```yaml
# strategies/semantic-pair.yaml
strategy_name: "semantic-search-pair"
version: "1.0.0"

indexer:
  strategy: "VectorEmbeddingIndexer"
  services:
    embedding: "$embedding1"  # References service registry
    db: "$db1"
  
  # Strategy-specific table mapping on shared DB
  db_config:
    tables:
      chunks: "semantic_chunks"
      vectors: "semantic_vectors"
      metadata: "semantic_metadata"
    fields:
      content: "text_content"
      embedding: "vector_embedding"
      doc_id: "document_id"
  
  config:
    chunk_size: 512
    overlap: 50

retriever:
  strategy: "SemanticRetriever"
  services:
    embedding: "$embedding1"  # Same embedding service as indexer
    db: "$db1"               # Same database connection pool
  
  db_config:
    tables:
      vectors: "semantic_vectors"  # Same table as indexer
    fields:
      embedding: "vector_embedding"
      similarity_threshold: 0.7
  
  config:
    top_k: 5

# References existing Alembic migrations (Epic 16)
migrations:
  required_revisions:
    - "abc123_semantic_vectors_table"
    - "def456_vector_indexes"

# Expected schema after migrations
expected_schema:
  tables:
    - "semantic_chunks"
    - "semantic_vectors"
  indexes:
    - "idx_semantic_vectors_embedding"
  extensions:
    - "vector"  # pgvector
```

### 3. Database Context - Strategy-Specific Table Access

**Problem:** Multiple strategies sharing one database but needing different tables.

**Solution:** Each strategy gets a `DatabaseContext` that maps logical names to physical tables:

```python
# Both strategies use SAME connection pool
db_service = registry.get("db1")  # PostgreSQL connection pool

# But each gets its own table mapping
semantic_context = db_service.get_context({
    "chunks": "semantic_chunks",
    "vectors": "semantic_vectors"
})

keyword_context = db_service.get_context({
    "chunks": "keyword_chunks", 
    "inverted_index": "keyword_index"
})

# Strategies use logical names
semantic_context.insert("chunks", {...})  # Inserts into semantic_chunks
keyword_context.insert("chunks", {...})   # Inserts into keyword_chunks
```

### 4. Environment Variable Resolution

Strategy configurations support `${VAR}` syntax:

```yaml
services:
  llm1:
    api_key: "${LM_STUDIO_API_KEY}"  # Resolved from .env
  
  db1:
    connection_string: "${DATABASE_URL}"
```

```bash
# .env file
LM_STUDIO_API_KEY=sk-local-key
DATABASE_URL=postgresql://user:pass@192.168.56.1:5432/rag_factory
```

---

## Story 17.1: Design Service Registry and Configuration Schema

**As a** developer  
**I want** well-defined YAML schemas for services and strategy pairs  
**So that** configurations are consistent, validatable, and support environment variables

**Acceptance Criteria:**
- Define service registry YAML schema (services.yaml)
- Define strategy pair YAML schema (strategy-pair.yaml)
- Support environment variable resolution with `${VAR}` syntax
- Support service references with `$service_name` syntax
- Include versioning for backward compatibility
- Comprehensive documentation with examples
- Schema validation utilities
- Support for inline service configuration (optional)

**Service Registry Schema:**
```yaml
# config/services.yaml
services:
  [service_name]:
    name: string                    # Human-readable name
    # LLM Service
    url: string (optional)          # For HTTP-based services
    api_key: string (optional)      # Can use ${ENV_VAR}
    model: string (optional)
    temperature: float (optional)
    max_tokens: int (optional)
    timeout: int (optional)
    # Embedding Service
    provider: string (optional)     # "onnx", "openai", "cohere"
    cache_dir: string (optional)
    batch_size: int (optional)
    dimensions: int (optional)
    # Database Service
    type: string (optional)         # "postgres", "neo4j"
    connection_string: string (optional)
    host: string (optional)
    port: int (optional)
    database: string (optional)
    user: string (optional)
    password: string (optional)
    pool_size: int (optional)
    max_overflow: int (optional)
```

**Strategy Pair Schema:**
```yaml
# strategies/[pair-name].yaml
strategy_name: string               # Unique identifier
version: semver                     # Configuration version
description: string (optional)

indexer:
  strategy: string                  # Class name
  services:
    [service_type]: string          # "$service_name" reference or inline config
  db_config:
    tables:
      [logical_name]: string        # Physical table name
    fields:
      [logical_name]: string        # Physical field name
  config:
    [key]: value                    # Strategy-specific config

retriever:
  strategy: string
  services:
    [service_type]: string
  db_config:
    tables:
      [logical_name]: string
    fields:
      [logical_name]: string
  config:
    [key]: value

migrations:
  required_revisions: list[string]  # Alembic revision IDs

expected_schema:
  tables: list[string]
  indexes: list[string]
  extensions: list[string]

tags: list[string] (optional)       # For discovery/filtering
```

**Environment Variable Resolution Rules:**
1. `${VAR_NAME}` - Must exist in environment, error if missing
2. `${VAR_NAME:-default}` - Use default if not in environment
3. `${VAR_NAME:?error message}` - Custom error if missing

**Service Reference Rules:**
1. `$service_name` or `"$service_name"` - Reference from registry
2. Inline dict `{provider: "onnx", ...}` - Create one-off service instance
3. Validation: Referenced services must exist in registry

**Technical Notes:**
- Use `pyyaml` for parsing
- Use `jsonschema` for validation
- Use `os.path.expandvars()` for ${VAR} resolution
- Create `ConfigValidator` class for schema checking

**Story Points:** 8

---

## Story 17.2: Implement Service Registry

**As a** developer  
**I want** a service registry that instantiates and caches service instances  
**So that** multiple strategies can share the same service instances efficiently

**Acceptance Criteria:**
- Create `ServiceRegistry` class that loads services.yaml
- Instantiate services lazily (only when first requested)
- Cache service instances (singleton per service definition)
- Thread-safe instantiation with locks
- Support environment variable resolution in config
- Integrate with existing service interfaces from Epic 11
- Clear error messages for missing services or environment variables
- Unit tests with various service types
- Integration tests with shared services

**Implementation:**

```python
from threading import Lock
from collections import defaultdict
from typing import Dict, Any
import os
import yaml

# Import existing service interfaces from Epic 11
from rag_factory.services import (
    ILLMService,
    IEmbeddingService,
    IDatabaseService
)

# Import existing service implementations from Epic 11
from rag_factory.services.llm import LMStudioLLMService, OpenAILLMService
from rag_factory.services.embedding import ONNXEmbeddingService
from rag_factory.services.database import PostgresqlDatabaseService


class ServiceRegistry:
    """
    Central registry for service definitions and instances.
    
    Loads service configurations from YAML and creates/caches service instances.
    Multiple strategies can share the same service instance for efficiency.
    """
    
    def __init__(self, config_path: str = "config/services.yaml"):
        """
        Initialize service registry from configuration file.
        
        Args:
            config_path: Path to services.yaml configuration
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._instances: Dict[str, Any] = {}  # service_name -> instance
        self._locks = defaultdict(Lock)  # service_name -> lock
    
    def _load_config(self) -> dict:
        """Load and parse services.yaml"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required structure
        if 'services' not in config:
            raise ValueError(f"Invalid services config: missing 'services' key")
        
        return config
    
    def get(self, service_ref: str) -> Any:
        """
        Get or create a service instance.
        
        Args:
            service_ref: Service reference like "$llm1" or "llm1"
        
        Returns:
            Service instance implementing appropriate interface
        
        Raises:
            KeyError: If service not found in registry
            ValueError: If service configuration is invalid
        """
        # Strip $ prefix if present
        service_name = service_ref.lstrip('$')
        
        # Return cached instance if exists
        if service_name in self._instances:
            return self._instances[service_name]
        
        # Thread-safe instantiation
        with self._locks[service_name]:
            # Double-check after acquiring lock
            if service_name in self._instances:
                return self._instances[service_name]
            
            # Get service configuration
            if service_name not in self.config['services']:
                raise KeyError(
                    f"Service '{service_name}' not found in registry. "
                    f"Available services: {list(self.config['services'].keys())}"
                )
            
            service_config = self.config['services'][service_name]
            
            # Resolve environment variables
            resolved_config = self._resolve_env_vars(service_config)
            
            # Create service instance
            service_instance = self._create_service(service_name, resolved_config)
            
            # Cache and return
            self._instances[service_name] = service_instance
            return service_instance
    
    def _resolve_env_vars(self, config: dict) -> dict:
        """
        Recursively resolve ${VAR} in configuration values.
        
        Supports:
        - ${VAR} - Required variable
        - ${VAR:-default} - Optional with default
        - ${VAR:?error message} - Required with custom error
        """
        resolved = {}
        
        for key, value in config.items():
            if isinstance(value, str) and '${' in value:
                resolved[key] = self._expand_var(value)
            elif isinstance(value, dict):
                resolved[key] = self._resolve_env_vars(value)
            else:
                resolved[key] = value
        
        return resolved
    
    def _expand_var(self, value: str) -> str:
        """Expand environment variable with default/error handling"""
        import re
        
        # Pattern: ${VAR}, ${VAR:-default}, ${VAR:?error}
        pattern = r'\$\{([^}:]+)(?::-([^}]+)|:\?([^}]+))?\}'
        
        def replacer(match):
            var_name = match.group(1)
            default = match.group(2)
            error_msg = match.group(3)
            
            env_value = os.getenv(var_name)
            
            if env_value is not None:
                return env_value
            elif default is not None:
                return default
            elif error_msg is not None:
                raise ValueError(f"Environment variable ${{{var_name}}}: {error_msg}")
            else:
                raise ValueError(
                    f"Required environment variable ${{{var_name}}} is not set"
                )
        
        return re.sub(pattern, replacer, value)
    
    def _create_service(self, service_name: str, config: dict) -> Any:
        """
        Factory method to create service instances based on configuration.
        
        Uses existing service implementations from Epic 11.
        """
        # Determine service type
        if 'url' in config and 'model' in config:
            # LLM Service (HTTP-based)
            if 'openai.com' in config.get('url', ''):
                return OpenAILLMService(
                    api_key=config['api_key'],
                    model=config['model'],
                    temperature=config.get('temperature', 0.7),
                    max_tokens=config.get('max_tokens')
                )
            else:
                # LM Studio or other OpenAI-compatible
                return LMStudioLLMService(
                    base_url=config['url'],
                    api_key=config.get('api_key', 'not-needed'),
                    model=config['model'],
                    temperature=config.get('temperature', 0.7),
                    timeout=config.get('timeout', 30)
                )
        
        elif config.get('provider') == 'onnx':
            # ONNX Embedding Service
            return ONNXEmbeddingService(
                model_name=config['model'],
                cache_dir=config.get('cache_dir', './models'),
                batch_size=config.get('batch_size', 32)
            )
        
        elif config.get('type') == 'postgres':
            # PostgreSQL Database Service (from Epic 16)
            if 'connection_string' in config:
                conn_str = config['connection_string']
            else:
                conn_str = (
                    f"postgresql://{config['user']}:{config['password']}"
                    f"@{config['host']}:{config.get('port', 5432)}"
                    f"/{config['database']}"
                )
            
            return PostgresqlDatabaseService(
                connection_string=conn_str,
                pool_size=config.get('pool_size', 10),
                max_overflow=config.get('max_overflow', 20)
            )
        
        else:
            raise ValueError(
                f"Cannot determine service type for '{service_name}'. "
                f"Configuration: {config}"
            )
    
    def list_services(self) -> list[str]:
        """List all available service names"""
        return list(self.config['services'].keys())
    
    def reload(self, service_name: str) -> Any:
        """
        Force reload a service (useful after config changes).
        
        Closes old instance if it has a close() method.
        """
        service_name = service_name.lstrip('$')
        
        if service_name in self._instances:
            # Cleanup old instance
            old_instance = self._instances[service_name]
            if hasattr(old_instance, 'close'):
                old_instance.close()
            
            # Remove from cache
            del self._instances[service_name]
        
        # Next get() will create new instance
        return self.get(service_name)
    
    def shutdown(self):
        """Close all service instances"""
        for instance in self._instances.values():
            if hasattr(instance, 'close'):
                instance.close()
        
        self._instances.clear()
```

**Usage Example:**
```python
# Load service registry
registry = ServiceRegistry("config/services.yaml")

# Get services (creates on first call, returns cached on subsequent calls)
llm = registry.get("$llm1")           # Creates LLM service
embedding = registry.get("embedding1")  # Creates embedding service
db = registry.get("db1")               # Creates database service

# Multiple strategies share same instances
strategy1 = Strategy1(llm=llm, embedding=embedding, db=db)
strategy2 = Strategy2(llm=llm, embedding=embedding, db=db)  # Same instances!

# Check that they're actually the same object
assert strategy1.llm is strategy2.llm  # True - same instance
```

**Integration with Epic 11:**
- Uses existing `ILLMService`, `IEmbeddingService`, `IDatabaseService` interfaces
- Uses existing service implementations (ONNXEmbeddingService, PostgresqlDatabaseService, etc.)
- No changes needed to existing service code

**Story Points:** 13

---

## Story 17.3: Implement DatabaseContext for Table Mapping

**As a** developer  
**I want** each strategy to access strategy-specific tables on a shared database  
**So that** strategies are isolated but share connection pools efficiently

**Acceptance Criteria:**
- Extend `PostgresqlDatabaseService` from Epic 16 with `get_context()` method
- Create `DatabaseContext` class with table/field mapping
- Support logical-to-physical table name translation
- Support logical-to-physical field name translation
- Provide CRUD operations using logical names
- Support vector search operations (for pgvector)
- Cache reflected table metadata
- Unit tests with multiple contexts on same database
- Integration tests showing isolation between strategies

**Implementation:**

```python
from sqlalchemy import MetaData, Table, select, insert, update, delete
from sqlalchemy.engine import Engine

class DatabaseContext:
    """
    Strategy-specific view of a shared database.
    
    Provides access to tables using logical names that map to
    physical table names, enabling strategy isolation.
    """
    
    def __init__(self, engine: Engine, table_mapping: dict, field_mapping: dict = None):
        """
        Create database context with table/field mappings.
        
        Args:
            engine: SQLAlchemy engine (shared across contexts)
            table_mapping: Dict mapping logical → physical table names
                          e.g., {"chunks": "semantic_chunks"}
            field_mapping: Optional dict mapping logical → physical field names
                          e.g., {"content": "text_content"}
        """
        self.engine = engine
        self.tables = table_mapping
        self.fields = field_mapping or {}
        self._metadata = MetaData()
        self._reflected_tables = {}
    
    def get_table(self, logical_name: str) -> Table:
        """
        Get SQLAlchemy Table object by logical name.
        
        Args:
            logical_name: Logical table name like "chunks" or "vectors"
        
        Returns:
            Reflected Table object for physical table
        
        Raises:
            KeyError: If logical_name not in table_mapping
        """
        if logical_name not in self.tables:
            available = list(self.tables.keys())
            raise KeyError(
                f"No table mapping for '{logical_name}'. "
                f"Available: {available}"
            )
        
        physical_name = self.tables[logical_name]
        
        # Reflect table structure if not cached
        if physical_name not in self._reflected_tables:
            self._reflected_tables[physical_name] = Table(
                physical_name,
                self._metadata,
                autoload_with=self.engine
            )
        
        return self._reflected_tables[physical_name]
    
    def _map_field(self, logical_field: str) -> str:
        """Map logical field name to physical field name"""
        return self.fields.get(logical_field, logical_field)
    
    def insert(self, logical_table: str, data: dict):
        """
        Insert row into a logically-named table.
        
        Args:
            logical_table: Logical table name
            data: Dict with logical field names as keys
        """
        table = self.get_table(logical_table)
        
        # Map logical field names to physical field names
        physical_data = {
            self._map_field(k): v for k, v in data.items()
        }
        
        with self.engine.begin() as conn:
            conn.execute(insert(table).values(**physical_data))
    
    def query(
        self,
        logical_table: str,
        filters: dict = None,
        limit: int = None
    ) -> list:
        """
        Query a logically-named table.
        
        Args:
            logical_table: Logical table name
            filters: Optional dict of logical_field → value
            limit: Optional row limit
        
        Returns:
            List of row results
        """
        table = self.get_table(logical_table)
        query = select(table)
        
        # Apply filters (map logical to physical field names)
        if filters:
            for logical_field, value in filters.items():
                physical_field = self._map_field(logical_field)
                query = query.where(table.c[physical_field] == value)
        
        if limit:
            query = query.limit(limit)
        
        with self.engine.connect() as conn:
            result = conn.execute(query)
            return result.fetchall()
    
    def update(
        self,
        logical_table: str,
        filters: dict,
        updates: dict
    ):
        """
        Update rows in a logically-named table.
        
        Args:
            logical_table: Logical table name
            filters: Dict of logical_field → value for WHERE clause
            updates: Dict of logical_field → new_value for SET clause
        """
        table = self.get_table(logical_table)
        stmt = update(table)
        
        # Apply WHERE filters
        for logical_field, value in filters.items():
            physical_field = self._map_field(logical_field)
            stmt = stmt.where(table.c[physical_field] == value)
        
        # Apply SET updates
        physical_updates = {
            self._map_field(k): v for k, v in updates.items()
        }
        stmt = stmt.values(**physical_updates)
        
        with self.engine.begin() as conn:
            conn.execute(stmt)
    
    def delete(self, logical_table: str, filters: dict):
        """Delete rows from a logically-named table"""
        table = self.get_table(logical_table)
        stmt = delete(table)
        
        for logical_field, value in filters.items():
            physical_field = self._map_field(logical_field)
            stmt = stmt.where(table.c[physical_field] == value)
        
        with self.engine.begin() as conn:
            conn.execute(stmt)
    
    def vector_search(
        self,
        logical_table: str,
        vector_field: str,
        query_vector: list[float],
        top_k: int = 5,
        distance_metric: str = "cosine"
    ) -> list:
        """
        Perform vector similarity search (for pgvector).
        
        Args:
            logical_table: Logical table name
            vector_field: Logical field name containing vectors
            query_vector: Query vector
            top_k: Number of results
            distance_metric: "cosine", "l2", or "inner_product"
        
        Returns:
            List of (row, distance) tuples
        """
        table = self.get_table(logical_table)
        physical_vector_field = self._map_field(vector_field)
        
        # pgvector distance functions
        if distance_metric == "cosine":
            distance_func = table.c[physical_vector_field].cosine_distance
        elif distance_metric == "l2":
            distance_func = table.c[physical_vector_field].l2_distance
        elif distance_metric == "inner_product":
            distance_func = table.c[physical_vector_field].max_inner_product
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        
        query = select(
            table,
            distance_func(query_vector).label('distance')
        ).order_by('distance').limit(top_k)
        
        with self.engine.connect() as conn:
            result = conn.execute(query)
            return result.fetchall()


# Extend PostgresqlDatabaseService from Epic 16
class PostgresqlDatabaseService(IDatabaseService):
    """
    PostgreSQL database service with connection pooling.
    Extended to support strategy-specific table contexts.
    """
    
    def __init__(self, connection_string: str, pool_size: int = 10, max_overflow: int = 20):
        from sqlalchemy import create_engine
        
        self.connection_string = connection_string
        self.engine = create_engine(
            connection_string,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True  # Verify connections before using
        )
        self._contexts = {}  # Cache contexts
    
    def get_context(
        self,
        table_mapping: dict,
        field_mapping: dict = None
    ) -> DatabaseContext:
        """
        Create strategy-specific database context.
        
        Args:
            table_mapping: Dict mapping logical → physical table names
            field_mapping: Optional dict mapping logical → physical field names
        
        Returns:
            DatabaseContext with specified mappings
        """
        # Create unique key from mappings
        table_key = frozenset(table_mapping.items())
        field_key = frozenset(field_mapping.items()) if field_mapping else frozenset()
        cache_key = (table_key, field_key)
        
        if cache_key not in self._contexts:
            self._contexts[cache_key] = DatabaseContext(
                engine=self.engine,  # Shared engine
                table_mapping=table_mapping,
                field_mapping=field_mapping
            )
        
        return self._contexts[cache_key]
    
    def close(self):
        """Close database connection pool"""
        self.engine.dispose()
    
    # ... other IDatabaseService methods from Epic 16 ...
```

**Usage Example:**
```python
# Get shared database service from registry
db_service = registry.get("db1")

# Strategy 1: Semantic search with its own tables
semantic_context = db_service.get_context(
    table_mapping={
        "chunks": "semantic_chunks",
        "vectors": "semantic_vectors"
    },
    field_mapping={
        "content": "text_content",
        "embedding": "vector_embedding"
    }
)

# Strategy 2: Keyword search with different tables
keyword_context = db_service.get_context(
    table_mapping={
        "chunks": "keyword_chunks",
        "index": "keyword_inverted_index"
    }
)

# Both use same connection pool
assert semantic_context.engine is keyword_context.engine  # True

# But access different tables
semantic_context.insert("chunks", {"content": "hello", "doc_id": "123"})
# Inserts into semantic_chunks.text_content

keyword_context.insert("chunks", {"content": "hello", "doc_id": "123"})
# Inserts into keyword_chunks.content (different table!)
```

**Story Points:** 13

---

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

## Story 17.4: Implement Migration Validator (Alembic Integration)

**As a** system  
**I want** to validate that required Alembic migrations are applied  
**So that** strategy pairs only run on properly configured databases

**Acceptance Criteria:**
- Query `alembic_version` table from Epic 16's Alembic setup
- Check if required revision IDs are in migration history
- Provide clear error messages listing missing migrations
- Suggest `alembic upgrade` commands for missing revisions
- Handle case where alembic_version table doesn't exist
- Integration tests with test Alembic migrations

**Implementation:**
```python
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
import sqlalchemy as sa

class MigrationValidator:
    """Validates Alembic migrations against strategy pair requirements"""
    
    def __init__(self, db_service: DatabaseService, alembic_config_path: str = "alembic.ini"):
        self.db = db_service
        self.alembic_cfg = Config(alembic_config_path)
        self.script_dir = ScriptDirectory.from_config(self.alembic_cfg)
    
    def validate(self, required_revisions: list[str]) -> tuple[bool, list[str]]:
        """
        Check if required migrations are applied.
        
        Returns:
            (is_valid, missing_revisions)
        """
        current_revision = self._get_current_revision()
        
        if not current_revision:
            return (False, required_revisions)
        
        # Get all revisions between base and current
        applied_revisions = self._get_applied_revisions(current_revision)
        
        # Check which required revisions are missing
        missing = [r for r in required_revisions if r not in applied_revisions]
        
        return (len(missing) == 0, missing)
    
    def _get_current_revision(self) -> str:
        """Get current revision from alembic_version table"""
        with self.db.engine.connect() as conn:
            context = MigrationContext.configure(conn)
            return context.get_current_revision()
    
    def _get_applied_revisions(self, current_revision: str) -> set[str]:
        """Get all revisions up to current"""
        revisions = set()
        for rev in self.script_dir.iterate_revisions(current_revision, "base"):
            revisions.add(rev.revision)
        return revisions
```

**Story Points:** 5

---

## Story 17.5: Implement StrategyPair Loader and StrategyPairManager

**As a** user  
**I want** a high-level manager to load and initialize complete strategy pairs  
**So that** I can deploy RAG configurations with one command

**Acceptance Criteria:**
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

**Implementation:**
```python
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

**Usage:**
```python
registry = ServiceRegistry("config/services.yaml")
manager = StrategyPairManager(registry, config_dir="strategies/")

# Load complete strategy pair
indexing, retrieval = manager.load_pair("semantic-pair")

# Use immediately
indexing.index(documents)
results = retrieval.retrieve(query)
```

**Story Points:** 13

---

## Story 17.6: Create First Strategy Pair Configuration (Testing & CLI Integration)

**As a** developer  
**I want** a single, complete strategy pair configuration  
**So that** I can test the entire Epic 17 implementation and integrate with CLI (Epic 14)

**Acceptance Criteria:**
- Create ONE fully-functional strategy pair: `semantic-local-pair.yaml`
- Uses local ONNX services (no API keys needed for testing)
- Complete `services.yaml` with all required services
- All required Alembic migrations documented and created
- Comprehensive README with step-by-step setup instructions
- Example usage code demonstrating full workflow
- Unit tests validating the configuration loads correctly
- Integration tests showing end-to-end indexing and retrieval
- CLI integration tests (coordinate with Epic 14)
- Performance benchmarks documented

**Why This Strategy First:**
- ✅ No API keys required (uses local ONNX models)
- ✅ Fast to test (small model, quick loading)
- ✅ Common use case (semantic search is fundamental)
- ✅ Tests all Epic 17 components (ServiceRegistry, DatabaseContext, etc.)
- ✅ Can be used immediately by CLI for smoke tests

**Deliverables:**

```
config/
└── services.yaml              # Service definitions

strategies/
├── semantic-local-pair.yaml   # The test strategy pair
└── README.md                  # Quick start guide

migrations/versions/
└── xxxx_semantic_local_schema.py  # Alembic migration

tests/integration/
└── test_semantic_local_pair.py    # End-to-end tests

docs/
└── semantic-local-pair-guide.md   # Comprehensive guide
```

**Configuration Files:**

**config/services.yaml:**
```yaml
# Service Registry - Local ONNX Services (No API Keys Needed)
services:
  embedding_local:
    name: "local-onnx-minilm"
    provider: "onnx"
    model: "Xenova/all-MiniLM-L6-v2"
    cache_dir: "./models/embeddings"
    batch_size: 32
    dimensions: 384
  
  db_main:
    name: "main-postgres"
    type: "postgres"
    connection_string: "${DATABASE_URL}"
    pool_size: 10
    max_overflow: 20
```

**strategies/semantic-local-pair.yaml:**
```yaml
strategy_name: "semantic-local-pair"
version: "1.0.0"
description: "Semantic search using local ONNX embeddings (no API keys required)"

indexer:
  strategy: "VectorEmbeddingIndexer"
  services:
    embedding: "$embedding_local"
    db: "$db_main"
  
  db_config:
    tables:
      chunks: "semantic_local_chunks"
      vectors: "semantic_local_vectors"
      metadata: "semantic_local_metadata"
    fields:
      content: "text_content"
      embedding: "vector_embedding"
      doc_id: "document_id"
      chunk_id: "chunk_id"
  
  config:
    chunk_size: 512
    overlap: 50
    min_chunk_size: 100

retriever:
  strategy: "SemanticRetriever"
  services:
    embedding: "$embedding_local"  # Same service as indexer
    db: "$db_main"                # Same database
  
  db_config:
    tables:
      vectors: "semantic_local_vectors"
    fields:
      embedding: "vector_embedding"
      content: "text_content"
  
  config:
    top_k: 5
    similarity_threshold: 0.7
    distance_metric: "cosine"

migrations:
  required_revisions:
    - "semantic_local_schema"  # Alembic revision ID

expected_schema:
  tables:
    - "semantic_local_chunks"
    - "semantic_local_vectors"
    - "semantic_local_metadata"
  indexes:
    - "idx_semantic_local_vectors_embedding"
  extensions:
    - "vector"  # pgvector

tags:
  - "semantic"
  - "local"
  - "onnx"
  - "testing"
```

**Alembic Migration:**
```python
# migrations/versions/xxxx_semantic_local_schema.py
"""Create semantic local tables

Revision ID: semantic_local_schema
Revises: base_schema
Create Date: 2024-12-15
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = 'semantic_local_schema'
down_revision = 'base_schema'

def upgrade():
    # Chunks table
    op.create_table(
        'semantic_local_chunks',
        sa.Column('chunk_id', sa.String(255), primary_key=True),
        sa.Column('document_id', sa.String(255), nullable=False),
        sa.Column('text_content', sa.Text(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now())
    )
    
    # Vectors table
    op.create_table(
        'semantic_local_vectors',
        sa.Column('id', sa.UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('chunk_id', sa.String(255), sa.ForeignKey('semantic_local_chunks.chunk_id')),
        sa.Column('vector_embedding', Vector(384)),  # MiniLM dimension
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now())
    )
    
    # Metadata table
    op.create_table(
        'semantic_local_metadata',
        sa.Column('document_id', sa.String(255), primary_key=True),
        sa.Column('title', sa.String(500)),
        sa.Column('source', sa.String(255)),
        sa.Column('metadata', sa.JSON()),
        sa.Column('indexed_at', sa.TIMESTAMP(), server_default=sa.func.now())
    )
    
    # Index for fast vector search
    op.create_index(
        'idx_semantic_local_vectors_embedding',
        'semantic_local_vectors',
        ['vector_embedding'],
        postgresql_using='ivfflat',
        postgresql_ops={'vector_embedding': 'vector_cosine_ops'},
        postgresql_with={'lists': 100}
    )

def downgrade():
    op.drop_table('semantic_local_vectors')
    op.drop_table('semantic_local_chunks')
    op.drop_table('semantic_local_metadata')
```

**Example Usage Code:**
```python
# examples/semantic_local_example.py
"""
Quick start example for semantic-local-pair.
Tests Epic 17 implementation end-to-end.
"""
from rag_factory.registry import ServiceRegistry
from rag_factory.manager import StrategyPairManager

# Step 1: Initialize service registry
registry = ServiceRegistry("config/services.yaml")

# Step 2: Initialize strategy pair manager
manager = StrategyPairManager(
    service_registry=registry,
    config_dir="strategies/"
)

# Step 3: Load the strategy pair (validates migrations, instantiates strategies)
indexing, retrieval = manager.load_pair("semantic-local-pair")

# Step 4: Index some documents
documents = [
    {"id": "doc1", "content": "Python is a programming language"},
    {"id": "doc2", "content": "Machine learning uses neural networks"},
    {"id": "doc3", "content": "Embeddings represent text as vectors"}
]

print("Indexing documents...")
indexing.index(documents)
print("✅ Indexing complete")

# Step 5: Query
query = "What is machine learning?"
print(f"\nQuery: {query}")

results = retrieval.retrieve(query, top_k=3)
print(f"✅ Found {len(results)} results")

for i, result in enumerate(results):
    print(f"\n{i+1}. Score: {result.score:.3f}")
    print(f"   Content: {result.content[:100]}...")

# Step 6: Verify service sharing (important test!)
assert indexing.embedding is retrieval.embedding, "Services should be shared!"
print("\n✅ Service sharing verified - same embedding instance used")
```

**CLI Integration (for Epic 14):**
```bash
# Test the strategy pair via CLI
rag-factory index \
  --pair semantic-local-pair \
  --path ./test-docs \
  --verbose

rag-factory query \
  --pair semantic-local-pair \
  --query "What is machine learning?" \
  --top-k 5

# Validate configuration
rag-factory validate-pair semantic-local-pair

# Check service registry
rag-factory list-services
```

**README.md Quick Start:**
```markdown
# Semantic Local Pair - Quick Start

## Prerequisites
1. PostgreSQL with pgvector extension
2. Python 3.10+
3. No API keys needed!

## Setup (5 minutes)

### 1. Install dependencies
```bash
pip install rag-factory
```

### 2. Configure environment
```bash
# .env
DATABASE_URL=postgresql://user:pass@localhost:5432/rag_factory
```

### 3. Run migrations
```bash
alembic upgrade semantic_local_schema
```

### 4. Test it!
```bash
python examples/semantic_local_example.py
```

## What This Tests
✅ ServiceRegistry loads and caches services  
✅ DatabaseContext provides table isolation  
✅ Migration validation works  
✅ Service sharing works (memory efficient)  
✅ End-to-end indexing and retrieval  
✅ CLI integration (Epic 14)

## Performance
- Indexing: ~500 chunks/sec (local ONNX)
- Retrieval: ~50ms (top_k=5)
- Memory: ~500MB (model + overhead)
- Cost: $0 (fully local)
```

**Testing Requirements:**
- Unit tests for configuration loading
- Integration tests for end-to-end workflow
- CLI smoke tests (coordinate with Epic 14)
- Performance benchmarks
- Memory usage profiling
- Error handling validation

**Story Points:** 5

---

## Story 17.8: End-to-End CLI Validation with Sample Documents

**As a** developer  
**I want** to index 3 sample documents and retrieve information using the CLI  
**So that** I can validate the complete system works end-to-end with minimal setup

**Acceptance Criteria:**
- Create 3 sample documents with known content (for testing retrieval)
- Create CLI configuration file (`cli-config.yaml`)
- CLI reads configuration from YAML file (not command-line args)
- Use `semantic-local-pair` from Story 17.6 (no API keys needed)
- Index all 3 documents via CLI
- Perform 5 test queries via CLI
- Verify correct documents are retrieved for each query
- Document the complete workflow in a tutorial
- Create automated test script that validates the workflow
- Measure and document performance metrics

**Why This Story Is Important:**
- ✅ **Proves Epic 17 works**: Complete system validation
- ✅ **Proves Epic 14 works**: CLI can use strategy pairs
- ✅ **User-facing validation**: Not just unit tests, real usage
- ✅ **Documentation artifact**: Tutorial becomes user guide
- ✅ **Onboarding tool**: New users can follow this exact workflow

**Sample Documents:**

```
sample-docs/
├── python_basics.txt
├── machine_learning.txt
└── embeddings_explained.txt
```

**sample-docs/python_basics.txt:**
```
Python Programming Basics

Python is a high-level, interpreted programming language known for its 
clear syntax and readability. It was created by Guido van Rossum and 
first released in 1991.

Key Features:
- Easy to learn and use
- Extensive standard library
- Supports multiple programming paradigms
- Large community and ecosystem

Common Uses:
- Web development (Django, Flask)
- Data science and machine learning
- Automation and scripting
- Scientific computing
```

**sample-docs/machine_learning.txt:**
```
Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience without being explicitly 
programmed. It focuses on the development of computer programs that can 
access data and use it to learn for themselves.

Types of Machine Learning:
1. Supervised Learning - Training with labeled data
2. Unsupervised Learning - Finding patterns in unlabeled data
3. Reinforcement Learning - Learning through trial and error

Applications:
- Image recognition
- Natural language processing
- Recommendation systems
- Autonomous vehicles
```

**sample-docs/embeddings_explained.txt:**
```
Understanding Embeddings

Embeddings are dense vector representations of data, commonly used in 
natural language processing and machine learning. They convert discrete 
objects (like words or sentences) into continuous vectors in a 
high-dimensional space.

How Embeddings Work:
Embeddings capture semantic meaning by positioning similar items close 
together in vector space. For example, the embeddings for "king" and 
"queen" would be closer than "king" and "car".

Common Embedding Models:
- Word2Vec - Maps words to vectors
- BERT - Contextual embeddings
- Sentence-BERT - Sentence-level embeddings

Benefits:
- Capture semantic relationships
- Enable similarity search
- Reduce dimensionality
- Work with neural networks
```

**CLI Configuration File:**

**cli-config.yaml:**
```yaml
# CLI Configuration for RAG Factory
# This file configures the CLI to use strategy pairs

# Strategy pair to use (from strategies/ directory)
strategy_pair: "semantic-local-pair"

# Service registry configuration
service_registry: "config/services.yaml"

# Alembic configuration
alembic_config: "alembic.ini"

# Default behavior
defaults:
  top_k: 5
  verbose: true
  validate_migrations: true

# Output formatting
output:
  format: "table"  # "table", "json", "simple"
  show_scores: true
  show_metadata: true
  max_content_length: 200  # Characters to show per result

# Performance settings
performance:
  batch_size: 32
  show_timing: true
```

**CLI Commands (Epic 14 Integration):**

```bash
# 1. Initialize (runs migrations, validates setup)
rag-factory init --config cli-config.yaml

# 2. Index the sample documents
rag-factory index \
  --config cli-config.yaml \
  --path ./sample-docs \
  --verbose

# 3. Query for information
rag-factory query \
  --config cli-config.yaml \
  --query "What is Python used for?"

rag-factory query \
  --config cli-config.yaml \
  --query "Explain supervised learning"

rag-factory query \
  --config cli-config.yaml \
  --query "How do embeddings work?"

rag-factory query \
  --config cli-config.yaml \
  --query "What are the benefits of using Python?"

rag-factory query \
  --config cli-config.yaml \
  --query "What is Word2Vec?"
```

**Expected Results:**

**Query 1: "What is Python used for?"**
```
Top 5 Results:
┌─────┬───────┬──────────────────────────────────────────────┐
│ Rank│ Score │ Content                                      │
├─────┼───────┼──────────────────────────────────────────────┤
│  1  │ 0.89  │ Common Uses: Web development (Django, Flask),│
│     │       │ Data science and machine learning...        │
├─────┼───────┼──────────────────────────────────────────────┤
│  2  │ 0.76  │ Python is a high-level, interpreted         │
│     │       │ programming language known for its clear... │
└─────┴───────┴──────────────────────────────────────────────┘

Source: python_basics.txt
Retrieved in 87ms
```

**Query 2: "Explain supervised learning"**
```
Top 5 Results:
┌─────┬───────┬──────────────────────────────────────────────┐
│ Rank│ Score │ Content                                      │
├─────┼───────┼──────────────────────────────────────────────┤
│  1  │ 0.93  │ Types of Machine Learning: 1. Supervised    │
│     │       │ Learning - Training with labeled data...    │
├─────┼───────┼──────────────────────────────────────────────┤
│  2  │ 0.71  │ Machine learning is a subset of artificial  │
│     │       │ intelligence that enables systems to learn..│
└─────┴───────┴──────────────────────────────────────────────┘

Source: machine_learning.txt
Retrieved in 92ms
```

**Query 3: "How do embeddings work?"**
```
Top 5 Results:
┌─────┬───────┬──────────────────────────────────────────────┐
│ Rank│ Score │ Content                                      │
├─────┼───────┼──────────────────────────────────────────────┤
│  1  │ 0.91  │ How Embeddings Work: Embeddings capture     │
│     │       │ semantic meaning by positioning similar...  │
├─────┼───────┼──────────────────────────────────────────────┤
│  2  │ 0.84  │ Embeddings are dense vector representations │
│     │       │ of data, commonly used in natural language..│
└─────┴───────┴──────────────────────────────────────────────┘

Source: embeddings_explained.txt
Retrieved in 78ms
```

**Automated Validation Script:**

**tests/e2e/test_cli_workflow.sh:**
```bash
#!/bin/bash
# End-to-End CLI Validation Script

set -e  # Exit on error

echo "🚀 Starting End-to-End CLI Validation"
echo "======================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Clean slate
echo "📝 Step 1: Cleaning previous data..."
rag-factory reset --config cli-config.yaml --yes
echo -e "${GREEN}✓ Clean slate ready${NC}\n"

# Step 2: Initialize
echo "🔧 Step 2: Initializing RAG Factory..."
rag-factory init --config cli-config.yaml
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Initialization successful${NC}\n"
else
    echo -e "${RED}✗ Initialization failed${NC}"
    exit 1
fi

# Step 3: Index documents
echo "📚 Step 3: Indexing sample documents..."
rag-factory index \
  --config cli-config.yaml \
  --path ./sample-docs \
  --verbose

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Indexing successful${NC}\n"
else
    echo -e "${RED}✗ Indexing failed${NC}"
    exit 1
fi

# Step 4: Validate document count
echo "🔍 Step 4: Validating indexed documents..."
DOC_COUNT=$(rag-factory stats --config cli-config.yaml --format json | jq '.document_count')
if [ "$DOC_COUNT" -eq 3 ]; then
    echo -e "${GREEN}✓ Found 3 documents as expected${NC}\n"
else
    echo -e "${RED}✗ Expected 3 documents, found $DOC_COUNT${NC}"
    exit 1
fi

# Step 5: Test queries
echo "💬 Step 5: Testing retrieval queries..."

QUERIES=(
    "What is Python used for?"
    "Explain supervised learning"
    "How do embeddings work?"
    "What are the benefits of using Python?"
    "What is Word2Vec?"
)

for i in "${!QUERIES[@]}"; do
    QUERY="${QUERIES[$i]}"
    echo "  Query $((i+1)): $QUERY"
    
    rag-factory query \
      --config cli-config.yaml \
      --query "$QUERY" \
      --format json \
      > /tmp/query_result_$i.json
    
    RESULT_COUNT=$(jq '.results | length' /tmp/query_result_$i.json)
    
    if [ "$RESULT_COUNT" -gt 0 ]; then
        echo -e "  ${GREEN}✓ Retrieved $RESULT_COUNT results${NC}"
    else
        echo -e "  ${RED}✗ No results returned${NC}"
        exit 1
    fi
done

echo ""

# Step 6: Performance validation
echo "⚡ Step 6: Performance validation..."
AVG_RETRIEVAL_TIME=$(rag-factory stats --config cli-config.yaml --format json | jq '.avg_retrieval_time_ms')

if (( $(echo "$AVG_RETRIEVAL_TIME < 500" | bc -l) )); then
    echo -e "${GREEN}✓ Average retrieval time: ${AVG_RETRIEVAL_TIME}ms (< 500ms target)${NC}\n"
else
    echo -e "${RED}⚠ Average retrieval time: ${AVG_RETRIEVAL_TIME}ms (slower than 500ms target)${NC}\n"
fi

# Success!
echo "======================================"
echo -e "${GREEN}🎉 All validation tests passed!${NC}"
echo "======================================"

# Print summary
echo ""
echo "Summary:"
echo "  Documents indexed: $DOC_COUNT"
echo "  Queries tested: ${#QUERIES[@]}"
echo "  Average retrieval time: ${AVG_RETRIEVAL_TIME}ms"
echo "  Strategy pair: semantic-local-pair"
echo ""
```

**Python Alternative (for pytest integration):**

**tests/e2e/test_cli_workflow.py:**
```python
"""
End-to-End CLI Validation Test
Tests complete workflow: init -> index -> query -> validate
"""
import subprocess
import json
import pytest
from pathlib import Path

@pytest.fixture
def cli_config():
    return "cli-config.yaml"

@pytest.fixture
def sample_docs_path():
    return Path("sample-docs")

def run_cli(command: list) -> tuple[int, str, str]:
    """Run CLI command and return (returncode, stdout, stderr)"""
    result = subprocess.run(
        ["rag-factory"] + command,
        capture_output=True,
        text=True
    )
    return result.returncode, result.stdout, result.stderr

class TestEndToEndCLI:
    """Complete end-to-end CLI workflow validation"""
    
    def test_01_init(self, cli_config):
        """Test initialization"""
        returncode, stdout, stderr = run_cli([
            "init", "--config", cli_config
        ])
        
        assert returncode == 0, f"Init failed: {stderr}"
        assert "✓" in stdout or "success" in stdout.lower()
    
    def test_02_index_documents(self, cli_config, sample_docs_path):
        """Test indexing sample documents"""
        returncode, stdout, stderr = run_cli([
            "index",
            "--config", cli_config,
            "--path", str(sample_docs_path),
            "--verbose"
        ])
        
        assert returncode == 0, f"Indexing failed: {stderr}"
        assert "3" in stdout  # Should mention 3 documents
    
    def test_03_document_count(self, cli_config):
        """Verify correct number of documents indexed"""
        returncode, stdout, stderr = run_cli([
            "stats", "--config", cli_config, "--format", "json"
        ])
        
        assert returncode == 0
        stats = json.loads(stdout)
        assert stats["document_count"] == 3
    
    @pytest.mark.parametrize("query,expected_source", [
        ("What is Python used for?", "python_basics"),
        ("Explain supervised learning", "machine_learning"),
        ("How do embeddings work?", "embeddings_explained"),
        ("What are the benefits of using Python?", "python_basics"),
        ("What is Word2Vec?", "embeddings_explained"),
    ])
    def test_04_queries(self, cli_config, query, expected_source):
        """Test retrieval queries return relevant results"""
        returncode, stdout, stderr = run_cli([
            "query",
            "--config", cli_config,
            "--query", query,
            "--format", "json"
        ])
        
        assert returncode == 0, f"Query failed: {stderr}"
        
        results = json.loads(stdout)
        assert len(results["results"]) > 0, "No results returned"
        
        # Check that top result is from expected source
        top_result = results["results"][0]
        assert expected_source in top_result["metadata"]["source"]
    
    def test_05_performance(self, cli_config):
        """Verify performance meets targets"""
        returncode, stdout, stderr = run_cli([
            "stats", "--config", cli_config, "--format", "json"
        ])
        
        assert returncode == 0
        stats = json.loads(stdout)
        
        # Check average retrieval time
        avg_time = stats["avg_retrieval_time_ms"]
        assert avg_time < 500, f"Retrieval too slow: {avg_time}ms"
        
        # Check indexing throughput
        indexing_rate = stats["indexing_rate_chunks_per_sec"]
        assert indexing_rate > 100, f"Indexing too slow: {indexing_rate} chunks/sec"
```

**Tutorial Documentation:**

**docs/tutorials/quick-start-cli.md:**
```markdown
# Quick Start: CLI End-to-End Tutorial

This tutorial walks you through indexing documents and retrieving 
information using the RAG Factory CLI.

**Time:** 10 minutes  
**Prerequisites:** PostgreSQL with pgvector, Python 3.10+  
**API Keys:** None required (uses local ONNX models)

## Step 1: Setup (2 minutes)

1. Install RAG Factory:
   ```bash
   pip install rag-factory
   ```

2. Configure environment:
   ```bash
   echo "DATABASE_URL=postgresql://user:pass@localhost:5432/rag_factory" > .env
   ```

3. Initialize:
   ```bash
   rag-factory init --config cli-config.yaml
   ```

## Step 2: Index Documents (2 minutes)

We'll index 3 sample documents about Python, ML, and embeddings:

```bash
rag-factory index \
  --config cli-config.yaml \
  --path ./sample-docs \
  --verbose
```

**Expected output:**
```
📚 Indexing documents from ./sample-docs
Found 3 documents
✓ python_basics.txt (356 chars, 2 chunks)
✓ machine_learning.txt (412 chars, 3 chunks)
✓ embeddings_explained.txt (389 chars, 2 chunks)

Total: 3 documents, 7 chunks indexed in 2.3s
Indexing rate: ~3.0 chunks/sec
```

## Step 3: Query Information (5 minutes)

Now let's retrieve information:

### Query 1: Python Use Cases
```bash
rag-factory query \
  --config cli-config.yaml \
  --query "What is Python used for?"
```

### Query 2: Machine Learning Concepts  
```bash
rag-factory query \
  --config cli-config.yaml \
  --query "Explain supervised learning"
```

### Query 3: Embeddings
```bash
rag-factory query \
  --config cli-config.yaml \
  --query "How do embeddings work?"
```

## Step 4: View Statistics

```bash
rag-factory stats --config cli-config.yaml
```

## What You Just Did

✅ Configured a complete RAG system (semantic-local-pair)  
✅ Indexed 3 documents into PostgreSQL with pgvector  
✅ Performed semantic search to retrieve relevant information  
✅ Used only local ONNX models (no API keys!)  

## Next Steps

- Try indexing your own documents
- Explore other strategy pairs (Story 17.7)
- Combine multiple strategies (hybrid search)
- Deploy to production
```

**Integration with Epic 14:**

The CLI needs these new commands:
- `rag-factory init --config FILE` - Initialize from config file
- `rag-factory index --config FILE --path DIR` - Index documents
- `rag-factory query --config FILE --query TEXT` - Query
- `rag-factory stats --config FILE` - Show statistics
- `rag-factory reset --config FILE` - Clean database

**Story Points:** 5

---

## Story 17.7: Create Remaining Strategy Pair Configurations

**As a** user  
**I want** pre-built configurations for all RAG strategies from previous epics  
**So that** I can quickly deploy any RAG approach without writing YAML

**Acceptance Criteria:**
- Create strategy pair configurations for ALL strategies from Epics 4-7, 12-13:
  1. **semantic-api-pair.yaml** - OpenAI/Cohere API embeddings
  2. **reranking-pair.yaml** - Two-stage retrieval with reranking (Epic 4)
  3. **query-expansion-pair.yaml** - LLM-based query enhancement (Epic 4)
  4. **context-aware-chunking-pair.yaml** - Semantic boundary chunking (Epic 4)
  5. **agentic-rag-pair.yaml** - Agent-based tool selection (Epic 5)
  6. **hierarchical-rag-pair.yaml** - Parent-child chunks (Epic 5)
  7. **self-reflective-pair.yaml** - Self-correcting retrieval (Epic 5)
  8. **multi-query-pair.yaml** - Multiple query variants (Epic 6)
  9. **contextual-retrieval-pair.yaml** - LLM-enriched chunks (Epic 6)
  10. **keyword-pair.yaml** - BM25 keyword search (Epic 12/13)
  11. **hybrid-search-pair.yaml** - Semantic + keyword fusion (Epic 12/13)
  12. **knowledge-graph-pair.yaml** - Graph + vector search (Epic 7)
  13. **late-chunking-pair.yaml** - Embed-then-chunk (Epic 7)
  14. **fine-tuned-embeddings-pair.yaml** - Custom models (Epic 7)
- Each configuration includes:
  - Complete services.yaml entries (can reference or extend base)
  - Required Alembic migrations documented
  - db_config with table/field mappings
  - Example usage code
  - Performance characteristics
  - Cost estimates (if using APIs)
  - Recommended use cases
- All configurations tested with actual strategies
- Documentation matrix showing which pairs can be combined
- Migration dependencies documented (which must run first)

**Deliverables:**

```
strategies/
├── semantic-local-pair.yaml          # From Story 17.6 ✅
├── semantic-api-pair.yaml            # OpenAI/Cohere
├── reranking-pair.yaml               # Epic 4
├── query-expansion-pair.yaml         # Epic 4
├── context-aware-chunking-pair.yaml  # Epic 4
├── agentic-rag-pair.yaml            # Epic 5
├── hierarchical-rag-pair.yaml       # Epic 5
├── self-reflective-pair.yaml        # Epic 5
├── multi-query-pair.yaml            # Epic 6
├── contextual-retrieval-pair.yaml   # Epic 6
├── keyword-pair.yaml                # Epic 12/13
├── hybrid-search-pair.yaml          # Epic 12/13
├── knowledge-graph-pair.yaml        # Epic 7
├── late-chunking-pair.yaml          # Epic 7
├── fine-tuned-embeddings-pair.yaml  # Epic 7
└── README.md                         # Index of all pairs

docs/strategies/
├── strategy-pair-matrix.md           # Compatibility matrix
├── migration-dependencies.md         # Which migrations needed for what
└── [individual-pair-guides].md       # One per pair
```

**Strategy Pair Matrix (Deliverable):**
```markdown
# Strategy Pair Compatibility Matrix

## Can Be Combined
| Base Strategy | Compatible Add-ons |
|---------------|-------------------|
| semantic-local-pair | + reranking, + query-expansion, + hierarchical |
| keyword-pair | + reranking |
| semantic + keyword | → hybrid-search-pair |
| any semantic | + contextual-retrieval (at indexing time) |

## Require Different Tables (Isolated)
- semantic-local vs semantic-api (different embedding dimensions)
- keyword vs semantic (different index structures)
- graph vs vector (different storage backends)

## Migration Dependencies
1. Base schema (Epic 2)
2. Vector tables (semantic pairs)
3. Keyword tables (keyword pair)
4. Graph tables (knowledge graph pair)
5. Hierarchy tables (hierarchical pair)
```

**Example Configuration (Hybrid Search):**
```yaml
# strategies/hybrid-search-pair.yaml
strategy_name: "hybrid-search-pair"
version: "1.0.0"
description: "Combines semantic vector search with BM25 keyword search"

indexer:
  strategy: "HybridIndexer"
  services:
    embedding: "$embedding_local"
    db: "$db_main"
  
  db_config:
    tables:
      # Uses tables from both semantic AND keyword
      semantic_vectors: "semantic_local_vectors"
      keyword_index: "keyword_inverted_index"
      hybrid_results: "hybrid_search_cache"
    fields:
      vector: "vector_embedding"
      keywords: "indexed_terms"
  
  config:
    use_vectors: true
    use_keywords: true
    vector_weight: 0.7
    keyword_weight: 0.3

retriever:
  strategy: "HybridRetriever"
  services:
    embedding: "$embedding_local"
    db: "$db_main"
  
  db_config:
    tables:
      semantic_vectors: "semantic_local_vectors"
      keyword_index: "keyword_inverted_index"
  
  config:
    fusion_algorithm: "reciprocal_rank_fusion"
    top_k: 10

migrations:
  required_revisions:
    - "semantic_local_schema"  # From Story 17.6
    - "keyword_index_schema"   # New for keywords
    - "hybrid_cache_schema"    # New for hybrid

expected_schema:
  tables:
    - "semantic_local_vectors"
    - "keyword_inverted_index"
    - "hybrid_search_cache"
```

**Documentation Requirements:**
- Each pair gets comprehensive guide (setup, usage, performance)
- Compatibility matrix showing which pairs work together
- Migration dependency graph
- Cost comparison table (local vs API)
- Performance benchmarks for all pairs
- Troubleshooting guide for common issues

**Story Points:** 8

---

## Appendix: Integration with Existing Epics

### Dependencies on Previous Work

**Epic 11 (Dependency Injection):**
- ✅ Uses existing `ILLMService`, `IEmbeddingService`, `IDatabaseService` interfaces
- ✅ Uses existing service implementations (ONNXEmbeddingService, etc.)
- ✅ Extends with ServiceRegistry for instance sharing

**Epic 12 (Pipeline Separation):**
- ✅ Uses existing `IIndexingStrategy` and `IRetrievalStrategy` interfaces
- ✅ Uses existing `IndexCapability` enums for validation
- ✅ Uses existing factory validation methods

**Epic 16 (Database Consolidation):**
- ✅ Uses existing `PostgresqlDatabaseService`
- ✅ References existing Alembic migrations (not duplicating)
- ✅ Extends with `DatabaseContext` for table mapping

### What's New in Epic 17

1. **ServiceRegistry** - Central service definition and sharing
2. **DatabaseContext** - Strategy-specific table/field mapping on shared DB
3. **StrategyPair YAML Format** - Configuration that ties everything together
4. **Environment Variable Resolution** - `${VAR}` syntax support
5. **StrategyPairManager** - High-level orchestration

---

## Appendix: Total Story Points

**Epic 17: Strategy Pair Configuration System:**
- Story 17.1: Configuration Schema = 8 points
- Story 17.2: Service Registry = 13 points
- Story 17.3: DatabaseContext = 13 points
- Story 17.4: Migration Validator = 5 points
- Story 17.5: StrategyPairManager = 13 points
- Story 17.6: First Strategy Pair (Testing) = 5 points
- Story 17.7: Remaining Strategy Pairs = 8 points
- Story 17.8: End-to-End CLI Validation = 5 points

**Total:** 70 story points (~3 sprints)

**Sprint Recommendation:**
- **Sprint 1 (Stories 17.1-17.3):** 34 points - Foundation (schemas, registry, database context)
- **Sprint 2 (Stories 17.4-17.6):** 23 points - Integration (validation, manager, first test pair)
- **Sprint 3 (Stories 17.7-17.8):** 13 points - Completion (remaining pairs + E2E validation)

**Combined Project Total:**
- Epics 1-9: 280 points
- Epic 10 (DI): Completed ✅
- Epic 11 (Pipelines): 73 points
- Epic 12 (Indexing): 34 points
- Epic 13 (Retrieval): 21 points
- Epic 14 (CLI): 13 points
- Epic 15 (Test Coverage): 33 points
- Epic 16 (DB Consolidation): 23 points
- **Epic 17 (Strategy Pairs): 70 points**

---

## Benefits Summary

### 1. Service Sharing (Memory & Performance)
- **Before:** Each strategy loads its own embedding model (3x memory usage)
- **After:** One embedding model shared by all strategies (1x memory usage)

### 2. Strategy Isolation (Clean Architecture)
- **Before:** All strategies forced to use same table names
- **After:** Each strategy has its own tables/fields via DatabaseContext

### 3. Configuration Portability (DevOps)
- **Before:** Manual setup, tribal knowledge
- **After:** Copy YAML file, run `manager.load_pair()`

### 4. Alembic Integration (No Duplication)
- **Before:** Would need separate migration system
- **After:** References existing Alembic revisions

### 5. Environment Variables (Security)
- **Before:** Hardcoded API keys in code
- **After:** `${OPENAI_API_KEY}` resolved from .env

This epic completes the RAG Factory's transformation into an enterprise-grade, production-ready system with proper resource management, clean architecture, and operational excellence.

