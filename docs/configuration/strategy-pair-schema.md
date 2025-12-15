# Strategy Pair Schema Reference

## Overview

A strategy pair configuration defines a complete RAG workflow by bundling compatible indexing and retrieval strategies with their required services, database mappings, and migrations. This document provides a complete reference for the strategy pair schema.

## Schema Version

Current version: **1.0.0**

## File Format

```yaml
strategy_name: "strategy-name"
version: "1.0.0"
description: "Human-readable description"

indexer:
  # Indexing strategy configuration

retriever:
  # Retrieval strategy configuration

migrations:
  # Optional: Database migration requirements

expected_schema:
  # Optional: Expected database schema

tags:
  # Optional: Tags for discovery/filtering
```

## Required Fields

### strategy_name

- **Type**: string
- **Pattern**: `^[a-z0-9-]+$` (lowercase letters, numbers, hyphens only)
- **Description**: Unique identifier for the strategy pair

### version

- **Type**: string
- **Pattern**: `^\d+\.\d+\.\d+$` (semantic versioning)
- **Description**: Configuration version

### indexer

- **Type**: object
- **Description**: Indexing strategy configuration
- **See**: [Strategy Configuration](#strategy-configuration)

### retriever

- **Type**: object
- **Description**: Retrieval strategy configuration
- **See**: [Strategy Configuration](#strategy-configuration)

## Optional Fields

### description

- **Type**: string
- **Description**: Human-readable description of the strategy pair

### migrations

- **Type**: object
- **Description**: Database migration requirements

```yaml
migrations:
  required_revisions:
    - "revision_id_1"
    - "revision_id_2"
```

### expected_schema

- **Type**: object
- **Description**: Expected database schema elements

```yaml
expected_schema:
  tables:
    - "documents"
    - "chunks"
  indexes:
    - "idx_chunks_embedding"
  extensions:
    - "vector"
```

### tags

- **Type**: array of strings
- **Description**: Tags for discovery and filtering

```yaml
tags:
  - "semantic-search"
  - "local"
  - "production-ready"
```

## Strategy Configuration

Both `indexer` and `retriever` follow the same configuration structure:

### Required Fields

#### strategy

- **Type**: string
- **Description**: Strategy class name (e.g., `"VectorEmbeddingIndexer"`, `"SemanticRetriever"`)

#### services

- **Type**: object
- **Description**: Service references or inline configurations

Services can be specified in two ways:

1. **Service Reference** (recommended): Reference a service from the service registry
   ```yaml
   services:
     embedding: "$embedding_local"
     llm: "$llm_openai"
     database: "$db_main"
   ```

2. **Inline Configuration**: Define service inline
   ```yaml
   services:
     embedding:
       name: "Inline Embedding"
       type: "embedding"
       provider: "onnx"
       model: "all-MiniLM-L6-v2"
   ```

### Optional Fields

#### db_config

- **Type**: object
- **Description**: Database table and field mappings

```yaml
db_config:
  tables:
    documents: "documents"
    chunks: "chunks"
    metadata: "chunk_metadata"
  fields:
    content: "content"
    embedding: "embedding"
    document_id: "document_id"
```

#### config

- **Type**: object
- **Description**: Strategy-specific configuration parameters

```yaml
config:
  chunk_size: 512
  chunk_overlap: 50
  top_k: 5
  similarity_threshold: 0.7
```

## Service References

Service references use the `$service_name` syntax to reference services defined in the service registry.

### Syntax

```yaml
services:
  service_type: "$service_name"
```

- Must start with `$`
- Service name must exist in the service registry
- Service type must match strategy requirements

### Example

```yaml
# In services.yaml
services:
  embedding_local:
    name: "Local ONNX"
    type: "embedding"
    provider: "onnx"

# In strategy-pair.yaml
indexer:
  strategy: "VectorEmbeddingIndexer"
  services:
    embedding: "$embedding_local"  # References the service above
```

## Complete Examples

### Basic Semantic Search

```yaml
strategy_name: "semantic-local"
version: "1.0.0"
description: "Basic semantic search with local ONNX embeddings"

indexer:
  strategy: "VectorEmbeddingIndexer"
  services:
    embedding: "$embedding_local"
    database: "$db_main"
  db_config:
    tables:
      documents: "documents"
      chunks: "chunks"
  config:
    chunk_size: 512
    chunk_overlap: 50

retriever:
  strategy: "SemanticRetriever"
  services:
    embedding: "$embedding_local"
    database: "$db_main"
  config:
    top_k: 5
    similarity_threshold: 0.7

migrations:
  required_revisions:
    - "001_create_documents_table"
    - "002_create_chunks_table"
    - "003_add_vector_extension"

expected_schema:
  tables:
    - "documents"
    - "chunks"
  indexes:
    - "idx_chunks_embedding"
  extensions:
    - "vector"

tags:
  - "semantic-search"
  - "local"
  - "basic"
```

### Hybrid Search

```yaml
strategy_name: "hybrid-search"
version: "1.0.0"
description: "Hybrid search combining semantic and keyword search"

indexer:
  strategy: "HybridIndexer"
  services:
    embedding: "$embedding_local"
    database: "$db_main"
  db_config:
    tables:
      documents: "documents"
      chunks: "chunks"
      keywords: "chunk_keywords"
  config:
    chunk_size: 512
    chunk_overlap: 50
    extract_keywords: true
    max_keywords: 10

retriever:
  strategy: "HybridRetriever"
  services:
    embedding: "$embedding_local"
    database: "$db_main"
  config:
    top_k: 10
    semantic_weight: 0.7
    keyword_weight: 0.3
    fusion_method: "rrf"  # Reciprocal Rank Fusion

migrations:
  required_revisions:
    - "001_create_documents_table"
    - "002_create_chunks_table"
    - "003_add_vector_extension"
    - "004_create_keywords_table"

expected_schema:
  tables:
    - "documents"
    - "chunks"
    - "chunk_keywords"
  indexes:
    - "idx_chunks_embedding"
    - "idx_keywords_term"
  extensions:
    - "vector"

tags:
  - "hybrid-search"
  - "semantic"
  - "keyword"
  - "advanced"
```

### Knowledge Graph RAG

```yaml
strategy_name: "knowledge-graph"
version: "1.0.0"
description: "Knowledge graph-based RAG with entity and relationship extraction"

indexer:
  strategy: "KnowledgeGraphIndexer"
  services:
    embedding: "$embedding_local"
    llm: "$llm_openai"
    database: "$db_neo4j"
  db_config:
    tables:
      documents: "documents"
      entities: "entities"
      relationships: "relationships"
  config:
    chunk_size: 512
    extract_entities: true
    extract_relationships: true
    entity_types:
      - "PERSON"
      - "ORGANIZATION"
      - "LOCATION"
      - "CONCEPT"

retriever:
  strategy: "KnowledgeGraphRetriever"
  services:
    embedding: "$embedding_local"
    database: "$db_neo4j"
  config:
    top_k: 5
    max_depth: 2
    include_relationships: true
    relationship_weight: 0.5

migrations:
  required_revisions:
    - "kg_001_create_graph_schema"
    - "kg_002_create_indexes"

expected_schema:
  tables:
    - "documents"
  indexes:
    - "idx_entity_name"
    - "idx_relationship_type"

tags:
  - "knowledge-graph"
  - "entity-extraction"
  - "advanced"
  - "neo4j"
```

### Multi-Query with Reranking

```yaml
strategy_name: "multi-query-rerank"
version: "1.0.0"
description: "Multi-query expansion with cross-encoder reranking"

indexer:
  strategy: "VectorEmbeddingIndexer"
  services:
    embedding: "$embedding_openai"
    database: "$db_main"
  db_config:
    tables:
      documents: "documents"
      chunks: "chunks"
  config:
    chunk_size: 512
    chunk_overlap: 50

retriever:
  strategy: "MultiQueryRetriever"
  services:
    embedding: "$embedding_openai"
    llm: "$llm_openai"
    database: "$db_main"
  config:
    num_queries: 3
    top_k_per_query: 10
    final_top_k: 5
    rerank: true
    reranker_model: "cross-encoder/ms-marco-MiniLM-L-12-v2"

migrations:
  required_revisions:
    - "001_create_documents_table"
    - "002_create_chunks_table"
    - "003_add_vector_extension"

tags:
  - "multi-query"
  - "reranking"
  - "advanced"
  - "cloud"
```

## Environment Variables

Strategy pair configurations support the same environment variable syntax as service registry:

```yaml
indexer:
  strategy: "VectorEmbeddingIndexer"
  services:
    embedding: "$embedding_local"
  config:
    chunk_size: ${CHUNK_SIZE:-512}
    chunk_overlap: ${CHUNK_OVERLAP:-50}
    model_path: "${MODEL_PATH:?Model path is required}"
```

## Validation

### Schema Validation

```python
from rag_factory.config.validator import load_yaml_with_validation
from rag_factory.config.env_resolver import EnvResolver
import yaml

# Load service registry first
with open("config/services.yaml") as f:
    services = yaml.safe_load(f)

# Load and validate strategy pair
strategy = load_yaml_with_validation(
    "config/semantic-local-pair.yaml",
    config_type="strategy_pair",
    service_registry=services
)

# Resolve environment variables
strategy = EnvResolver.resolve(strategy)
```

### Service Reference Validation

The validator checks:
- Referenced services exist in the service registry
- Service types match strategy requirements
- No circular dependencies
- Inline service configurations are valid

### Common Validation Errors

1. **"Service reference '$xyz' not found in registry"**
   - Solution: Add the service to `services.yaml` or fix the reference

2. **"Schema validation failed: 'strategy' is a required property"**
   - Solution: Add the `strategy` field to indexer/retriever

3. **"Invalid strategy_name: must match pattern ^[a-z0-9-]+$"**
   - Solution: Use only lowercase letters, numbers, and hyphens

## Best Practices

1. **Use service references**: Define services once, reference everywhere
2. **Version your configurations**: Use semantic versioning
3. **Document migrations**: List all required database migrations
4. **Specify expected schema**: Document required tables and indexes
5. **Use descriptive names**: Choose clear, meaningful strategy names
6. **Add tags**: Use tags for discovery and filtering
7. **Provide defaults**: Use environment variables with defaults for flexibility
8. **Test configurations**: Validate before deployment
9. **Keep it DRY**: Reuse services across multiple strategy pairs
10. **Document requirements**: Clearly specify all dependencies

## Database Configuration

### Table Mapping

Map logical table names to physical database tables:

```yaml
db_config:
  tables:
    documents: "my_documents"      # Logical -> Physical
    chunks: "my_chunks"
    metadata: "my_metadata"
```

### Field Mapping

Map logical field names to physical column names:

```yaml
db_config:
  fields:
    content: "text_content"        # Logical -> Physical
    embedding: "vector_embedding"
    document_id: "doc_id"
    created_at: "timestamp"
```

### Benefits

- **Flexibility**: Adapt to existing database schemas
- **Portability**: Move strategies between databases
- **Isolation**: Multiple strategies can use different tables
- **Migration**: Easier schema evolution

## Migration Management

### Specifying Migrations

```yaml
migrations:
  required_revisions:
    - "001_create_documents_table"
    - "002_create_chunks_table"
    - "003_add_vector_extension"
    - "004_create_indexes"
```

### Validation

The system can validate that:
- All required migrations have been applied
- Database schema matches expectations
- Required extensions are installed

## Troubleshooting

### Common Issues

1. **Service not found**
   - Check service name spelling
   - Verify service exists in registry
   - Ensure `$` prefix is used

2. **Invalid strategy name**
   - Use lowercase, numbers, hyphens only
   - No spaces or special characters

3. **Missing required fields**
   - Check schema requirements
   - Ensure all required fields are present

4. **Database mapping errors**
   - Verify table names exist
   - Check field name mappings
   - Ensure migrations are applied

### Getting Help

- See `service-registry-schema.md` for service configuration
- See `environment-variables.md` for environment variable guide
- See example files in `rag_factory/config/examples/`
- Check the JSON schema files for detailed validation rules
