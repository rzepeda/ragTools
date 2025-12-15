# Epic 17: Strategy Pair Configuration System

This epic implements a configuration system that bundles compatible indexing and retrieval strategies with their required services, database table mappings, and migrations, enabling portable and reproducible RAG deployments.

---

## Overview

**Epic Goal:** Create a configuration system that bundles compatible indexing and retrieval strategies with their required services, database table mappings, and migrations, enabling portable and reproducible RAG deployments while leveraging shared service instances.

**Total Story Points:** 70

**Dependencies:**
- Epic 11 (Dependency Injection - service interfaces must exist) ✅
- Epic 12 (Pipeline Separation - capability system must exist) ✅
- Epic 16 (Database Consolidation - Alembic must be working) ✅

---

## Stories

### Story 17.1: Design Service Registry and Configuration Schema (8 points)
**Status:** Planned
**File:** [story-17.1-service-registry-configuration-schema.md](./story-17.1-service-registry-configuration-schema.md)

**Description:** Define well-structured YAML schemas for service registry and strategy pair configurations with environment variable support.

**Key Features:**
- Service registry YAML schema (services.yaml)
- Strategy pair YAML schema (strategy-pair.yaml)
- Environment variable resolution (`${VAR}` syntax)
- Service references (`$service_name` syntax)
- Schema validation utilities
- Versioning for backward compatibility

**Acceptance Criteria:**
- YAML schemas defined and documented
- Environment variable resolution working (${VAR}, ${VAR:-default}, ${VAR:?error})
- Service reference validation
- Schema validation with jsonschema
- Comprehensive examples provided

---

### Story 17.2: Implement Service Registry (13 points)
**Status:** Planned
**File:** [story-17.2-service-registry-implementation.md](./story-17.2-service-registry-implementation.md)

**Description:** Create a service registry that loads service configurations from YAML and creates/caches service instances for sharing across strategies.

**Key Features:**
- ServiceRegistry class loads services.yaml
- Lazy service instantiation
- Singleton pattern per service definition
- Thread-safe with locks
- Environment variable resolution
- Integration with Epic 11 service interfaces

**Acceptance Criteria:**
- Service instances cached and reused
- Thread-safe instantiation
- Environment variables resolved correctly
- Clear error messages for missing services/variables
- Unit and integration tests passing
- Multiple strategies can share same service instances

---

### Story 17.3: Implement DatabaseContext for Table Mapping (13 points)
**Status:** ✅ Completed
**File:** [story-17.3-database-context-table-mapping.md](./story-17.3-database-context-table-mapping.md)

**Description:** Extend PostgresqlDatabaseService to support strategy-specific database contexts with table and field name mapping for strategy isolation on shared databases.

**Key Features:**
- DatabaseContext class with table/field mapping
- Logical-to-physical table name translation
- Logical-to-physical field name translation
- CRUD operations using logical names
- Vector search operations (pgvector support)
- PostgresqlDatabaseService.get_context() method

**Acceptance Criteria:**
- Multiple contexts share same connection pool
- Table/field mappings work correctly
- CRUD operations use logical names
- Vector search supports cosine/L2/inner product
- Context caching implemented
- Integration tests show strategy isolation
- No breaking changes to Epic 16 API

---

### Story 17.4: Implement Migration Validator (5 points)
**Status:** Planned
**File:** [story-17.4-migration-validator-alembic.md](./story-17.4-migration-validator-alembic.md)

**Description:** Create a migration validator that checks if required Alembic migrations are applied before loading strategy pairs.

**Key Features:**
- Query alembic_version table
- Check required revisions in migration history
- Clear error messages for missing migrations
- Suggest alembic upgrade commands
- Handle missing alembic_version table

**Acceptance Criteria:**
- Validation checks current Alembic revision
- Missing migrations detected and listed
- Clear error messages with upgrade suggestions
- Integration tests with test Alembic setup
- Performance: validation < 100ms

---

### Story 17.5: Implement StrategyPair Loader and Manager (13 points)
**Status:** Planned
**File:** [story-17.5-strategy-pair-manager.md](./story-17.5-strategy-pair-manager.md)

**Description:** Create high-level manager that loads strategy pair YAML configurations and instantiates complete indexing/retrieval strategy pairs with all dependencies.

**Key Features:**
- StrategyPairLoader parses YAML configurations
- StrategyPairManager orchestrates loading
- Migration validation before instantiation
- Service resolution from ServiceRegistry
- DatabaseContext creation with mappings
- Capability compatibility validation (Epic 12)

**Acceptance Criteria:**
- Strategy pairs load from YAML configuration
- Migrations validated before loading
- Services resolved and shared correctly
- Database contexts created with proper mappings
- Capability validation working
- Clear error messages for all validation failures
- Caching for loaded pairs

---

### Story 17.6: Create First Strategy Pair Configuration (5 points)
**Status:** Planned
**File:** [story-17.6-first-strategy-pair-testing.md](./story-17.6-first-strategy-pair-testing.md)

**Description:** Create one complete, tested strategy pair configuration (semantic-local-pair) to validate the entire Epic 17 implementation.

**Key Features:**
- semantic-local-pair.yaml using local ONNX
- Complete services.yaml configuration
- Alembic migrations for semantic tables
- Comprehensive README with setup instructions
- Example usage code
- CLI integration tests

**Acceptance Criteria:**
- Configuration loads without errors
- No API keys required (local ONNX)
- End-to-end indexing and retrieval working
- Service sharing verified (memory efficient)
- Migration validation working
- CLI integration tested
- Performance benchmarks documented

---

### Story 17.7: Create Remaining Strategy Pair Configurations (8 points)
**Status:** Planned
**File:** [story-17.7-remaining-strategy-pairs.md](./story-17.7-remaining-strategy-pairs.md)

**Description:** Create strategy pair configurations for all RAG strategies from Epics 4-7, 12-13.

**Strategy Pairs to Create:**
- semantic-api-pair.yaml (OpenAI/Cohere embeddings)
- reranking-pair.yaml (Epic 4)
- query-expansion-pair.yaml (Epic 4)
- context-aware-chunking-pair.yaml (Epic 4)
- agentic-rag-pair.yaml (Epic 5)
- hierarchical-rag-pair.yaml (Epic 5)
- self-reflective-pair.yaml (Epic 5)
- multi-query-pair.yaml (Epic 6)
- contextual-retrieval-pair.yaml (Epic 6)
- keyword-pair.yaml (Epic 12/13)
- hybrid-search-pair.yaml (Epic 12/13)
- knowledge-graph-pair.yaml (Epic 7)
- late-chunking-pair.yaml (Epic 7)
- fine-tuned-embeddings-pair.yaml (Epic 7)

**Acceptance Criteria:**
- All 14 strategy pairs configured
- Each includes complete service definitions
- Required migrations documented
- db_config with table/field mappings
- Example usage for each pair
- Performance and cost characteristics documented
- Compatibility matrix showing which pairs can be combined

---

### Story 17.8: End-to-End CLI Validation (5 points)
**Status:** Planned
**File:** [story-17.8-cli-validation-sample-docs.md](./story-17.8-cli-validation-sample-docs.md)

**Description:** Create end-to-end validation by indexing 3 sample documents and performing queries via CLI to prove complete system integration.

**Key Features:**
- 3 sample documents (Python, ML, Embeddings)
- CLI configuration file (cli-config.yaml)
- Index documents via CLI
- 5 test queries via CLI
- Automated validation script
- Performance metrics tracking

**Acceptance Criteria:**
- Sample documents created with known content
- CLI reads configuration from YAML
- All 3 documents indexed successfully
- All 5 queries return correct results
- Automated test script validates workflow
- Performance metrics meet requirements
- Complete tutorial documentation

---

## Sprint Planning

**Sprint 1 (Stories 17.1-17.3):** 34 points - Foundation
- Configuration schemas
- Service Registry
- DatabaseContext

**Sprint 2 (Stories 17.4-17.6):** 23 points - Integration
- Migration validation
- StrategyPairManager
- First test pair (semantic-local)

**Sprint 3 (Stories 17.7-17.8):** 13 points - Completion
- Remaining strategy pairs
- End-to-end CLI validation

---

## Technical Stack

### Configuration
- **PyYAML** for YAML parsing
- **jsonschema** for schema validation
- **os.path.expandvars()** for environment variable resolution

### Service Management
- **Threading.Lock** for thread-safe instantiation
- **Collections.defaultdict** for lock management
- Service interfaces from Epic 11

### Database
- **SQLAlchemy** for database operations
- **PostgreSQL** with connection pooling
- **pgvector** for vector operations
- **Alembic** for migration tracking

---

## Success Criteria

Epic 17 is complete when:

- [ ] **Service Registry** loads and caches services from YAML
- [ ] **DatabaseContext** provides strategy isolation on shared database
- [ ] **Migration Validator** checks Alembic migrations before loading pairs
- [ ] **StrategyPairManager** loads complete strategy pairs from configuration
- [ ] **First strategy pair** (semantic-local) fully tested and working
- [ ] **All strategy pairs** from previous epics configured
- [ ] **CLI integration** tested end-to-end with sample documents
- [ ] All integration tests passing
- [ ] Documentation complete with usage examples
- [ ] Performance metrics meeting requirements

---

## Getting Started

### Prerequisites

Before starting Epic 17, ensure:

1. **Epic 11 Complete:** Service interfaces (ILLMService, IEmbeddingService, IDatabaseService)
2. **Epic 12 Complete:** IndexCapability enums and pipeline separation
3. **Epic 16 Complete:** PostgresqlDatabaseService with connection pooling and Alembic
4. **PostgreSQL:** With pgvector extension installed
5. **Python:** 3.10+ with type hints support

### Installation

```bash
# Install Epic 17 dependencies
pip install pyyaml jsonschema sqlalchemy psycopg2-binary pgvector alembic

# Verify installations
python -c "import yaml; print('PyYAML OK')"
python -c "import jsonschema; print('jsonschema OK')"
python -c "import sqlalchemy; print('SQLAlchemy OK')"
```

### Configuration Files

```bash
# Create configuration directories
mkdir -p config strategies

# Create services registry
cat > config/services.yaml << 'EOF'
services:
  embedding_local:
    name: "local-onnx-minilm"
    provider: "onnx"
    model: "Xenova/all-MiniLM-L6-v2"
    cache_dir: "./models/embeddings"
    batch_size: 32

  db_main:
    name: "main-postgres"
    type: "postgres"
    connection_string: "${DATABASE_URL}"
    pool_size: 10
    max_overflow: 20
EOF

# Set environment variables
echo "DATABASE_URL=postgresql://user:pass@localhost:5432/rag_factory" > .env
```

---

## Implementation Order

**Recommended order:**

1. **Story 17.1: Configuration Schema** (Foundation)
   - Define YAML schemas first
   - Establish configuration patterns
   - Create validation utilities

2. **Story 17.2: Service Registry** (Service Layer)
   - Implement service loading and caching
   - Test with existing service implementations
   - Verify thread-safety

3. **Story 17.3: DatabaseContext** (Database Layer) ✅
   - Extend PostgresqlDatabaseService
   - Implement table/field mapping
   - Test strategy isolation

4. **Story 17.4: Migration Validator** (Validation)
   - Add migration checking
   - Integrate with Alembic
   - Clear error messages

5. **Story 17.5: StrategyPairManager** (Orchestration)
   - Tie everything together
   - Load complete strategy pairs
   - Validate compatibility

6. **Story 17.6: First Strategy Pair** (Testing)
   - Prove the system works end-to-end
   - Create reusable template
   - Document best practices

7. **Story 17.7: Remaining Pairs** (Scale Out)
   - Apply template to all strategies
   - Document compatibility
   - Create usage examples

8. **Story 17.8: CLI Validation** (User Testing)
   - End-to-end user workflow
   - Automated validation
   - Tutorial documentation

---

## Architecture Benefits

### 1. Service Sharing (Memory & Performance)
**Before Epic 17:**
```python
# Each strategy loads its own embedding model (3x memory)
semantic = SemanticStrategy(embedding=EmbeddingService("model.onnx"))
keyword = KeywordStrategy(embedding=EmbeddingService("model.onnx"))
hybrid = HybridStrategy(embedding=EmbeddingService("model.onnx"))
```

**After Epic 17:**
```python
# One model shared by all strategies (1x memory)
registry = ServiceRegistry("config/services.yaml")
embedding = registry.get("embedding1")  # Load once

semantic = SemanticStrategy(embedding=embedding)  # Reuse
keyword = KeywordStrategy(embedding=embedding)    # Reuse
hybrid = HybridStrategy(embedding=embedding)      # Reuse
```

### 2. Strategy Isolation (Clean Architecture)
**Before Epic 17:**
```python
# All strategies use same table names - conflict!
semantic_strategy.insert_to_chunks_table(...)
keyword_strategy.insert_to_chunks_table(...)  # COLLISION!
```

**After Epic 17:**
```python
# Each strategy has its own tables via DatabaseContext
semantic_ctx.insert("chunks", {...})  # → semantic_chunks
keyword_ctx.insert("chunks", {...})   # → keyword_chunks
```

### 3. Configuration Portability (DevOps)
**Before Epic 17:**
```python
# Manual setup, hardcoded values
db = PostgresqlDatabaseService("postgresql://...")
llm = LMStudioLLMService("http://localhost:1234", "key", "model")
strategy = SemanticStrategy(db, llm)
```

**After Epic 17:**
```yaml
# Copy YAML file, run one command
# config/services.yaml + strategies/semantic-pair.yaml
manager = StrategyPairManager(registry)
indexing, retrieval = manager.load_pair("semantic-pair")
```

### 4. Environment Variables (Security)
**Before Epic 17:**
```python
# API keys in code or config files
api_key = "sk-1234..."  # Hardcoded secret!
```

**After Epic 17:**
```yaml
# Environment variable resolution
services:
  llm1:
    api_key: "${OPENAI_API_KEY}"  # Resolved from .env
```

---

## Testing Strategy

### Unit Tests
Each story includes comprehensive unit tests:
- Service Registry: Service instantiation, caching, thread-safety
- DatabaseContext: Table/field mapping, CRUD operations, vector search
- Migration Validator: Version checking, error messages
- StrategyPairManager: YAML loading, service resolution, validation

### Integration Tests
End-to-end tests with real services:
- Multiple strategies sharing services
- Strategy isolation on shared database
- Complete workflow: load config → index → retrieve
- CLI integration tests

### Performance Benchmarks
- Service Registry: < 10ms per service (cached)
- DatabaseContext: < 1ms mapping overhead
- Migration Validator: < 100ms validation
- StrategyPairManager: < 500ms pair loading

---

## Cost Considerations

### Local vs API Services

**Local ONNX (Recommended for Development):**
- Zero API costs
- ~500MB memory per embedding model
- ~100 chunks/sec indexing speed
- Good for testing and small deployments

**API Services (Production Scale):**
- OpenAI Embeddings: ~$0.0001 per 1K tokens
- Anthropic Claude: ~$0.008 per 1K tokens
- Unlimited scale
- Higher latency (~100-500ms per call)

**Cost Comparison Example:**
- Indexing 10,000 documents (1M tokens):
  - Local ONNX: $0 (500MB RAM for ~5 minutes)
  - OpenAI API: ~$0.10 (instant, unlimited scale)

---

## Monitoring and Observability

All components include comprehensive logging:

- **Service Registry:** Service instantiation, cache hits, errors
- **DatabaseContext:** Table mappings, query execution, reflection
- **Migration Validator:** Missing migrations, validation results
- **StrategyPairManager:** Pair loading, service resolution, validation

Example logs:
```
[ServiceRegistry] Loading service 'embedding1' from config
[ServiceRegistry] Instantiating ONNXEmbeddingService with model 'Xenova/all-MiniLM-L6-v2'
[ServiceRegistry] Service 'embedding1' cached for reuse
[DatabaseContext] Mapping logical table 'chunks' → physical 'semantic_chunks'
[DatabaseContext] Reflected table 'semantic_chunks' (5 columns, cached)
[MigrationValidator] Checking required revisions: ['semantic_local_schema']
[MigrationValidator] Current revision: semantic_local_schema ✓
[StrategyPairManager] Loading strategy pair 'semantic-local-pair'
[StrategyPairManager] Resolved services: embedding=embedding1, db=db_main
[StrategyPairManager] Created semantic indexing strategy
[StrategyPairManager] Created semantic retrieval strategy
[StrategyPairManager] Pair 'semantic-local-pair' loaded successfully
```

---

## Common Patterns

### Pattern 1: Service Sharing

```python
# Load registry once
registry = ServiceRegistry("config/services.yaml")

# Multiple strategies share services
semantic = SemanticStrategy(
    embedding=registry.get("embedding1"),
    db=registry.get("db1")
)
keyword = KeywordStrategy(
    embedding=registry.get("embedding1"),  # Same instance
    db=registry.get("db1")                # Same instance
)

# Verify sharing
assert semantic.embedding is keyword.embedding  # True
```

### Pattern 2: Strategy Isolation

```python
# Get shared database service
db_service = registry.get("db1")

# Create isolated contexts
semantic_ctx = db_service.get_context(
    table_mapping={"chunks": "semantic_chunks"}
)
keyword_ctx = db_service.get_context(
    table_mapping={"chunks": "keyword_chunks"}
)

# Both use logical names, map to different tables
semantic_ctx.insert("chunks", {...})  # → semantic_chunks
keyword_ctx.insert("chunks", {...})   # → keyword_chunks
```

### Pattern 3: Configuration-Driven Deployment

```python
# Development environment
registry = ServiceRegistry("config/dev-services.yaml")
manager = StrategyPairManager(registry)
indexing, retrieval = manager.load_pair("semantic-local-pair")

# Production environment (just swap config files!)
registry = ServiceRegistry("config/prod-services.yaml")
manager = StrategyPairManager(registry)
indexing, retrieval = manager.load_pair("semantic-api-pair")
```

---

## Troubleshooting

### Service Registry Issues

**Problem:** Service not found in registry
**Solution:** Check `config/services.yaml` has service definition, verify service name

**Problem:** Environment variable not set
**Solution:** Check `.env` file, verify variable name in `${VAR}` syntax

**Problem:** Service instantiation fails
**Solution:** Check service configuration parameters, verify dependencies installed

### DatabaseContext Issues

**Problem:** Table mapping not found
**Solution:** Check `db_config.tables` in strategy pair YAML

**Problem:** Physical table doesn't exist
**Solution:** Run Alembic migrations: `alembic upgrade head`

**Problem:** Field mapping incorrect
**Solution:** Check `db_config.fields` matches physical table schema

### StrategyPairManager Issues

**Problem:** Migration validation fails
**Solution:** Run missing migrations listed in error message

**Problem:** Capability compatibility error
**Solution:** Verify indexer produces what retriever requires (Epic 12)

**Problem:** Service reference not found
**Solution:** Check service name starts with `$` and exists in registry

---

## Documentation

Each story includes:
- Detailed requirements (functional and non-functional)
- Acceptance criteria with checkboxes
- Technical specifications with code examples
- Unit test examples with expected behavior
- Integration test scenarios
- Setup instructions
- Usage examples
- Performance characteristics
- Troubleshooting guide

---

## Related Epics

- **Epic 11:** Dependency Injection (service interfaces - prerequisite) ✅
- **Epic 12:** Pipeline Separation (capability system - prerequisite) ✅
- **Epic 16:** Database Consolidation (Alembic, PostgreSQL service - prerequisite) ✅
- **Epic 14:** CLI Development (will use strategy pairs)
- **Epics 4-7:** RAG Strategies (will be configured as strategy pairs)

---

## References

### Documentation
- [PyYAML Documentation](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [jsonschema Documentation](https://python-jsonschema.readthedocs.io/)
- [SQLAlchemy Connection Pooling](https://docs.sqlalchemy.org/en/20/core/pooling.html)
- [Alembic Migrations](https://alembic.sqlalchemy.org/en/latest/)
- [pgvector Extension](https://github.com/pgvector/pgvector)

### Design Patterns
- Service Registry Pattern
- Dependency Injection
- Strategy Pattern
- Factory Pattern
- Adapter Pattern (DatabaseContext)

---

## Support

For questions or issues with Epic 17:

1. Check story-specific documentation
2. Review test examples for usage patterns
3. Check troubleshooting section
4. Consult related epics for dependencies
5. Review configuration examples

---

## License

This project and all documentation is licensed under MIT License.
