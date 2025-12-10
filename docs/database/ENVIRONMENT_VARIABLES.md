# Environment Variables Reference

Complete reference for all environment variables used in RAG Factory.

---

## Database Configuration

### Production Variables

#### `DATABASE_URL`

**Required:** Yes (for production and development)  
**Format:** `postgresql://user:password@host:port/database_name`  
**Description:** Main database connection URL for production and development environments.

**Examples:**

```bash
# Local development
DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_factory

# Production
DATABASE_URL=postgresql://prod_user:secure_password@db.example.com:5432/rag_production

# VM development (accessing host machine)
HOST_IP=192.168.56.1
DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_factory
```

---

### Test Variables

#### `TEST_DATABASE_URL`

**Required:** Yes (for running tests)  
**Format:** `postgresql://user:password@host:port/database_name`  
**Description:** Test database connection URL. This is the **standard name** for test database configuration.

> [!IMPORTANT]
> Use `TEST_DATABASE_URL` (not `DATABASE_TEST_URL`). The old name is deprecated.

**Examples:**

```bash
# Local testing
TEST_DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_test

# CI/CD
TEST_DATABASE_URL=postgresql://test_user:test_pass@postgres:5432/test_db

# VM development
TEST_DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_test
```

#### `DATABASE_TEST_URL` (DEPRECATED)

**Status:** ⚠️ **DEPRECATED** - Will be removed in future version  
**Replacement:** Use `TEST_DATABASE_URL` instead

> [!WARNING]
> This variable name is deprecated. Please update your configuration to use `TEST_DATABASE_URL`.
> 
> **Migration:** Simply rename the variable in your `.env` file:
> ```bash
> # Old (deprecated)
> DATABASE_TEST_URL=postgresql://...
> 
> # New (standard)
> TEST_DATABASE_URL=postgresql://...
> ```

**Backward Compatibility:** The old name is still supported but will trigger deprecation warnings. Support will be removed in a future major version.

---

### Database Configuration with `DB_` Prefix

The `DatabaseConfig` class uses the `DB_` prefix for all configuration variables. These are automatically loaded from environment variables.

#### `DB_DATABASE_URL`

**Required:** Yes (alternative to `DATABASE_URL`)  
**Format:** Same as `DATABASE_URL`  
**Description:** Alternative name with `DB_` prefix for main database URL.

#### `DB_TEST_DATABASE_URL`

**Required:** Yes (alternative to `TEST_DATABASE_URL`)  
**Format:** Same as `TEST_DATABASE_URL`  
**Description:** Alternative name with `DB_` prefix for test database URL.

**Other DB_ Variables:**

- `DB_POOL_SIZE` - Connection pool size (default: 10)
- `DB_MAX_OVERFLOW` - Max overflow connections (default: 20)
- `DB_POOL_TIMEOUT` - Pool timeout in seconds (default: 30)
- `DB_POOL_RECYCLE` - Connection recycle time in seconds (default: 3600)
- `DB_ECHO` - Enable SQL query logging (default: false)
- `DB_POOL_PRE_PING` - Test connections before use (default: true)
- `DB_VECTOR_DIMENSIONS` - Vector embedding dimensions (default: 1536)

---

## Neo4j Configuration (Optional)

### `NEO4J_URI`

**Required:** No  
**Format:** `bolt://host:port` or `neo4j://host:port`  
**Description:** Neo4j database connection URI for graph-based features.

**Examples:**

```bash
# Local
NEO4J_URI=bolt://localhost:7687

# VM development
NEO4J_URI=bolt://${HOST_IP}:7687

# Production with encryption
NEO4J_URI=neo4j+s://neo4j.example.com:7687
```

### `NEO4J_USER`

**Required:** No (required if using Neo4j)  
**Default:** `neo4j`  
**Description:** Neo4j username.

### `NEO4J_PASSWORD`

**Required:** No (required if using Neo4j)  
**Description:** Neo4j password.

---

## LLM Configuration (Optional)

### OpenAI

#### `OPENAI_API_KEY`

**Required:** No (required for OpenAI LLM features)  
**Format:** `sk-...`  
**Description:** OpenAI API key for GPT models.

#### `OPENAI_MODEL`

**Required:** No  
**Default:** `gpt-3.5-turbo`  
**Description:** OpenAI model to use.

**Examples:** `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`

#### `OPENAI_API_BASE`

**Required:** No  
**Default:** `https://api.openai.com/v1`  
**Description:** Custom OpenAI API endpoint (useful for proxies or local LM Studio).

---

### Cohere

#### `COHERE_API_KEY`

**Required:** No (required for Cohere features)  
**Description:** Cohere API key.

---

### Local LM Studio

#### `LM_STUDIO_BASE_URL`

**Required:** No (required for LM Studio)  
**Format:** `http://host:port/v1`  
**Default:** `http://localhost:1234/v1`  
**Description:** LM Studio API endpoint (OpenAI-compatible).

**Examples:**

```bash
# Local
LM_STUDIO_BASE_URL=http://localhost:1234/v1

# VM development
LM_STUDIO_BASE_URL=http://${HOST_IP}:1234/v1
```

#### `LM_STUDIO_MODEL`

**Required:** No  
**Description:** Model name loaded in LM Studio.

---

## Embedding Models (Optional)

### `ONNX_MODEL_PATH`

**Required:** No  
**Description:** Path to ONNX embedding models for local inference.

### `HF_HOME`

**Required:** No  
**Default:** `~/.cache/huggingface`  
**Description:** HuggingFace cache directory for downloaded models.

---

## Test Execution Flags

### `RUN_DB_TESTS`

**Required:** No  
**Default:** `false`  
**Values:** `true`, `false`  
**Description:** Enable/disable database tests. Requires PostgreSQL with pgvector extension.

### `RUN_INTEGRATION_TESTS`

**Required:** No  
**Default:** `false`  
**Values:** `true`, `false`  
**Description:** Enable/disable integration tests. Requires all services to be available.

### `RUN_LLM_TESTS`

**Required:** No  
**Default:** `false`  
**Values:** `true`, `false`  
**Description:** Enable/disable LLM tests. Requires API keys to be configured.

---

## Development Settings

### `LOG_LEVEL`

**Required:** No  
**Default:** `INFO`  
**Values:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`  
**Description:** Logging level for the application.

### `PYTHONPATH`

**Required:** No  
**Description:** Python module search path (usually not needed).

---

## VM Development

### `HOST_IP`

**Required:** No (required for VM development)  
**Description:** IP address of the host machine when running in a VM.

**Common Values:**

- `10.0.2.2` - VirtualBox NAT (most common)
- `192.168.56.1` - VirtualBox Host-Only
- `192.168.122.1` - KVM/QEMU
- `172.16.0.1` - VMware NAT

**Finding Your Host IP:**

```bash
# Linux/Mac
ip route | grep default

# Or use the provided script
./find-host-ip.sh
```

**Usage:**

```bash
HOST_IP=192.168.56.1
DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_factory
TEST_DATABASE_URL=postgresql://rag_user:rag_password@${HOST_IP}:5432/rag_test
NEO4J_URI=bolt://${HOST_IP}:7687
LM_STUDIO_BASE_URL=http://${HOST_IP}:1234/v1
```

---

## Migration Guide

### Migrating from `DATABASE_TEST_URL` to `TEST_DATABASE_URL`

> [!WARNING]
> **Breaking Change:** The environment variable `DATABASE_TEST_URL` has been renamed to `TEST_DATABASE_URL` to follow standard naming conventions.

#### Quick Migration

```bash
# Update your .env file
sed -i 's/DATABASE_TEST_URL=/TEST_DATABASE_URL=/g' .env

# Verify tests still work
pytest tests/ -v
```

#### Manual Migration

1. Open your `.env` file
2. Find the line: `DATABASE_TEST_URL=...`
3. Rename to: `TEST_DATABASE_URL=...`
4. Save the file
5. Run tests to verify: `pytest tests/`

#### Transition Period

- **Both variable names work** during the transition period
- **Deprecation warnings** will appear if using the old name
- **Update at your convenience** before the next major release
- **No immediate action required** - backward compatibility is maintained

#### Example

```bash
# Before (deprecated)
DATABASE_TEST_URL=postgresql://rag_user:rag_password@localhost:5432/rag_test

# After (standard)
TEST_DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_test
```

---

## Troubleshooting

### "Missing required environment variables" Error

**Problem:** Application fails to start with missing variable error.

**Solution:**

1. Check which variables are missing in the error message
2. Add them to your `.env` file
3. Ensure `.env` is in the project root directory
4. Verify the file is not named `.env.txt` or similar

### Deprecation Warnings

**Problem:** Seeing warnings about deprecated variable names.

**Solution:**

1. Update your `.env` file to use the new variable names
2. See the Migration Guide above
3. Warnings are informational - the application will still work

### Database Connection Fails

**Problem:** Cannot connect to database.

**Solution:**

1. Verify database is running: `pg_isready -h localhost -p 5432`
2. Check credentials in `DATABASE_URL`
3. For VM development, verify `HOST_IP` is correct
4. Test connection: `psql $DATABASE_URL`

### Tests Skip Due to Missing `TEST_DATABASE_URL`

**Problem:** Database tests are skipped.

**Solution:**

1. Set `TEST_DATABASE_URL` in your `.env` file
2. Create test database: `createdb rag_test`
3. Install pgvector extension: `psql rag_test -c "CREATE EXTENSION vector;"`
4. Run tests: `pytest tests/integration/database/ -v`

---

## Environment File Templates

### Development (`.env`)

See [.env.example](file:///mnt/MCPProyects/ragTools/.env.example) for a complete template.

### Testing (`tests/.env.test`)

See [tests/.env.test](file:///mnt/MCPProyects/ragTools/tests/.env.test) for a test-specific template.

---

## See Also

- [Database README](file:///mnt/MCPProyects/ragTools/docs/database/README.md) - Database setup and migration guide
- [Main README](file:///mnt/MCPProyects/ragTools/README.md) - Project overview and quick start
- [Migration Audit](file:///mnt/MCPProyects/ragTools/docs/database/MIGRATION_AUDIT.md) - Database migration system audit
