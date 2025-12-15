"""Integration tests for configuration system."""

import pytest
import os
import yaml
from pathlib import Path
from rag_factory.config import (
    load_yaml_with_validation,
    EnvResolver,
    ConfigValidationError,
)


@pytest.mark.integration
class TestConfigIntegration:
    """Integration tests for configuration loading and validation."""

    def test_load_real_services_yaml(self, tmp_path):
        """Test loading real services.yaml file."""
        services_yaml = tmp_path / "services.yaml"
        services_yaml.write_text("""
version: "1.0.0"

services:
  embedding_local:
    name: "local-onnx-minilm"
    type: "embedding"
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
""")

        os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost:5432/test"

        try:
            # Load and validate
            config = load_yaml_with_validation(str(services_yaml), "service_registry")

            # Resolve environment variables
            config = EnvResolver.resolve(config)

            assert "services" in config
            assert "embedding_local" in config["services"]
            assert config["services"]["db_main"]["connection_string"] == "postgresql://user:pass@localhost:5432/test"
        finally:
            del os.environ["DATABASE_URL"]

    def test_load_real_strategy_pair_yaml(self, tmp_path):
        """Test loading real strategy pair YAML."""
        # Create services file
        services_yaml = tmp_path / "services.yaml"
        services_yaml.write_text("""
services:
  embedding1:
    name: "test-embedding"
    type: "embedding"
    provider: "onnx"
    model: "test-model"
    dimensions: 384
""")

        # Create strategy pair file
        strategy_yaml = tmp_path / "test-pair.yaml"
        strategy_yaml.write_text("""
strategy_name: "test-pair"
version: "1.0.0"
description: "Test strategy pair"

indexer:
  strategy: "TestIndexer"
  services:
    embedding: "$embedding1"
  db_config:
    tables:
      chunks: "test_chunks"
  config:
    chunk_size: 512

retriever:
  strategy: "TestRetriever"
  services:
    embedding: "$embedding1"
  config:
    top_k: 5
""")

        # Load service registry
        with open(services_yaml) as f:
            service_registry = yaml.safe_load(f)

        # Load and validate strategy pair
        config = load_yaml_with_validation(
            str(strategy_yaml),
            "strategy_pair",
            service_registry=service_registry
        )

        assert config["strategy_name"] == "test-pair"
        assert config["indexer"]["services"]["embedding"] == "$embedding1"

    def test_full_workflow_with_env_vars(self, tmp_path):
        """Test complete workflow with environment variables."""
        # Create services file with env vars
        services_yaml = tmp_path / "services.yaml"
        services_yaml.write_text("""
services:
  llm_local:
    name: "local-llm"
    type: "llm"
    url: "${LLM_URL:-http://localhost:1234/v1}"
    model: "${LLM_MODEL:-local-model}"
    temperature: 0.7

  embedding_local:
    name: "local-embedding"
    type: "embedding"
    provider: "onnx"
    model: "${EMBEDDING_MODEL}"
    dimensions: 384

  db_main:
    name: "main-db"
    type: "postgres"
    host: "${DB_HOST:-localhost}"
    port: 5432
    database: "${DB_NAME:-rag_factory}"
    user: "${DB_USER:-postgres}"
    password: "${DB_PASSWORD:?Database password is required}"
""")

        # Set environment variables
        os.environ["EMBEDDING_MODEL"] = "test-model"
        os.environ["DB_PASSWORD"] = "secret"

        try:
            # Load and validate
            config = load_yaml_with_validation(str(services_yaml), "service_registry")

            # Resolve environment variables
            config = EnvResolver.resolve(config)

            # Check resolved values
            assert config["services"]["llm_local"]["url"] == "http://localhost:1234/v1"
            assert config["services"]["llm_local"]["model"] == "local-model"
            assert config["services"]["embedding_local"]["model"] == "test-model"
            assert config["services"]["db_main"]["host"] == "localhost"
            assert config["services"]["db_main"]["database"] == "rag_factory"
            assert config["services"]["db_main"]["password"] == "secret"
        finally:
            del os.environ["EMBEDDING_MODEL"]
            del os.environ["DB_PASSWORD"]

    def test_missing_required_env_var(self, tmp_path):
        """Test error when required environment variable is missing."""
        services_yaml = tmp_path / "services.yaml"
        services_yaml.write_text("""
services:
  db_main:
    name: "main-db"
    type: "postgres"
    connection_string: "${DATABASE_URL:?Database URL is required}"
""")

        # Load config (should succeed)
        config = load_yaml_with_validation(str(services_yaml), "service_registry")

        # Resolve should fail
        from rag_factory.config import EnvironmentVariableError
        with pytest.raises(EnvironmentVariableError) as exc_info:
            EnvResolver.resolve(config)

        assert "DATABASE_URL" in str(exc_info.value)
        assert "required" in str(exc_info.value).lower()

    def test_service_reference_validation_integration(self, tmp_path):
        """Test service reference validation in real scenario."""
        # Create services file
        services_yaml = tmp_path / "services.yaml"
        services_yaml.write_text("""
services:
  embedding1:
    name: "embedding-service"
    type: "embedding"
    provider: "onnx"
    model: "test-model"
  
  db1:
    name: "database-service"
    type: "postgres"
    connection_string: "postgresql://localhost/test"
""")

        # Create strategy pair with invalid reference
        strategy_yaml = tmp_path / "test-pair.yaml"
        strategy_yaml.write_text("""
strategy_name: "test-pair"
version: "1.0.0"

indexer:
  strategy: "TestIndexer"
  services:
    embedding: "$nonexistent_service"

retriever:
  strategy: "TestRetriever"
  services:
    embedding: "$embedding1"
""")

        # Load service registry
        with open(services_yaml) as f:
            service_registry = yaml.safe_load(f)

        # Should fail validation
        with pytest.raises(ConfigValidationError) as exc_info:
            load_yaml_with_validation(
                str(strategy_yaml),
                "strategy_pair",
                service_registry=service_registry
            )

        assert "nonexistent_service" in str(exc_info.value)

    def test_plaintext_secret_warning_integration(self, tmp_path):
        """Test plaintext secret detection in real scenario."""
        services_yaml = tmp_path / "services.yaml"
        services_yaml.write_text("""
services:
  llm1:
    name: "test-llm"
    type: "llm"
    url: "http://localhost:1234/v1"
    api_key: "sk-plaintext-key-should-warn"
    model: "test-model"
""")

        # Should load but generate warnings
        import io
        import sys
        
        # Capture stdout to check for warnings
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            config = load_yaml_with_validation(str(services_yaml), "service_registry")
            output = captured_output.getvalue()
            
            # Should have warning about plaintext secret
            assert "WARNING" in output or "plaintext" in output.lower()
        finally:
            sys.stdout = old_stdout

    def test_complex_strategy_pair_with_all_features(self, tmp_path):
        """Test complex strategy pair with all features."""
        # Create services file
        services_yaml = tmp_path / "services.yaml"
        services_yaml.write_text("""
services:
  embedding1:
    name: "embedding-service"
    type: "embedding"
    provider: "onnx"
    model: "test-model"
    dimensions: 384
  
  llm1:
    name: "llm-service"
    type: "llm"
    url: "http://localhost:1234/v1"
    api_key: "${LLM_API_KEY:-test-key}"
    model: "test-model"
  
  db1:
    name: "database-service"
    type: "postgres"
    connection_string: "${DATABASE_URL:-postgresql://localhost/test}"
""")

        # Create complex strategy pair
        strategy_yaml = tmp_path / "complex-pair.yaml"
        strategy_yaml.write_text("""
strategy_name: "complex-pair"
version: "1.0.0"
description: "Complex strategy with all features"

tags:
  - "semantic"
  - "advanced"
  - "production"

indexer:
  strategy: "ComplexIndexer"
  services:
    embedding: "$embedding1"
    llm: "$llm1"
    database: "$db1"
  
  db_config:
    tables:
      documents: "documents"
      chunks: "chunks"
      metadata: "chunk_metadata"
    fields:
      chunk_id: "id"
      content: "content"
      embedding: "embedding_vector"
  
  config:
    chunk_size: 512
    chunk_overlap: 50
    batch_size: 32
    use_llm_enhancement: true

retriever:
  strategy: "ComplexRetriever"
  services:
    embedding: "$embedding1"
    llm: "$llm1"
    database: "$db1"
  
  db_config:
    tables:
      chunks: "chunks"
    fields:
      chunk_id: "id"
      content: "content"
      embedding: "embedding_vector"
  
  config:
    top_k: 10
    similarity_threshold: 0.7
    use_reranking: true

migrations:
  required_revisions:
    - "001_create_documents_table"
    - "002_create_chunks_table"
    - "003_add_pgvector_extension"
    - "004_create_metadata_table"

expected_schema:
  tables:
    - "documents"
    - "chunks"
    - "chunk_metadata"
  indexes:
    - "idx_chunks_embedding_vector"
    - "idx_chunks_document_id"
    - "idx_metadata_chunk_id"
  extensions:
    - "vector"
""")

        # Load service registry
        with open(services_yaml) as f:
            service_registry = yaml.safe_load(f)

        # Load and validate strategy pair
        config = load_yaml_with_validation(
            str(strategy_yaml),
            "strategy_pair",
            service_registry=service_registry
        )

        # Resolve environment variables
        config = EnvResolver.resolve(config)

        # Verify all sections are present
        assert config["strategy_name"] == "complex-pair"
        assert len(config["tags"]) == 3
        assert "indexer" in config
        assert "retriever" in config
        assert "migrations" in config
        assert "expected_schema" in config
        
        # Verify service references
        assert config["indexer"]["services"]["embedding"] == "$embedding1"
        assert config["indexer"]["services"]["llm"] == "$llm1"
        assert config["indexer"]["services"]["database"] == "$db1"
        
        # Verify db_config
        assert config["indexer"]["db_config"]["tables"]["chunks"] == "chunks"
        assert config["indexer"]["db_config"]["fields"]["embedding"] == "embedding_vector"
        
        # Verify migrations
        assert len(config["migrations"]["required_revisions"]) == 4
        
        # Verify expected schema
        assert "vector" in config["expected_schema"]["extensions"]


@pytest.mark.integration
class TestExampleConfigurations:
    """Test the example configuration files."""

    def test_example_services_yaml_is_valid(self):
        """Test that example services.yaml is valid."""
        example_path = Path(__file__).parent.parent.parent.parent / "rag_factory" / "config" / "examples" / "services.yaml"
        
        if not example_path.exists():
            pytest.skip("Example services.yaml not found")
        
        # Load without environment variables (should fail on resolution but pass validation)
        with open(example_path) as f:
            config = yaml.safe_load(f)
        
        from rag_factory.config import ConfigValidator
        validator = ConfigValidator()
        
        # Should validate successfully
        warnings = validator.validate_services_yaml(config, str(example_path))
        assert isinstance(warnings, list)

    def test_example_strategy_pairs_are_valid(self):
        """Test that example strategy pair files are valid."""
        examples_dir = Path(__file__).parent.parent.parent.parent / "rag_factory" / "config" / "examples"
        
        if not examples_dir.exists():
            pytest.skip("Examples directory not found")
        
        # Find all strategy pair YAML files
        strategy_files = list(examples_dir.glob("*-pair.yaml"))
        
        if not strategy_files:
            pytest.skip("No example strategy pair files found")
        
        # Load example services for reference validation
        services_path = examples_dir / "services.yaml"
        if services_path.exists():
            with open(services_path) as f:
                service_registry = yaml.safe_load(f)
        else:
            service_registry = None
        
        from rag_factory.config import ConfigValidator
        validator = ConfigValidator()
        
        # Validate each strategy pair
        for strategy_file in strategy_files:
            with open(strategy_file) as f:
                config = yaml.safe_load(f)
            
            warnings = validator.validate_strategy_pair_yaml(
                config,
                service_registry=service_registry,
                file_path=str(strategy_file)
            )
            assert isinstance(warnings, list)
