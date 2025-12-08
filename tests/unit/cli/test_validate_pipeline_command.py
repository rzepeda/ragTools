"""Unit tests for validate-pipeline command."""

import pytest
from typer.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock

from rag_factory.cli.main import app
from rag_factory.core.capabilities import IndexCapability, ValidationResult
from rag_factory.services.dependencies import ServiceDependency


@pytest.fixture
def cli_runner():
    """Create CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_factory():
    """Create mock factory."""
    factory = Mock()
    factory.dependencies = Mock()
    factory.dependencies.database_service = None
    factory.dependencies.embedding_service = None
    factory.dependencies.llm_service = None
    factory.dependencies.reranker_service = None
    return factory


@pytest.fixture
def mock_indexing_strategy():
    """Create mock indexing strategy."""
    strategy = Mock()
    strategy.produces = Mock(return_value={IndexCapability.VECTORS, IndexCapability.DATABASE})
    strategy.requires_services = Mock(return_value={ServiceDependency.EMBEDDING, ServiceDependency.DATABASE})
    return strategy


@pytest.fixture
def mock_retrieval_strategy():
    """Create mock retrieval strategy."""
    strategy = Mock()
    strategy.requires = Mock(return_value={IndexCapability.VECTORS})
    strategy.requires_services = Mock(return_value={ServiceDependency.EMBEDDING, ServiceDependency.DATABASE})
    return strategy


class TestValidatePipelineCommand:
    """Tests for validate-pipeline command."""

    def test_command_help(self, cli_runner):
        """Test command help displays correctly."""
        result = cli_runner.invoke(app, ["validate-pipeline", "--help"])
        
        assert result.exit_code == 0
        assert "validate-pipeline" in result.stdout.lower() or "validate pipeline" in result.stdout.lower()
        assert "--indexing" in result.stdout
        assert "--retrieval" in result.stdout
        assert "--config" in result.stdout

    def test_missing_indexing_argument(self, cli_runner):
        """Test error when indexing argument is missing."""
        result = cli_runner.invoke(app, ["validate-pipeline", "--retrieval", "reranking"])
        
        assert result.exit_code != 0

    def test_missing_retrieval_argument(self, cli_runner):
        """Test error when retrieval argument is missing."""
        result = cli_runner.invoke(app, ["validate-pipeline", "--indexing", "vector_embedding"])
        
        assert result.exit_code != 0

    @patch("rag_factory.cli.commands.validate_pipeline.RAGFactory")
    @patch("rag_factory.cli.commands.validate_pipeline.IndexingPipeline")
    @patch("rag_factory.cli.commands.validate_pipeline.RetrievalPipeline")
    def test_valid_pipeline_returns_zero(
        self,
        mock_retrieval_pipeline_cls,
        mock_indexing_pipeline_cls,
        mock_factory_cls,
        cli_runner,
        mock_factory,
        mock_indexing_strategy,
        mock_retrieval_strategy
    ):
        """Test valid pipeline returns exit code 0."""
        # Setup mocks
        mock_factory_cls.return_value = mock_factory
        mock_factory.create_strategy = Mock(side_effect=[mock_indexing_strategy, mock_retrieval_strategy])
        
        # Mock pipelines
        mock_indexing_pipeline = Mock()
        mock_indexing_pipeline.get_capabilities = Mock(return_value={IndexCapability.VECTORS, IndexCapability.DATABASE})
        mock_indexing_pipeline_cls.return_value = mock_indexing_pipeline
        
        mock_retrieval_pipeline = Mock()
        mock_retrieval_pipeline.get_requirements = Mock(return_value={IndexCapability.VECTORS})
        mock_retrieval_pipeline.get_service_requirements = Mock(return_value={ServiceDependency.EMBEDDING})
        mock_retrieval_pipeline_cls.return_value = mock_retrieval_pipeline
        
        # Mock validation result (valid)
        validation_result = ValidationResult(
            is_valid=True,
            missing_capabilities=set(),
            missing_services=set(),
            message="Pipeline is valid",
            suggestions=[]
        )
        mock_factory.validate_pipeline = Mock(return_value=validation_result)
        
        # Run command
        result = cli_runner.invoke(
            app,
            ["validate-pipeline", "--indexing", "vector_embedding", "--retrieval", "reranking"]
        )
        
        assert result.exit_code == 0

    @patch("rag_factory.cli.commands.validate_pipeline.RAGFactory")
    @patch("rag_factory.cli.commands.validate_pipeline.IndexingPipeline")
    @patch("rag_factory.cli.commands.validate_pipeline.RetrievalPipeline")
    def test_invalid_pipeline_returns_nonzero(
        self,
        mock_retrieval_pipeline_cls,
        mock_indexing_pipeline_cls,
        mock_factory_cls,
        cli_runner,
        mock_factory,
        mock_indexing_strategy,
        mock_retrieval_strategy
    ):
        """Test invalid pipeline returns non-zero exit code."""
        # Setup mocks
        mock_factory_cls.return_value = mock_factory
        mock_factory.create_strategy = Mock(side_effect=[mock_indexing_strategy, mock_retrieval_strategy])
        
        # Mock pipelines
        mock_indexing_pipeline = Mock()
        mock_indexing_pipeline.get_capabilities = Mock(return_value={IndexCapability.KEYWORDS})
        mock_indexing_pipeline_cls.return_value = mock_indexing_pipeline
        
        mock_retrieval_pipeline = Mock()
        mock_retrieval_pipeline.get_requirements = Mock(return_value={IndexCapability.VECTORS})
        mock_retrieval_pipeline.get_service_requirements = Mock(return_value=set())
        mock_retrieval_pipeline_cls.return_value = mock_retrieval_pipeline
        
        # Mock validation result (invalid)
        validation_result = ValidationResult(
            is_valid=False,
            missing_capabilities={IndexCapability.VECTORS},
            missing_services=set(),
            message="Missing capabilities",
            suggestions=["Add VectorEmbeddingIndexing strategy"]
        )
        mock_factory.validate_pipeline = Mock(return_value=validation_result)
        
        # Run command
        result = cli_runner.invoke(
            app,
            ["validate-pipeline", "--indexing", "keyword_extraction", "--retrieval", "vector_search"]
        )
        
        assert result.exit_code == 1

    @patch("rag_factory.cli.commands.validate_pipeline.RAGFactory")
    def test_invalid_strategy_name_error(
        self,
        mock_factory_cls,
        cli_runner,
        mock_factory
    ):
        """Test error handling for invalid strategy name."""
        # Setup mock to raise error
        mock_factory_cls.return_value = mock_factory
        mock_factory.create_strategy = Mock(side_effect=Exception("Strategy not found"))
        
        # Run command
        result = cli_runner.invoke(
            app,
            ["validate-pipeline", "--indexing", "invalid_strategy", "--retrieval", "reranking"]
        )
        
        assert result.exit_code == 1

    def test_parse_multiple_strategies(self, cli_runner):
        """Test parsing multiple comma-separated strategies."""
        # This test verifies the command accepts multiple strategies
        # We can't fully test without mocking, but we can verify it doesn't crash on parsing
        with patch("rag_factory.cli.commands.validate_pipeline.RAGFactory") as mock_factory_cls:
            mock_factory = Mock()
            mock_factory_cls.return_value = mock_factory
            mock_factory.dependencies = Mock()
            mock_factory.dependencies.database_service = None
            mock_factory.dependencies.embedding_service = None
            mock_factory.dependencies.llm_service = None
            mock_factory.dependencies.reranker_service = None
            
            # Make create_strategy fail to avoid complex mocking
            mock_factory.create_strategy = Mock(side_effect=Exception("Test"))
            
            result = cli_runner.invoke(
                app,
                ["validate-pipeline", "--indexing", "strategy1,strategy2", "--retrieval", "strategy3,strategy4"]
            )
            
            # Should fail on strategy creation, not parsing
            assert "strategy1" in str(mock_factory.create_strategy.call_args_list) or result.exit_code == 1

    @patch("rag_factory.cli.commands.validate_pipeline.validate_config_file")
    @patch("rag_factory.cli.commands.validate_pipeline.RAGFactory")
    @patch("rag_factory.cli.commands.validate_pipeline.IndexingPipeline")
    @patch("rag_factory.cli.commands.validate_pipeline.RetrievalPipeline")
    def test_config_file_loading(
        self,
        mock_retrieval_pipeline_cls,
        mock_indexing_pipeline_cls,
        mock_factory_cls,
        mock_validate_config,
        cli_runner,
        mock_factory,
        mock_indexing_strategy,
        mock_retrieval_strategy
    ):
        """Test config file is loaded when provided."""
        # Setup mocks
        mock_validate_config.return_value = {"some": "config"}
        mock_factory_cls.return_value = mock_factory
        mock_factory.create_strategy = Mock(side_effect=[mock_indexing_strategy, mock_retrieval_strategy])
        
        # Mock pipelines
        mock_indexing_pipeline = Mock()
        mock_indexing_pipeline.get_capabilities = Mock(return_value={IndexCapability.VECTORS})
        mock_indexing_pipeline_cls.return_value = mock_indexing_pipeline
        
        mock_retrieval_pipeline = Mock()
        mock_retrieval_pipeline.get_requirements = Mock(return_value={IndexCapability.VECTORS})
        mock_retrieval_pipeline.get_service_requirements = Mock(return_value=set())
        mock_retrieval_pipeline_cls.return_value = mock_retrieval_pipeline
        
        # Mock validation
        validation_result = ValidationResult(
            is_valid=True,
            missing_capabilities=set(),
            missing_services=set(),
            message="Valid",
            suggestions=[]
        )
        mock_factory.validate_pipeline = Mock(return_value=validation_result)
        
        # Run command with config
        result = cli_runner.invoke(
            app,
            [
                "validate-pipeline",
                "--indexing", "vector_embedding",
                "--retrieval", "reranking",
                "--config", "config.yaml"
            ]
        )
        
        # Verify config was loaded
        mock_validate_config.assert_called_once_with("config.yaml")
