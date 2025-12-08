"""Unit tests for validation formatter."""

import pytest
from io import StringIO
from rich.console import Console

from rag_factory.cli.formatters.validation import (
    format_validation_results,
    _display_capabilities,
    _display_requirements,
    _display_service_requirements,
    _display_validation_status,
    _display_suggestions,
)
from rag_factory.core.capabilities import IndexCapability, ValidationResult
from rag_factory.core.pipeline import IndexingPipeline, RetrievalPipeline
from rag_factory.core.indexing_interface import IndexingContext
from rag_factory.core.retrieval_interface import RetrievalContext
from rag_factory.services.dependencies import ServiceDependency
from unittest.mock import Mock


@pytest.fixture
def console():
    """Create console with string output for testing."""
    string_io = StringIO()
    return Console(file=string_io, force_terminal=True, width=120)


@pytest.fixture
def mock_indexing_pipeline():
    """Create mock indexing pipeline."""
    pipeline = Mock(spec=IndexingPipeline)
    pipeline.get_capabilities = Mock(return_value={
        IndexCapability.VECTORS,
        IndexCapability.DATABASE,
        IndexCapability.CHUNKS
    })
    return pipeline


@pytest.fixture
def mock_retrieval_pipeline():
    """Create mock retrieval pipeline."""
    pipeline = Mock(spec=RetrievalPipeline)
    pipeline.get_requirements = Mock(return_value={IndexCapability.VECTORS})
    pipeline.get_service_requirements = Mock(return_value={
        ServiceDependency.EMBEDDING,
        ServiceDependency.DATABASE
    })
    return pipeline


class TestDisplayCapabilities:
    """Tests for _display_capabilities function."""

    def test_display_capabilities_with_capabilities(self, console, mock_indexing_pipeline):
        """Test displaying capabilities when present."""
        _display_capabilities(mock_indexing_pipeline, console)
        
        output = console.file.getvalue()
        assert "Indexing Capabilities" in output
        assert "VECTORS" in output
        assert "DATABASE" in output
        assert "CHUNKS" in output

    def test_display_capabilities_empty(self, console):
        """Test displaying capabilities when none present."""
        pipeline = Mock(spec=IndexingPipeline)
        pipeline.get_capabilities = Mock(return_value=set())
        
        _display_capabilities(pipeline, console)
        
        output = console.file.getvalue()
        assert "Indexing Capabilities" in output
        assert "No capabilities" in output


class TestDisplayRequirements:
    """Tests for _display_requirements function."""

    def test_display_requirements_all_met(self, console, mock_retrieval_pipeline, mock_indexing_pipeline):
        """Test displaying requirements when all are met."""
        _display_requirements(mock_retrieval_pipeline, mock_indexing_pipeline, console)
        
        output = console.file.getvalue()
        assert "Retrieval Requirements" in output
        assert "VECTORS" in output
        assert "met" in output

    def test_display_requirements_unmet(self, console, mock_indexing_pipeline):
        """Test displaying requirements when some are unmet."""
        retrieval_pipeline = Mock(spec=RetrievalPipeline)
        retrieval_pipeline.get_requirements = Mock(return_value={
            IndexCapability.VECTORS,
            IndexCapability.KEYWORDS  # Not in indexing capabilities
        })
        
        _display_requirements(retrieval_pipeline, mock_indexing_pipeline, console)
        
        output = console.file.getvalue()
        assert "Retrieval Requirements" in output
        assert "VECTORS" in output
        assert "KEYWORDS" in output
        assert "unmet" in output

    def test_display_requirements_empty(self, console, mock_indexing_pipeline):
        """Test displaying requirements when none present."""
        retrieval_pipeline = Mock(spec=RetrievalPipeline)
        retrieval_pipeline.get_requirements = Mock(return_value=set())
        
        _display_requirements(retrieval_pipeline, mock_indexing_pipeline, console)
        
        output = console.file.getvalue()
        assert "Retrieval Requirements" in output
        assert "No requirements" in output


class TestDisplayServiceRequirements:
    """Tests for _display_service_requirements function."""

    def test_display_service_requirements_all_available(self, console, mock_retrieval_pipeline):
        """Test displaying service requirements when all available."""
        validation = ValidationResult(
            is_valid=True,
            missing_capabilities=set(),
            missing_services=set(),
            message="Valid",
            suggestions=[]
        )
        
        _display_service_requirements(mock_retrieval_pipeline, validation, console)
        
        output = console.file.getvalue()
        assert "Service Requirements" in output
        assert "EMBEDDING" in output
        assert "DATABASE" in output
        assert "available" in output

    def test_display_service_requirements_missing(self, console, mock_retrieval_pipeline):
        """Test displaying service requirements when some missing."""
        validation = ValidationResult(
            is_valid=False,
            missing_capabilities=set(),
            missing_services={ServiceDependency.EMBEDDING},
            message="Missing services",
            suggestions=[]
        )
        
        _display_service_requirements(mock_retrieval_pipeline, validation, console)
        
        output = console.file.getvalue()
        assert "Service Requirements" in output
        assert "EMBEDDING" in output
        assert "unavailable" in output

    def test_display_service_requirements_empty(self, console):
        """Test displaying service requirements when none present."""
        retrieval_pipeline = Mock(spec=RetrievalPipeline)
        retrieval_pipeline.get_service_requirements = Mock(return_value=set())
        
        validation = ValidationResult(
            is_valid=True,
            missing_capabilities=set(),
            missing_services=set(),
            message="Valid",
            suggestions=[]
        )
        
        _display_service_requirements(retrieval_pipeline, validation, console)
        
        output = console.file.getvalue()
        assert "Service Requirements" in output
        assert "No service requirements" in output


class TestDisplayValidationStatus:
    """Tests for _display_validation_status function."""

    def test_display_valid_status(self, console):
        """Test displaying valid status."""
        validation = ValidationResult(
            is_valid=True,
            missing_capabilities=set(),
            missing_services=set(),
            message="Pipeline is valid",
            suggestions=[]
        )
        
        _display_validation_status(validation, console)
        
        output = console.file.getvalue()
        assert "VALID" in output
        assert "requirements are met" in output

    def test_display_invalid_status_missing_capabilities(self, console):
        """Test displaying invalid status with missing capabilities."""
        validation = ValidationResult(
            is_valid=False,
            missing_capabilities={IndexCapability.VECTORS, IndexCapability.KEYWORDS},
            missing_services=set(),
            message="Missing capabilities",
            suggestions=[]
        )
        
        _display_validation_status(validation, console)
        
        output = console.file.getvalue()
        assert "INVALID" in output
        assert "VECTORS" in output
        assert "KEYWORDS" in output

    def test_display_invalid_status_missing_services(self, console):
        """Test displaying invalid status with missing services."""
        validation = ValidationResult(
            is_valid=False,
            missing_capabilities=set(),
            missing_services={ServiceDependency.EMBEDDING, ServiceDependency.LLM},
            message="Missing services",
            suggestions=[]
        )
        
        _display_validation_status(validation, console)
        
        output = console.file.getvalue()
        assert "INVALID" in output
        assert "EMBEDDING" in output
        assert "LLM" in output


class TestDisplaySuggestions:
    """Tests for _display_suggestions function."""

    def test_display_suggestions(self, console):
        """Test displaying suggestions."""
        suggestions = [
            "Add VectorEmbeddingIndexing strategy",
            "Configure embedding service",
            "Enable database persistence"
        ]
        
        _display_suggestions(suggestions, console)
        
        output = console.file.getvalue()
        assert "Suggestions" in output
        assert "VectorEmbeddingIndexing" in output
        assert "embedding service" in output
        assert "database persistence" in output
        # Check for numbered list (rich console adds styling around numbers)
        assert "1" in output and "." in output
        assert "2" in output
        assert "3" in output


class TestFormatValidationResults:
    """Tests for format_validation_results function."""

    def test_format_valid_pipeline(self, console, mock_indexing_pipeline, mock_retrieval_pipeline):
        """Test formatting valid pipeline results."""
        validation = ValidationResult(
            is_valid=True,
            missing_capabilities=set(),
            missing_services=set(),
            message="Pipeline is valid",
            suggestions=[]
        )
        
        format_validation_results(
            validation,
            mock_indexing_pipeline,
            mock_retrieval_pipeline,
            console
        )
        
        output = console.file.getvalue()
        assert "Pipeline Validation Results" in output
        assert "Indexing Capabilities" in output
        assert "Retrieval Requirements" in output
        assert "Service Requirements" in output
        assert "VALID" in output

    def test_format_invalid_pipeline_with_suggestions(
        self,
        console,
        mock_indexing_pipeline,
        mock_retrieval_pipeline
    ):
        """Test formatting invalid pipeline with suggestions."""
        validation = ValidationResult(
            is_valid=False,
            missing_capabilities={IndexCapability.KEYWORDS},
            missing_services={ServiceDependency.LLM},
            message="Missing capabilities and services",
            suggestions=[
                "Add KeywordIndexing strategy",
                "Configure LLM service"
            ]
        )
        
        format_validation_results(
            validation,
            mock_indexing_pipeline,
            mock_retrieval_pipeline,
            console
        )
        
        output = console.file.getvalue()
        assert "Pipeline Validation Results" in output
        assert "INVALID" in output
        assert "Suggestions" in output
        assert "KeywordIndexing" in output
        assert "LLM service" in output
