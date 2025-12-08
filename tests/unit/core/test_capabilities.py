"""Unit tests for capability models."""

import pytest

from rag_factory.core.capabilities import (
    IndexCapability,
    IndexingResult,
    ValidationResult,
)
from rag_factory.services.dependencies import ServiceDependency


class TestIndexCapability:
    """Test suite for IndexCapability enum."""
    
    def test_all_capabilities_defined(self):
        """Test that all expected capabilities are defined."""
        expected_capabilities = {
            # Storage types
            'VECTORS',
            'KEYWORDS',
            'GRAPH',
            'FULL_DOCUMENT',
            # Structure types
            'CHUNKS',
            'HIERARCHY',
            'LATE_CHUNKS',
            # Storage backends
            'IN_MEMORY',
            'FILE_BACKED',
            'DATABASE',
            # Enrichment types
            'CONTEXTUAL',
            'METADATA',
        }
        
        actual_capabilities = {cap.name for cap in IndexCapability}
        assert actual_capabilities == expected_capabilities
    
    def test_enum_values_unique(self):
        """Test that all enum values are unique."""
        values = [cap.value for cap in IndexCapability]
        assert len(values) == len(set(values))
    
    def test_enum_members_accessible(self):
        """Test that enum members can be accessed by name."""
        assert IndexCapability.VECTORS
        assert IndexCapability.KEYWORDS
        assert IndexCapability.GRAPH
        assert IndexCapability.FULL_DOCUMENT
        assert IndexCapability.CHUNKS
        assert IndexCapability.HIERARCHY
        assert IndexCapability.LATE_CHUNKS
        assert IndexCapability.IN_MEMORY
        assert IndexCapability.FILE_BACKED
        assert IndexCapability.DATABASE
        assert IndexCapability.CONTEXTUAL
        assert IndexCapability.METADATA
    
    def test_enum_string_representation(self):
        """Test that enum members have readable string representation."""
        assert 'VECTORS' in str(IndexCapability.VECTORS)
        assert 'KEYWORDS' in str(IndexCapability.KEYWORDS)
    
    def test_enum_comparison(self):
        """Test that enum members can be compared."""
        assert IndexCapability.VECTORS == IndexCapability.VECTORS
        assert IndexCapability.VECTORS != IndexCapability.KEYWORDS
    
    def test_enum_in_set(self):
        """Test that enum members work correctly in sets."""
        capabilities = {IndexCapability.VECTORS, IndexCapability.DATABASE}
        assert IndexCapability.VECTORS in capabilities
        assert IndexCapability.KEYWORDS not in capabilities
        assert len(capabilities) == 2


class TestIndexingResult:
    """Test suite for IndexingResult dataclass."""
    
    def test_has_capability_present(self):
        """Test detection of present capability."""
        result = IndexingResult(
            capabilities={IndexCapability.VECTORS, IndexCapability.DATABASE},
            metadata={},
            document_count=10,
            chunk_count=50
        )
        assert result.has_capability(IndexCapability.VECTORS)
        assert result.has_capability(IndexCapability.DATABASE)
    
    def test_has_capability_absent(self):
        """Test detection of absent capability."""
        result = IndexingResult(
            capabilities={IndexCapability.VECTORS},
            metadata={},
            document_count=10,
            chunk_count=50
        )
        assert not result.has_capability(IndexCapability.KEYWORDS)
        assert not result.has_capability(IndexCapability.GRAPH)
    
    def test_is_compatible_with_satisfied(self):
        """Test when all requirements are met."""
        result = IndexingResult(
            capabilities={IndexCapability.VECTORS, IndexCapability.DATABASE, IndexCapability.CHUNKS},
            metadata={},
            document_count=10,
            chunk_count=50
        )
        # Single requirement
        assert result.is_compatible_with({IndexCapability.VECTORS})
        # Multiple requirements
        assert result.is_compatible_with({IndexCapability.VECTORS, IndexCapability.DATABASE})
        # All capabilities
        assert result.is_compatible_with({IndexCapability.VECTORS, IndexCapability.DATABASE, IndexCapability.CHUNKS})
    
    def test_is_compatible_with_unsatisfied(self):
        """Test when requirements are not met."""
        result = IndexingResult(
            capabilities={IndexCapability.VECTORS, IndexCapability.DATABASE},
            metadata={},
            document_count=10,
            chunk_count=50
        )
        # Missing one capability
        assert not result.is_compatible_with({IndexCapability.KEYWORDS})
        # Has some but not all
        assert not result.is_compatible_with({IndexCapability.VECTORS, IndexCapability.KEYWORDS})
    
    def test_is_compatible_with_empty_requirements(self):
        """Test with empty requirements (should always pass)."""
        result = IndexingResult(
            capabilities={IndexCapability.VECTORS},
            metadata={},
            document_count=10,
            chunk_count=50
        )
        assert result.is_compatible_with(set())
    
    def test_repr_format(self):
        """Test string representation format."""
        result = IndexingResult(
            capabilities={IndexCapability.VECTORS, IndexCapability.DATABASE},
            metadata={"duration": 12.5},
            document_count=100,
            chunk_count=450
        )
        repr_str = repr(result)
        
        # Check format
        assert "IndexingResult" in repr_str
        assert "capabilities=" in repr_str
        assert "docs=100" in repr_str
        assert "chunks=450" in repr_str
        
        # Check capabilities are present (sorted alphabetically)
        assert "DATABASE" in repr_str
        assert "VECTORS" in repr_str
    
    def test_repr_sorted_capabilities(self):
        """Test that capabilities are sorted in repr."""
        result = IndexingResult(
            capabilities={IndexCapability.VECTORS, IndexCapability.CHUNKS, IndexCapability.DATABASE},
            metadata={},
            document_count=10,
            chunk_count=50
        )
        repr_str = repr(result)
        
        # Capabilities should be sorted alphabetically
        # CHUNKS, DATABASE, VECTORS
        chunks_pos = repr_str.index("CHUNKS")
        database_pos = repr_str.index("DATABASE")
        vectors_pos = repr_str.index("VECTORS")
        
        assert chunks_pos < database_pos < vectors_pos


class TestValidationResult:
    """Test suite for ValidationResult dataclass."""
    
    def test_repr_valid(self):
        """Test repr for valid result."""
        result = ValidationResult(
            is_valid=True,
            missing_capabilities=set(),
            missing_services=set(),
            message="Validation passed",
            suggestions=[]
        )
        repr_str = repr(result)
        
        assert repr_str == "ValidationResult(valid=True)"
    
    def test_repr_missing_capabilities_only(self):
        """Test repr with only missing capabilities."""
        result = ValidationResult(
            is_valid=False,
            missing_capabilities={IndexCapability.VECTORS, IndexCapability.KEYWORDS},
            missing_services=set(),
            message="Missing capabilities",
            suggestions=[]
        )
        repr_str = repr(result)
        
        assert "ValidationResult(valid=False" in repr_str
        assert "capabilities:" in repr_str
        assert "KEYWORDS" in repr_str
        assert "VECTORS" in repr_str
        assert "services:" not in repr_str
    
    def test_repr_missing_services_only(self):
        """Test repr with only missing services."""
        result = ValidationResult(
            is_valid=False,
            missing_capabilities=set(),
            missing_services={ServiceDependency.EMBEDDING, ServiceDependency.LLM},
            message="Missing services",
            suggestions=[]
        )
        repr_str = repr(result)
        
        assert "ValidationResult(valid=False" in repr_str
        assert "services:" in repr_str
        assert "EMBEDDING" in repr_str
        assert "LLM" in repr_str
        assert "capabilities:" not in repr_str
    
    def test_repr_missing_both(self):
        """Test repr with both missing capabilities and services."""
        result = ValidationResult(
            is_valid=False,
            missing_capabilities={IndexCapability.VECTORS},
            missing_services={ServiceDependency.EMBEDDING},
            message="Missing both",
            suggestions=[]
        )
        repr_str = repr(result)
        
        assert "ValidationResult(valid=False" in repr_str
        assert "capabilities: VECTORS" in repr_str
        assert "services: EMBEDDING" in repr_str
        assert ";" in repr_str  # Separator between issues
    
    def test_repr_empty_missing(self):
        """Test repr when is_valid=False but no missing items."""
        result = ValidationResult(
            is_valid=False,
            missing_capabilities=set(),
            missing_services=set(),
            message="Invalid for other reason",
            suggestions=[]
        )
        repr_str = repr(result)
        
        assert repr_str == "ValidationResult(valid=False)"
    
    def test_repr_sorted_items(self):
        """Test that missing items are sorted in repr."""
        result = ValidationResult(
            is_valid=False,
            missing_capabilities={IndexCapability.VECTORS, IndexCapability.KEYWORDS, IndexCapability.GRAPH},
            missing_services={ServiceDependency.LLM, ServiceDependency.EMBEDDING, ServiceDependency.DATABASE},
            message="Multiple missing",
            suggestions=[]
        )
        repr_str = repr(result)
        
        # Check capabilities are sorted
        graph_pos = repr_str.index("GRAPH")
        keywords_pos = repr_str.index("KEYWORDS")
        vectors_pos = repr_str.index("VECTORS")
        assert graph_pos < keywords_pos < vectors_pos
        
        # Check services are sorted
        database_pos = repr_str.index("DATABASE")
        embedding_pos = repr_str.index("EMBEDDING")
        llm_pos = repr_str.index("LLM")
        assert database_pos < embedding_pos < llm_pos
