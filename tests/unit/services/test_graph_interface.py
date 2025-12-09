"""Unit tests for IGraphService interface."""
import pytest
import inspect
from typing import Dict, Any, Optional, List
from rag_factory.services.interfaces import IGraphService

class TestIGraphServiceInterface:
    """Test suite for IGraphService interface definition."""
    
    def test_interface_is_abstract(self):
        """Test that IGraphService cannot be instantiated directly."""
        with pytest.raises(TypeError):
            IGraphService()
    
    def test_interface_requires_all_methods(self):
        """Test that concrete class must implement all abstract methods."""
        class IncompleteGraphService(IGraphService):
            async def create_node(self, label: str, properties: Dict[str, Any]) -> str:
                return "id"
        
        with pytest.raises(TypeError):
            IncompleteGraphService()
            
    def test_create_node_signature(self):
        """Test create_node method signature."""
        sig = inspect.signature(IGraphService.create_node)
        params = sig.parameters
        assert 'label' in params
        assert 'properties' in params
        assert sig.return_annotation == str

    def test_create_relationship_signature(self):
        """Test create_relationship method signature."""
        sig = inspect.signature(IGraphService.create_relationship)
        params = sig.parameters
        assert 'from_node_id' in params
        assert 'to_node_id' in params
        assert 'relationship_type' in params
        assert 'properties' in params
        assert sig.return_annotation is None # Returns None

    def test_query_signature(self):
        """Test query method signature."""
        sig = inspect.signature(IGraphService.query)
        params = sig.parameters
        assert 'cypher_query' in params
        assert 'parameters' in params
        assert sig.return_annotation == List[Dict[str, Any]]

    def test_minimal_concrete_implementation(self):
        """Test a minimal concrete implementation works."""
        class MinimalGraphService(IGraphService):
            async def create_node(self, label: str, properties: Dict[str, Any]) -> str:
                return "node_1"
            
            async def create_relationship(
                self,
                from_node_id: str,
                to_node_id: str,
                relationship_type: str,
                properties: Optional[Dict[str, Any]] = None
            ) -> None:
                pass
            
            async def query(
                self,
                cypher_query: str,
                parameters: Optional[Dict[str, Any]] = None
            ) -> List[Dict[str, Any]]:
                return []

        service = MinimalGraphService()
        assert isinstance(service, IGraphService)
