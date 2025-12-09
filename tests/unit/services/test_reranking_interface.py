"""Unit tests for IRerankingService interface."""
import pytest
import inspect
from typing import List, Tuple
from rag_factory.services.interfaces import IRerankingService

class TestIRerankingServiceInterface:
    """Test suite for IRerankingService interface definition."""
    
    def test_interface_is_abstract(self):
        """Test that IRerankingService cannot be instantiated directly."""
        with pytest.raises(TypeError):
            IRerankingService()
    
    def test_interface_requires_rerank_method(self):
        """Test that concrete class must implement rerank method."""
        class IncompleteReranker(IRerankingService):
            pass
        
        with pytest.raises(TypeError):
            IncompleteReranker()
    
    def test_rerank_method_signature(self):
        """Test rerank method has correct signature."""
        sig = inspect.signature(IRerankingService.rerank)
        params = sig.parameters
        
        assert 'query' in params
        assert 'documents' in params
        assert 'top_k' in params
        assert sig.return_annotation == List[Tuple[int, float]]
    
    def test_minimal_concrete_implementation(self):
        """Test a minimal concrete implementation works."""
        class MinimalReranker(IRerankingService):
            async def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
                return []
        
        reranker = MinimalReranker()
        assert isinstance(reranker, IRerankingService)
        # We can't easily test async methods without an event loop in a simple unit test unless we use pytest-asyncio
        # But here we just want to verify instantiation and type
