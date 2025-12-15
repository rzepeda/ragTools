"""Unit tests for service interfaces.

This module tests that all service interfaces are properly defined
as abstract base classes and enforce implementation of abstract methods.
"""

import pytest
from typing import AsyncIterator, List, Dict, Any, Tuple, Optional

from rag_factory.services.interfaces import (
    ILLMService,
    IEmbeddingService,
    IGraphService,
    IRerankingService,
    IDatabaseService,
)


# ============================================================================
# ILLMService Tests
# ============================================================================


def test_illm_service_requires_implementation():
    """Test that ILLMService requires all methods to be implemented."""
    with pytest.raises(TypeError):

        class IncompleteService(ILLMService):
            pass

        IncompleteService()


def test_illm_service_requires_complete_method():
    """Test that ILLMService requires complete method."""
    with pytest.raises(TypeError):

        class PartialService(ILLMService):
            async def stream_complete(
                self,
                prompt: str,
                max_tokens: Optional[int] = None,
                temperature: float = 0.7,
                **kwargs
            ) -> AsyncIterator[str]:
                yield "test"

        PartialService()


def test_illm_service_requires_stream_complete_method():
    """Test that ILLMService requires stream_complete method."""
    with pytest.raises(TypeError):

        class PartialService(ILLMService):
            async def complete(
                self,
                prompt: str,
                max_tokens: Optional[int] = None,
                temperature: float = 0.7,
                **kwargs
            ) -> str:
                return "test"

        PartialService()


@pytest.mark.asyncio
async def test_illm_service_mock_implementation():
    """Test that complete ILLMService implementation works."""

    class MockLLMService(ILLMService):
        async def complete(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: float = 0.7,
            **kwargs
        ) -> str:
            return f"Response to: {prompt}"

        async def stream_complete(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: float = 0.7,
            **kwargs
        ) -> AsyncIterator[str]:
            yield "test"
            yield " response"

    service = MockLLMService()
    assert service is not None

    # Test complete method
    response = await service.complete("Hello")
    assert response == "Response to: Hello"

    # Test stream_complete method
    chunks = []
    async for chunk in service.stream_complete("Hello"):
        chunks.append(chunk)
    assert chunks == ["test", " response"]


def test_illm_service_has_docstring():
    """Test that ILLMService has docstring."""
    assert ILLMService.__doc__ is not None
    assert len(ILLMService.__doc__) > 0


# ============================================================================
# IEmbeddingService Tests
# ============================================================================


def test_iembedding_service_requires_implementation():
    """Test that IEmbeddingService requires all methods to be implemented."""
    with pytest.raises(TypeError):

        class IncompleteService(IEmbeddingService):
            pass

        IncompleteService()


def test_iembedding_service_requires_embed_method():
    """Test that IEmbeddingService requires embed method."""
    with pytest.raises(TypeError):

        class PartialService(IEmbeddingService):
            async def embed_batch(self, texts: List[str]) -> List[List[float]]:
                return [[0.1, 0.2]]

            def get_dimension(self) -> int:
                return 2

        PartialService()


@pytest.mark.asyncio
async def test_iembedding_service_mock_implementation():
    """Test that complete IEmbeddingService implementation works."""

    class MockEmbeddingService(IEmbeddingService):
        async def embed(self, text: str) -> List[float]:
            return [0.1, 0.2, 0.3]

        async def embed_batch(self, texts: List[str]) -> List[List[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

        def get_dimension(self) -> int:
            return 3

    service = MockEmbeddingService()
    assert service is not None

    # Test embed method
    embedding = await service.embed("test")
    assert embedding == [0.1, 0.2, 0.3]

    # Test embed_batch method
    embeddings = await service.embed_batch(["test1", "test2"])
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]

    # Test get_dimension method
    assert service.get_dimension() == 3


def test_iembedding_service_has_docstring():
    """Test that IEmbeddingService has docstring."""
    assert IEmbeddingService.__doc__ is not None
    assert len(IEmbeddingService.__doc__) > 0


# ============================================================================
# IGraphService Tests
# ============================================================================


def test_igraph_service_requires_implementation():
    """Test that IGraphService requires all methods to be implemented."""
    with pytest.raises(TypeError):

        class IncompleteService(IGraphService):
            pass

        IncompleteService()


def test_igraph_service_requires_create_node_method():
    """Test that IGraphService requires create_node method."""
    with pytest.raises(TypeError):

        class PartialService(IGraphService):
            async def create_relationship(
                self,
                from_node_id: str,
                to_node_id: str,
                relationship_type: str,
                properties: Optional[Dict[str, Any]] = None,
            ) -> None:
                pass

            async def query(
                self, cypher_query: str, parameters: Optional[Dict[str, Any]] = None
            ) -> List[Dict[str, Any]]:
                return []

        PartialService()


@pytest.mark.asyncio
async def test_igraph_service_mock_implementation():
    """Test that complete IGraphService implementation works."""

    class MockGraphService(IGraphService):
        def __init__(self):
            self.nodes = {}
            self.relationships = []
            self.next_id = 1

        async def create_node(self, label: str, properties: Dict[str, Any]) -> str:
            node_id = f"node_{self.next_id}"
            self.next_id += 1
            self.nodes[node_id] = {"label": label, "properties": properties}
            return node_id

        async def create_relationship(
            self,
            from_node_id: str,
            to_node_id: str,
            relationship_type: str,
            properties: Optional[Dict[str, Any]] = None,
        ) -> None:
            self.relationships.append(
                {
                    "from": from_node_id,
                    "to": to_node_id,
                    "type": relationship_type,
                    "properties": properties or {},
                }
            )

        async def query(
            self, cypher_query: str, parameters: Optional[Dict[str, Any]] = None
        ) -> List[Dict[str, Any]]:
            return [{"result": "mock"}]

    service = MockGraphService()
    assert service is not None

    # Test create_node method
    node_id = await service.create_node("Person", {"name": "Alice"})
    assert node_id == "node_1"
    assert "node_1" in service.nodes

    # Test create_relationship method
    node_id2 = await service.create_node("Person", {"name": "Bob"})
    await service.create_relationship(node_id, node_id2, "KNOWS")
    assert len(service.relationships) == 1

    # Test query method
    results = await service.query("MATCH (n) RETURN n")
    assert results == [{"result": "mock"}]


def test_igraph_service_has_docstring():
    """Test that IGraphService has docstring."""
    assert IGraphService.__doc__ is not None
    assert len(IGraphService.__doc__) > 0


# ============================================================================
# IRerankingService Tests
# ============================================================================


def test_ireranking_service_requires_implementation():
    """Test that IRerankingService requires all methods to be implemented."""
    with pytest.raises(TypeError):

        class IncompleteService(IRerankingService):
            pass

        IncompleteService()


@pytest.mark.asyncio
async def test_ireranking_service_mock_implementation():
    """Test that complete IRerankingService implementation works."""

    class MockRerankingService(IRerankingService):
        async def rerank(
            self, query: str, documents: List[str], top_k: int = 5
        ) -> List[Tuple[int, float]]:
            # Simple mock: return indices in reverse order with decreasing scores
            results = [(i, 1.0 - i * 0.1) for i in range(min(top_k, len(documents)))]
            return results

    service = MockRerankingService()
    assert service is not None

    # Test rerank method
    results = await service.rerank("query", ["doc1", "doc2", "doc3"], top_k=2)
    assert len(results) == 2
    assert results[0] == (0, 1.0)
    assert results[1] == (1, 0.9)


def test_ireranking_service_has_docstring():
    """Test that IRerankingService has docstring."""
    assert IRerankingService.__doc__ is not None
    assert len(IRerankingService.__doc__) > 0


# ============================================================================
# IDatabaseService Tests
# ============================================================================


def test_idatabase_service_requires_implementation():
    """Test that IDatabaseService requires all methods to be implemented."""
    with pytest.raises(TypeError):

        class IncompleteService(IDatabaseService):
            pass

        IncompleteService()


def test_idatabase_service_requires_store_chunks_method():
    """Test that IDatabaseService requires store_chunks method."""
    with pytest.raises(TypeError):

        class PartialService(IDatabaseService):
            async def search_chunks(
                self, query_embedding: List[float], top_k: int = 10
            ) -> List[Dict[str, Any]]:
                return []

            async def get_chunk(self, chunk_id: str) -> Dict[str, Any]:
                return {}

        PartialService()


@pytest.mark.asyncio
async def test_idatabase_service_mock_implementation():
    """Test that complete IDatabaseService implementation works."""

    class MockDatabaseService(IDatabaseService):
        def __init__(self):
            self.chunks = {}

        async def store_chunks(self, chunks: List[Dict[str, Any]]) -> None:
            for chunk in chunks:
                chunk_id = chunk.get("id", f"chunk_{len(self.chunks)}")
                self.chunks[chunk_id] = chunk

        async def search_chunks(
            self, query_embedding: List[float], top_k: int = 10
        ) -> List[Dict[str, Any]]:
            # Simple mock: return first top_k chunks
            return list(self.chunks.values())[:top_k]

        async def get_chunk(self, chunk_id: str) -> Dict[str, Any]:
            return self.chunks.get(chunk_id, {})
        
        async def get_chunks_for_documents(self, document_ids: List[str]) -> List[Dict[str, Any]]:
            # Simple mock: return chunks that match document_ids
            return [chunk for chunk in self.chunks.values() 
                    if chunk.get("document_id") in document_ids]
        
        async def store_chunks_with_hierarchy(self, chunks: List[Dict[str, Any]]) -> None:
            # Simple mock: store chunks same as store_chunks
            await self.store_chunks(chunks)

    service = MockDatabaseService()
    assert service is not None

    # Test store_chunks method
    chunks = [
        {"id": "1", "text": "Hello", "embedding": [0.1, 0.2]},
        {"id": "2", "text": "World", "embedding": [0.3, 0.4]},
    ]
    await service.store_chunks(chunks)
    assert len(service.chunks) == 2

    # Test search_chunks method
    results = await service.search_chunks([0.1, 0.2], top_k=1)
    assert len(results) == 1

    # Test get_chunk method
    chunk = await service.get_chunk("1")
    assert chunk["text"] == "Hello"


def test_idatabase_service_has_docstring():
    """Test that IDatabaseService has docstring."""
    assert IDatabaseService.__doc__ is not None
    assert len(IDatabaseService.__doc__) > 0


# ============================================================================
# Type Hint Validation Tests
# ============================================================================


def test_all_interfaces_have_type_hints():
    """Test that all interface methods have type hints."""
    from inspect import signature, get_annotations

    interfaces = [
        ILLMService,
        IEmbeddingService,
        IGraphService,
        IRerankingService,
        IDatabaseService,
    ]

    for interface in interfaces:
        for method_name in dir(interface):
            if method_name.startswith("_"):
                continue

            method = getattr(interface, method_name)
            if callable(method):
                # Check that method has annotations
                annotations = get_annotations(method)
                # Abstract methods should have return type annotation
                assert (
                    "return" in annotations or method_name in ["__init__"]
                ), f"{interface.__name__}.{method_name} missing return type hint"
