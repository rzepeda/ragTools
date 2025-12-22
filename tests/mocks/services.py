"""Service mock builders for testing.

This module provides builders for creating mock service objects
with consistent behavior across all tests.
"""

from typing import List, Dict, Any, Optional
from unittest.mock import Mock, AsyncMock

from .builders import ServiceMockBuilder


def create_mock_embedding_service(
    dimension: int = 384,
    model_name: str = "mock-embedding-model",
    embed_return_value: Optional[List[float]] = None,
    embed_batch_return_value: Optional[List[List[float]]] = None,
    max_batch_size: int = 32,
    cost: float = 0.0
) -> Mock:
    """Create a mock embedding service.
    
    Args:
        dimension: Embedding dimension (default: 384)
        model_name: Model name to return
        embed_return_value: Custom return value for embed() method
        embed_batch_return_value: Custom return value for embed_batch() method
        max_batch_size: Maximum batch size
        cost: Cost per embedding (default: 0.0 for local models)
        
    Returns:
        Mock embedding service
        
    Example:
        >>> service = create_mock_embedding_service(dimension=768)
        >>> result = await service.embed("test text")
        >>> assert len(result) == 768
    """
    service = Mock()
    
    # Default return values
    if embed_return_value is None:
        embed_return_value = [0.1] * dimension
    if embed_batch_return_value is None:
        embed_batch_return_value = [[0.1] * dimension]
    
    # Async methods
    service.embed = AsyncMock(return_value=embed_return_value)
    service.embed_batch = AsyncMock(return_value=embed_batch_return_value)
    service.close = AsyncMock()
    
    # Sync methods
    service.get_dimension = Mock(return_value=dimension)
    service.get_dimensions = Mock(return_value=dimension)
    service.get_model_name = Mock(return_value=model_name)
    service.get_max_batch_size = Mock(return_value=max_batch_size)
    service.calculate_cost = Mock(return_value=cost)
    service.estimate_cost = Mock(return_value=cost)
    
    # Properties
    service.model_name = model_name
    service.dimension = dimension
    service.dimensions = dimension
    
    return service


def create_mock_database_service(
    get_chunks_return_value: Optional[List[Dict[str, Any]]] = None,
    search_chunks_return_value: Optional[List[Dict[str, Any]]] = None,
    store_chunks_return_value: Optional[Any] = None
) -> Mock:
    """Create a mock database service.
    
    Args:
        get_chunks_return_value: Return value for get_chunks_for_documents()
        search_chunks_return_value: Return value for search_chunks()
        store_chunks_return_value: Return value for store_chunks()
        
    Returns:
        Mock database service
        
    Example:
        >>> service = create_mock_database_service()
        >>> chunks = await service.search_chunks([0.1] * 384, top_k=5)
        >>> assert len(chunks) == 1
    """
    service = Mock()
    
    # Default return values
    if get_chunks_return_value is None:
        get_chunks_return_value = [
            {'id': 'chunk1', 'text': 'mock content', 'metadata': {}}
        ]
    if search_chunks_return_value is None:
        search_chunks_return_value = [
            {'id': 'chunk1', 'text': 'mock content', 'score': 0.9, 'metadata': {}}
        ]
    
    # Async methods
    service.get_chunks_for_documents = AsyncMock(return_value=get_chunks_return_value)
    service.search_chunks = AsyncMock(return_value=search_chunks_return_value)
    service.search_chunks_by_keywords = AsyncMock(return_value=search_chunks_return_value)  # For hybrid search
    service.store_chunks = AsyncMock(return_value=store_chunks_return_value)
    service.store_chunks_with_hierarchy = AsyncMock(return_value=store_chunks_return_value)  # For hierarchical indexing
    service.store_keyword_index = AsyncMock(return_value=None)
    service.search_keyword = AsyncMock(return_value=search_chunks_return_value)
    service.close = AsyncMock(return_value=None)
    service.asearch = AsyncMock(return_value=search_chunks_return_value)
    
    # Sync methods - get_context returns self for chaining
    service.get_context = Mock(return_value=service)
    
    # Mock engine and connection for MigrationValidator
    mock_engine = Mock()
    mock_connection = Mock()
    mock_connection.__enter__ = Mock(return_value=mock_connection)
    mock_connection.__exit__ = Mock(return_value=None)
    mock_engine.connect = Mock(return_value=mock_connection)
    service.get_engine = Mock(return_value=mock_engine)
    
    return service


def create_mock_llm_service(
    generate_return_value: str = "Mock LLM response",
    agenerate_return_value: Optional[str] = None,
    model_name: str = "mock-llm-model",
    cost: float = 0.0
) -> Mock:
    """Create a mock LLM service.
    
    Args:
        generate_return_value: Return value for generate() method
        agenerate_return_value: Return value for agenerate() (defaults to generate_return_value)
        model_name: Model name to return
        cost: Cost per generation (default: 0.0 for local models)
        
    Returns:
        Mock LLM service
        
    Example:
        >>> service = create_mock_llm_service(generate_return_value="Custom response")
        >>> result = await service.agenerate("prompt")
        >>> assert result == "Custom response"
    """
    service = Mock()
    
    if agenerate_return_value is None:
        agenerate_return_value = generate_return_value
    
    # Create mock response object
    response = Mock()
    response.text = generate_return_value
    response.content = generate_return_value
    response.prompt_tokens = 10
    response.completion_tokens = 20
    response.total_tokens = 30
    response.cost = cost
    response.latency = 0.1
    
    # Async methods
    service.generate = AsyncMock(return_value=generate_return_value)
    service.agenerate = AsyncMock(return_value=agenerate_return_value)
    service.close = AsyncMock()
    
    # Sync methods
    service.complete = Mock(return_value=response)
    service.count_tokens = Mock(return_value=10)
    service.get_stats = Mock(return_value={
        "total_requests": 1,
        "total_prompt_tokens": 10,
        "total_completion_tokens": 20,
        "total_cost": cost,
        "total_latency": 0.1,
        "average_latency": 0.1,
        "model": model_name,
        "provider": "mock"
    })
    service.estimate_cost = Mock(return_value=cost)
    service.get_model_name = Mock(return_value=model_name)
    
    # Properties
    service.model_name = model_name
    
    return service


def create_mock_neo4j_service(
    execute_query_return_value: Optional[List[Dict[str, Any]]] = None
) -> Mock:
    """Create a mock Neo4j graph database service.
    
    Args:
        execute_query_return_value: Return value for execute_query() method
        
    Returns:
        Mock Neo4j service
        
    Example:
        >>> service = create_mock_neo4j_service()
        >>> result = await service.execute_query("MATCH (n) RETURN n")
        >>> assert result == []
    """
    service = Mock()
    
    if execute_query_return_value is None:
        execute_query_return_value = []
    
    # Async methods
    service.execute_query = AsyncMock(return_value=execute_query_return_value)
    service.close = AsyncMock()
    service.create_node = AsyncMock()
    service.create_relationship = AsyncMock()
    service.query_nodes = AsyncMock(return_value=[])
    
    # Sync methods
    service.get_stats = Mock(return_value={
        "total_queries": 0,
        "total_nodes": 0,
        "total_relationships": 0
    })
    
    return service


def create_mock_reranker_service(
    rerank_return_value: Optional[List[Dict[str, Any]]] = None,
    model_name: str = "mock-reranker"
) -> Mock:
    """Create a mock reranking service.
    
    Args:
        rerank_return_value: Return value for rerank() method
        model_name: Model name to return
        
    Returns:
        Mock reranker service
        
    Example:
        >>> service = create_mock_reranker_service()
        >>> results = await service.rerank("query", chunks)
        >>> assert len(results) > 0
    """
    service = Mock()
    
    if rerank_return_value is None:
        rerank_return_value = [
            {'id': 'chunk1', 'text': 'content', 'score': 0.95, 'metadata': {}}
        ]
    
    # Async methods
    service.rerank = AsyncMock(return_value=rerank_return_value)
    service.close = AsyncMock()
    
    # Sync methods
    service.get_model_name = Mock(return_value=model_name)
    
    # Properties
    service.model_name = model_name
    
    return service


def create_mock_cohere_embedding_service(
    dimension: int = 1024,
    model_name: str = "embed-english-v3.0"
) -> Mock:
    """Create a mock Cohere embedding service.
    
    Args:
        dimension: Embedding dimension (default: 1024 for Cohere)
        model_name: Model name
        
    Returns:
        Mock Cohere embedding service
    """
    return create_mock_embedding_service(
        dimension=dimension,
        model_name=model_name,
        cost=0.0001  # Cohere has cost
    )


def create_mock_openai_embedding_service(
    dimension: int = 1536,
    model_name: str = "text-embedding-ada-002"
) -> Mock:
    """Create a mock OpenAI embedding service.
    
    Args:
        dimension: Embedding dimension (default: 1536 for Ada-002)
        model_name: Model name
        
    Returns:
        Mock OpenAI embedding service
    """
    return create_mock_embedding_service(
        dimension=dimension,
        model_name=model_name,
        cost=0.0001  # OpenAI has cost
    )


def create_mock_openai_llm_service(
    model_name: str = "gpt-3.5-turbo"
) -> Mock:
    """Create a mock OpenAI LLM service.
    
    Args:
        model_name: Model name
        
    Returns:
        Mock OpenAI LLM service
    """
    return create_mock_llm_service(
        model_name=model_name,
        cost=0.002  # OpenAI has cost
    )
