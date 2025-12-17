"""Strategy mock builders for testing.

This module provides builders for creating mock strategy objects
including indexing, retrieval, and reranking strategies.
"""

from typing import Optional, List, Dict, Any
from unittest.mock import Mock, AsyncMock

from .builders import ServiceMockBuilder


def create_mock_indexing_strategy(
    capabilities: Optional[List[str]] = None,
    process_return_value: Optional[Any] = None
) -> Mock:
    """Create a mock indexing strategy.
    
    Args:
        capabilities: List of capability names (e.g., ["VECTORS", "KEYWORDS"])
        process_return_value: Custom return value for process() method
        
    Returns:
        Mock indexing strategy
        
    Example:
        >>> strategy = create_mock_indexing_strategy(capabilities=["VECTORS"])
        >>> result = await strategy.process(docs, context)
        >>> assert "VECTORS" in result.capabilities
    """
    strategy = Mock()
    
    if capabilities is None:
        capabilities = ["VECTORS"]
    
    # Create mock IndexingResult
    if process_return_value is None:
        result = Mock()
        result.capabilities = capabilities
        result.indexed_count = 10
        result.failed_count = 0
        result.metadata = {}
        process_return_value = result
    
    # Async methods
    strategy.process = AsyncMock(return_value=process_return_value)
    strategy.close = AsyncMock()
    
    # Sync methods
    strategy.get_capabilities = Mock(return_value=capabilities)
    strategy.validate_config = Mock(return_value=True)
    
    # Properties
    strategy.capabilities = capabilities
    strategy.name = "MockIndexingStrategy"
    
    # Dependencies
    strategy.deps = Mock()
    strategy.deps.embedding_service = None
    strategy.deps.database_service = None
    strategy.deps.llm_service = None
    
    return strategy


def create_mock_retrieval_strategy(
    retrieve_return_value: Optional[List[Any]] = None,
    capabilities: Optional[List[str]] = None
) -> Mock:
    """Create a mock retrieval strategy.
    
    Args:
        retrieve_return_value: Custom return value for retrieve() method
        capabilities: List of required capabilities
        
    Returns:
        Mock retrieval strategy
        
    Example:
        >>> strategy = create_mock_retrieval_strategy()
        >>> chunks = await strategy.retrieve("query", context)
        >>> assert len(chunks) > 0
    """
    strategy = Mock()
    
    if capabilities is None:
        capabilities = ["VECTORS"]
    
    # Create mock chunks
    if retrieve_return_value is None:
        chunk = Mock()
        chunk.id = "chunk1"
        chunk.text = "mock content"
        chunk.score = 0.9
        chunk.metadata = {}
        retrieve_return_value = [chunk]
    
    # Async methods
    strategy.retrieve = AsyncMock(return_value=retrieve_return_value)
    strategy.close = AsyncMock()
    
    # Sync methods
    strategy.get_required_capabilities = Mock(return_value=capabilities)
    strategy.validate_config = Mock(return_value=True)
    
    # Properties
    strategy.required_capabilities = capabilities
    strategy.name = "MockRetrievalStrategy"
    
    # Dependencies
    strategy.deps = Mock()
    strategy.deps.embedding_service = None
    strategy.deps.database_service = None
    strategy.deps.llm_service = None
    
    return strategy


def create_mock_rag_strategy(
    index_return_value: Optional[Any] = None,
    retrieve_return_value: Optional[List[Any]] = None
) -> Mock:
    """Create a mock RAG strategy (combined indexing + retrieval).
    
    Args:
        index_return_value: Custom return value for index() method
        retrieve_return_value: Custom return value for retrieve() method
        
    Returns:
        Mock RAG strategy
        
    Example:
        >>> strategy = create_mock_rag_strategy()
        >>> await strategy.index(docs)
        >>> chunks = await strategy.retrieve("query")
    """
    strategy = Mock()
    
    # Create mock results
    if index_return_value is None:
        index_result = Mock()
        index_result.indexed_count = 10
        index_result.failed_count = 0
        index_return_value = index_result
    
    if retrieve_return_value is None:
        chunk = Mock()
        chunk.text = "mock content"
        chunk.score = 0.9
        retrieve_return_value = [chunk]
    
    # Async methods
    strategy.index = AsyncMock(return_value=index_return_value)
    strategy.retrieve = AsyncMock(return_value=retrieve_return_value)
    strategy.close = AsyncMock()
    
    # Sync methods
    strategy.get_stats = Mock(return_value={
        "total_indexed": 10,
        "total_retrieved": 5
    })
    
    # Properties
    strategy.name = "MockRAGStrategy"
    
    return strategy


def create_mock_chunking_strategy(
    chunk_return_value: Optional[List[str]] = None
) -> Mock:
    """Create a mock chunking strategy.
    
    Args:
        chunk_return_value: Custom return value for chunk() method
        
    Returns:
        Mock chunking strategy
        
    Example:
        >>> strategy = create_mock_chunking_strategy()
        >>> chunks = strategy.chunk("long text")
        >>> assert len(chunks) > 0
    """
    strategy = Mock()
    
    if chunk_return_value is None:
        chunk_return_value = ["chunk1", "chunk2", "chunk3"]
    
    # Sync methods
    strategy.chunk = Mock(return_value=chunk_return_value)
    strategy.get_chunk_size = Mock(return_value=512)
    strategy.get_overlap = Mock(return_value=50)
    
    # Properties
    strategy.chunk_size = 512
    strategy.overlap = 50
    strategy.name = "MockChunkingStrategy"
    
    return strategy


def create_mock_reranking_strategy(
    rerank_return_value: Optional[List[Any]] = None,
    top_k: int = 5
) -> Mock:
    """Create a mock reranking strategy.
    
    Args:
        rerank_return_value: Custom return value for rerank() method
        top_k: Number of results to return
        
    Returns:
        Mock reranking strategy
        
    Example:
        >>> strategy = create_mock_reranking_strategy(top_k=3)
        >>> results = await strategy.rerank("query", chunks)
        >>> assert len(results) <= 3
    """
    strategy = Mock()
    
    if rerank_return_value is None:
        chunk = Mock()
        chunk.text = "reranked content"
        chunk.score = 0.95
        rerank_return_value = [chunk] * min(top_k, 3)
    
    # Async methods
    strategy.rerank = AsyncMock(return_value=rerank_return_value)
    strategy.close = AsyncMock()
    
    # Sync methods
    strategy.get_top_k = Mock(return_value=top_k)
    
    # Properties
    strategy.top_k = top_k
    strategy.name = "MockRerankingStrategy"
    
    return strategy


def create_mock_hybrid_retrieval_strategy(
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> Mock:
    """Create a mock hybrid retrieval strategy.
    
    Args:
        vector_weight: Weight for vector search results
        keyword_weight: Weight for keyword search results
        
    Returns:
        Mock hybrid retrieval strategy
        
    Example:
        >>> strategy = create_mock_hybrid_retrieval_strategy()
        >>> chunks = await strategy.retrieve("query", context)
    """
    strategy = create_mock_retrieval_strategy(
        capabilities=["VECTORS", "KEYWORDS"]
    )
    
    # Add hybrid-specific properties
    strategy.vector_weight = vector_weight
    strategy.keyword_weight = keyword_weight
    strategy.get_weights = Mock(return_value=(vector_weight, keyword_weight))
    
    return strategy


def create_mock_knowledge_graph_strategy() -> Mock:
    """Create a mock knowledge graph strategy.
    
    Returns:
        Mock knowledge graph strategy
        
    Example:
        >>> strategy = create_mock_knowledge_graph_strategy()
        >>> await strategy.extract_entities(text)
    """
    strategy = create_mock_rag_strategy()
    
    # Add knowledge graph-specific methods
    strategy.extract_entities = AsyncMock(return_value=[
        {"type": "Person", "name": "John"},
        {"type": "Company", "name": "Google"}
    ])
    strategy.extract_relationships = AsyncMock(return_value=[
        {"source": "John", "relation": "works_at", "target": "Google"}
    ])
    strategy.query_graph = AsyncMock(return_value=[])
    
    # Update dependencies
    strategy.deps.graph_service = None
    
    return strategy


def create_mock_multi_query_strategy(
    num_queries: int = 3
) -> Mock:
    """Create a mock multi-query expansion strategy.
    
    Args:
        num_queries: Number of expanded queries to generate
        
    Returns:
        Mock multi-query strategy
        
    Example:
        >>> strategy = create_mock_multi_query_strategy(num_queries=5)
        >>> queries = await strategy.expand_query("original query")
        >>> assert len(queries) == 5
    """
    strategy = create_mock_retrieval_strategy()
    
    # Add multi-query specific methods
    strategy.expand_query = AsyncMock(return_value=[
        f"expanded query {i}" for i in range(num_queries)
    ])
    strategy.get_num_queries = Mock(return_value=num_queries)
    
    # Properties
    strategy.num_queries = num_queries
    
    return strategy


def create_mock_contextual_retrieval_strategy() -> Mock:
    """Create a mock contextual retrieval strategy.
    
    Returns:
        Mock contextual retrieval strategy
        
    Example:
        >>> strategy = create_mock_contextual_retrieval_strategy()
        >>> chunks = await strategy.retrieve_with_context("query", context)
    """
    strategy = create_mock_retrieval_strategy()
    
    # Add contextual-specific methods
    strategy.add_context = AsyncMock(return_value="contextualized chunk")
    strategy.retrieve_with_context = AsyncMock(return_value=[
        Mock(text="chunk with context", score=0.9)
    ])
    
    return strategy
