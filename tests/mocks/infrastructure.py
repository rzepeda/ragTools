"""Infrastructure mock builders for testing.

This module provides builders for creating mock infrastructure objects
including service registries, ONNX environments, and configuration objects.
"""

from typing import Optional, Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from contextlib import contextmanager
from pathlib import Path

from .services import (
    create_mock_embedding_service,
    create_mock_database_service,
    create_mock_llm_service,
    create_mock_neo4j_service,
    create_mock_reranker_service
)
from .database import create_mock_migration_validator


def create_mock_registry(
    services: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Mock:
    """Create a mock service registry.
    
    Args:
        services: Dictionary of service name -> service instance
        config: Optional configuration dictionary
        
    Returns:
        Mock service registry
        
    Example:
        >>> registry = create_mock_registry({
        ...     "embedding_local": create_mock_embedding_service(),
        ...     "db_main": create_mock_database_service()
        ... })
        >>> service = registry.get("embedding_local")
    """
    registry = Mock()
    
    if services is None:
        services = {}
    
    if config is None:
        config = {'services': {}}
    
    # Store services in _instances
    registry._instances = services.copy()
    registry.config = config
    
    # Mock get method to return from _instances
    def mock_get(name: str) -> Any:
        return registry._instances.get(name)
    
    registry.get = Mock(side_effect=mock_get)
    registry.register = Mock()
    registry.unregister = Mock()
    registry.list_services = Mock(return_value=list(services.keys()))
    
    return registry


def create_mock_registry_with_services(
    include_embedding: bool = True,
    include_database: bool = True,
    include_llm: bool = False,
    include_neo4j: bool = False,
    include_reranker: bool = False,
    embedding_dimension: int = 384,
    **kwargs
) -> Mock:
    """Create a mock registry pre-populated with common services.
    
    Args:
        include_embedding: Include embedding service
        include_database: Include database service
        include_llm: Include LLM service
        include_neo4j: Include Neo4j service
        include_reranker: Include reranker service
        embedding_dimension: Dimension for embedding service
        **kwargs: Additional service configurations
        
    Returns:
        Mock service registry with services
        
    Example:
        >>> registry = create_mock_registry_with_services(
        ...     include_llm=True,
        ...     embedding_dimension=768
        ... )
    """
    services = {}
    
    if include_embedding:
        embedding_service = create_mock_embedding_service(dimension=embedding_dimension)
        # Add all common embedding service aliases used in YAML configs
        services["embedding_local"] = embedding_service
        services["local-onnx-minilm"] = embedding_service
        services["embedding_finetuned"] = embedding_service  # For fine-tuned-embeddings-pair
        services["embedding_api"] = embedding_service  # For semantic-api-pair
        services["embedding_openai"] = embedding_service  # For hybrid-search-pair
        services["embedding_cohere"] = embedding_service  # Alternative alias
    
    if include_database:
        db_service = create_mock_database_service()
        services["db_main"] = db_service
        services["main-postgres"] = db_service
        services["db1"] = db_service
    
    if include_llm:
        llm_service = create_mock_llm_service()
        services["llm_local"] = llm_service
        services["local-llama"] = llm_service
    
    if include_neo4j:
        neo4j_service = create_mock_neo4j_service()
        services["db_neo4j"] = neo4j_service
        services["neo4j-graph"] = neo4j_service
    
    if include_reranker:
        reranker_service = create_mock_reranker_service()
        services["reranker_local"] = reranker_service
        services["local-reranker"] = reranker_service
    
    return create_mock_registry(services=services)


def create_mock_strategy_pair_manager(
    config_dir: Optional[str] = None
) -> Mock:
    """Create a mock strategy pair manager.
    
    Args:
        config_dir: Configuration directory path
        
    Returns:
        Mock strategy pair manager
        
    Example:
        >>> manager = create_mock_strategy_pair_manager()
        >>> indexing, retrieval = manager.load_pair("semantic-local-pair")
    """
    manager = Mock()
    
    if config_dir is None:
        config_dir = "/mock/config/dir"
    
    # Mock methods
    manager.load_pair = Mock()
    manager.list_pairs = Mock(return_value=[
        "semantic-local-pair",
        "knowledge-graph-pair"
    ])
    manager.validate_pair = Mock(return_value=True)
    
    # Properties
    manager.config_dir = config_dir
    manager.migration_validator = create_mock_migration_validator()
    
    return manager


@contextmanager
def create_mock_onnx_environment(
    dimension: int = 384,
    model_path: Optional[str] = None,
    model_name: str = "Xenova/all-MiniLM-L6-v2"
):
    """Context manager for mocking ONNX environment.
    
    This patches all ONNX-related imports and utilities to allow
    testing ONNX providers without actual ONNX installation.
    
    Args:
        dimension: Embedding dimension
        model_path: Path to model file
        model_name: Model name
        
    Yields:
        None (patches are active within context)
        
    Example:
        >>> with create_mock_onnx_environment(dimension=768):
        ...     provider = ONNXLocalProvider(config)
        ...     embeddings = provider.get_embeddings(["text"])
    """
    import numpy as np
    
    if model_path is None:
        model_path = "/fake/path/to/onnx/model.onnx"
    
    # Mock ONNX session
    mock_session_obj = Mock()
    mock_output = Mock()
    mock_output.shape = [1, 512, dimension]
    mock_session_obj.get_outputs = Mock(return_value=[mock_output])
    
    mock_input = Mock()
    mock_input.name = "input_ids"
    mock_session_obj.get_inputs = Mock(return_value=[mock_input])
    
    # Mock ONNX inference
    mock_embeddings = np.random.randn(2, 512, dimension).astype(np.float32)
    mock_session_obj.run = Mock(return_value=[mock_embeddings])
    
    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_encoding = Mock()
    mock_encoding.ids = [1, 2, 3, 4, 5]
    mock_tokenizer.encode = Mock(return_value=mock_encoding)
    
    with patch("rag_factory.services.embedding.providers.onnx_local.create_onnx_session", return_value=mock_session_obj), \
         patch("rag_factory.services.embedding.providers.onnx_local.get_onnx_model_path", return_value=Path(model_path)), \
         patch("rag_factory.services.embedding.providers.onnx_local.validate_onnx_model"), \
         patch("rag_factory.services.embedding.providers.onnx_local.get_model_metadata", return_value={"embedding_dim": dimension}), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("tokenizers.Tokenizer.from_file", return_value=mock_tokenizer):
        yield


def create_mock_config(
    **kwargs
) -> Mock:
    """Create a mock configuration object.
    
    Args:
        **kwargs: Configuration key-value pairs
        
    Returns:
        Mock configuration
        
    Example:
        >>> config = create_mock_config(
        ...     model="gpt-3.5-turbo",
        ...     temperature=0.7
        ... )
        >>> assert config.model == "gpt-3.5-turbo"
    """
    config = Mock()
    
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    config.get = Mock(side_effect=lambda k, default=None: kwargs.get(k, default))
    config.to_dict = Mock(return_value=kwargs)
    
    return config


def create_mock_indexing_context(
    database_service: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None
) -> Mock:
    """Create a mock IndexingContext.
    
    Args:
        database_service: Optional database service
        config: Optional configuration dictionary
        
    Returns:
        Mock IndexingContext
        
    Example:
        >>> context = create_mock_indexing_context()
        >>> await strategy.process(docs, context)
    """
    context = Mock()
    
    if database_service is None:
        database_service = create_mock_database_service()
    
    if config is None:
        config = {}
    
    context.database_service = database_service
    context.config = config
    context.get_config = Mock(side_effect=lambda k, default=None: config.get(k, default))
    
    return context


def create_mock_retrieval_context(
    database_service: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None
) -> Mock:
    """Create a mock RetrievalContext.
    
    Args:
        database_service: Optional database service
        config: Optional configuration dictionary
        
    Returns:
        Mock RetrievalContext
        
    Example:
        >>> context = create_mock_retrieval_context()
        >>> chunks = await strategy.retrieve("query", context)
    """
    context = Mock()
    
    if database_service is None:
        database_service = create_mock_database_service()
    
    if config is None:
        config = {}
    
    context.database_service = database_service
    context.config = config
    context.get_config = Mock(side_effect=lambda k, default=None: config.get(k, default))
    
    return context


def create_mock_chunk(
    id: str = "chunk1",
    text: str = "mock content",
    score: float = 0.9,
    metadata: Optional[Dict[str, Any]] = None,
    embedding: Optional[List[float]] = None
) -> Mock:
    """Create a mock Chunk object.
    
    Args:
        id: Chunk ID
        text: Chunk text content
        score: Relevance score
        metadata: Optional metadata dictionary
        embedding: Optional embedding vector
        
    Returns:
        Mock Chunk
        
    Example:
        >>> chunk = create_mock_chunk(text="custom content", score=0.95)
        >>> assert chunk.text == "custom content"
    """
    chunk = Mock()
    
    if metadata is None:
        metadata = {}
    
    if embedding is None:
        embedding = [0.1] * 384
    
    chunk.id = id
    chunk.text = text
    chunk.score = score
    chunk.metadata = metadata
    chunk.embedding = embedding
    
    chunk.to_dict = Mock(return_value={
        "id": id,
        "text": text,
        "score": score,
        "metadata": metadata
    })
    
    return chunk


def create_mock_document(
    id: str = "doc1",
    filename: str = "test.txt",
    content: str = "document content",
    metadata: Optional[Dict[str, Any]] = None
) -> Mock:
    """Create a mock Document object.
    
    Args:
        id: Document ID
        filename: Document filename
        content: Document content
        metadata: Optional metadata dictionary
        
    Returns:
        Mock Document
        
    Example:
        >>> doc = create_mock_document(filename="report.pdf")
        >>> assert doc.filename == "report.pdf"
    """
    document = Mock()
    
    if metadata is None:
        metadata = {}
    
    document.id = id
    document.filename = filename
    document.content = content
    document.metadata = metadata
    
    document.to_dict = Mock(return_value={
        "id": id,
        "filename": filename,
        "content": content,
        "metadata": metadata
    })
    
    return document
