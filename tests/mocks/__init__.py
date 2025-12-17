"""Centralized mock system for RAG Factory tests.

This package provides reusable mock builders for all test types,
reducing code duplication and ensuring consistency across the test suite.

Quick Start:
    >>> from tests.mocks import create_mock_embedding_service
    >>> service = create_mock_embedding_service(dimension=768)
    
    >>> from tests.mocks import create_mock_registry_with_services
    >>> registry = create_mock_registry_with_services(include_llm=True)

See README.md for comprehensive documentation and examples.
"""

# Service mocks
from .services import (
    create_mock_embedding_service,
    create_mock_database_service,
    create_mock_llm_service,
    create_mock_neo4j_service,
    create_mock_reranker_service,
    create_mock_cohere_embedding_service,
    create_mock_openai_embedding_service,
    create_mock_openai_llm_service,
)

# Database mocks
from .database import (
    create_mock_engine,
    create_mock_connection,
    create_mock_session,
    create_mock_migration_validator,
    create_mock_async_pool,
    create_mock_database_context,
    mock_database_transaction,
    create_mock_alembic_config,
    create_mock_postgres_service,
    create_mock_neo4j_driver,
)

# Strategy mocks
from .strategies import (
    create_mock_indexing_strategy,
    create_mock_retrieval_strategy,
    create_mock_rag_strategy,
    create_mock_chunking_strategy,
    create_mock_reranking_strategy,
    create_mock_hybrid_retrieval_strategy,
    create_mock_knowledge_graph_strategy,
    create_mock_multi_query_strategy,
    create_mock_contextual_retrieval_strategy,
)

# Infrastructure mocks
from .infrastructure import (
    create_mock_registry,
    create_mock_registry_with_services,
    create_mock_strategy_pair_manager,
    create_mock_onnx_environment,
    create_mock_config,
    create_mock_indexing_context,
    create_mock_retrieval_context,
    create_mock_chunk,
    create_mock_document,
)

# Builder utilities
from .builders import (
    MockBuilder,
    ServiceMockBuilder,
    create_mock_with_context_manager,
    create_async_context_manager_mock,
    configure_mock_call_tracking,
)

__all__ = [
    # Service mocks
    "create_mock_embedding_service",
    "create_mock_database_service",
    "create_mock_llm_service",
    "create_mock_neo4j_service",
    "create_mock_reranker_service",
    "create_mock_cohere_embedding_service",
    "create_mock_openai_embedding_service",
    "create_mock_openai_llm_service",
    
    # Database mocks
    "create_mock_engine",
    "create_mock_connection",
    "create_mock_session",
    "create_mock_migration_validator",
    "create_mock_async_pool",
    "create_mock_database_context",
    "mock_database_transaction",
    "create_mock_alembic_config",
    "create_mock_postgres_service",
    "create_mock_neo4j_driver",
    
    # Strategy mocks
    "create_mock_indexing_strategy",
    "create_mock_retrieval_strategy",
    "create_mock_rag_strategy",
    "create_mock_chunking_strategy",
    "create_mock_reranking_strategy",
    "create_mock_hybrid_retrieval_strategy",
    "create_mock_knowledge_graph_strategy",
    "create_mock_multi_query_strategy",
    "create_mock_contextual_retrieval_strategy",
    
    # Infrastructure mocks
    "create_mock_registry",
    "create_mock_registry_with_services",
    "create_mock_strategy_pair_manager",
    "create_mock_onnx_environment",
    "create_mock_config",
    "create_mock_indexing_context",
    "create_mock_retrieval_context",
    "create_mock_chunk",
    "create_mock_document",
    
    # Builder utilities
    "MockBuilder",
    "ServiceMockBuilder",
    "create_mock_with_context_manager",
    "create_async_context_manager_mock",
    "configure_mock_call_tracking",
]

# Version
__version__ = "1.0.0"
