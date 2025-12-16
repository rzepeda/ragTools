"""Factory for creating service instances from configurations."""

from typing import Any, Dict
import logging

# Import existing service interfaces from Epic 11
from rag_factory.services.interfaces import (
    ILLMService,
    IEmbeddingService,
    IDatabaseService
)

# NOTE: Service implementations are imported lazily in the create methods
# to avoid circular import issues. Do NOT import them at module level.

from .exceptions import ServiceInstantiationError

logger = logging.getLogger(__name__)


class ServiceFactory:
    """Factory for creating service instances from configurations.
    
    Uses existing service implementations from Epic 11.
    """

    def create_service(
        self,
        service_name: str,
        config: Dict[str, Any]
    ) -> Any:
        """Factory method to create service instances based on configuration.
        
        Args:
            service_name: Name of service (for logging)
            config: Service configuration dictionary
            
        Returns:
            Service instance implementing appropriate interface
            
        Raises:
            ServiceInstantiationError: If service type cannot be determined
        """
        # Determine service type based on configuration keys
        if self._is_llm_service(config):
            return self._create_llm_service(service_name, config)
        elif self._is_embedding_service(config):
            return self._create_embedding_service(service_name, config)
        elif self._is_database_service(config):
            return self._create_database_service(service_name, config)
        else:
            raise ServiceInstantiationError(
                f"Cannot determine service type for '{service_name}'. "
                f"Configuration: {config}"
            )

    def _is_llm_service(self, config: Dict[str, Any]) -> bool:
        """Check if configuration represents an LLM service."""
        # Check explicit type field first
        if 'type' in config:
            return config['type'] == 'llm'
        # Fall back to heuristic detection
        return 'url' in config and 'model' in config

    def _is_embedding_service(self, config: Dict[str, Any]) -> bool:
        """Check if configuration represents an embedding service."""
        # Check explicit type field first
        if 'type' in config:
            return config['type'] == 'embedding'
        # Fall back to heuristic detection
        return 'provider' in config

    def _is_database_service(self, config: Dict[str, Any]) -> bool:
        """Check if configuration represents a database service."""
        # Check explicit type field
        if 'type' in config:
            return config['type'] in ['database', 'postgres', 'neo4j', 'mongodb']
        return False

    def _create_llm_service(
        self,
        service_name: str,
        config: Dict[str, Any]
    ) -> ILLMService:
        """Create LLM service from configuration."""
        # Lazy import to avoid circular dependency
        from rag_factory.services.api import OpenAILLMService
        
        logger.debug(f"Creating LLM service: {service_name}")

        url = config['url']

        # Detect provider based on URL
        if 'openai.com' in url or 'api.openai.com' in url:
            # OpenAI LLM service
            return OpenAILLMService(
                api_key=config.get('api_key'),
                model=config['model']
            )
        else:
            # LM Studio or other OpenAI-compatible service
            # Note: OpenAILLMService doesn't support base_url parameter
            # For now, we'll raise an error. In a real implementation,
            # we'd need a separate LMStudioLLMService or use a different approach
            raise ServiceInstantiationError(
                f"LM Studio and OpenAI-compatible services not yet fully supported. "
                f"Only OpenAI API (api.openai.com) is currently supported for LLM services. "
                f"Service '{service_name}' uses URL: {url}"
            )

    def _create_embedding_service(
        self,
        service_name: str,
        config: Dict[str, Any]
    ) -> IEmbeddingService:
        """Create embedding service from configuration."""
        # Lazy imports to avoid circular dependency
        from rag_factory.services.onnx import ONNXEmbeddingService
        from rag_factory.services.api import OpenAIEmbeddingService
        
        logger.debug(f"Creating embedding service: {service_name}")

        provider = config['provider']

        if provider == 'onnx':
            # ONNX local embedding service
            return ONNXEmbeddingService(
                model=config['model'],
                cache_dir=config.get('cache_dir', './models'),
                max_batch_size=config.get('batch_size', 32)
            )
        elif provider == 'openai':
            # OpenAI embedding service
            return OpenAIEmbeddingService(
                api_key=config['api_key'],
                model=config['model']
            )
        elif provider == 'cohere':
            # Cohere embedding service
            # Note: CohereRerankingService is for reranking, not embeddings
            # We'll need to check if there's a Cohere embedding service
            raise ServiceInstantiationError(
                f"Cohere embedding service not yet implemented. "
                f"Use 'onnx' or 'openai' provider instead."
            )
        else:
            raise ServiceInstantiationError(
                f"Unknown embedding provider: {provider}"
            )

    def _create_database_service(
        self,
        service_name: str,
        config: Dict[str, Any]
    ) -> IDatabaseService:
        """Create database service from configuration."""
        # Lazy imports to avoid circular dependency
        from rag_factory.services.database import PostgresqlDatabaseService, Neo4jGraphService
        
        logger.debug(f"Creating database service: {service_name}")

        db_type = config['type']

        if db_type == 'postgres':
            # PostgreSQL database service (from Epic 16)
            if 'connection_string' in config:
                conn_str = config['connection_string']
            else:
                # Build connection string from components
                conn_str = (
                    f"postgresql://{config['user']}:{config['password']}"
                    f"@{config['host']}:{config.get('port', 5432)}"
                    f"/{config['database']}"
                )

            return PostgresqlDatabaseService(
                connection_string=conn_str,
                pool_size=config.get('pool_size', 10),
                max_overflow=config.get('max_overflow', 20)
            )
        elif db_type == 'neo4j':
            # Neo4j graph database service
            # Get URI - either explicit or build from host/port
            if 'uri' in config:
                uri = config['uri']
            else:
                host = config.get('host', 'localhost')
                port = config.get('port', 7687)
                uri = f"bolt://{host}:{port}"
            
            return Neo4jGraphService(
                uri=uri,
                user=config['user'],
                password=config['password']
            )
        else:
            raise ServiceInstantiationError(
                f"Unknown database type: {db_type}"
            )
