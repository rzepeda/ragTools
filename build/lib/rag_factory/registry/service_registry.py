"""Central registry for service definitions and instances."""

from threading import Lock
from collections import defaultdict
from typing import Dict, Any, Optional, List
import logging
import yaml
import time

from rag_factory.config.validator import ConfigValidator
from rag_factory.config.env_resolver import EnvResolver
from .service_factory import ServiceFactory
from .exceptions import ServiceNotFoundError, ServiceInstantiationError

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Central registry for service definitions and instances.
    
    Loads service configurations from YAML and creates/caches service instances.
    Multiple strategies can share the same service instance for efficiency.
    
    Example:
        >>> registry = ServiceRegistry("config/services.yaml")
        >>> llm = registry.get("$llm1")
        >>> embedding = registry.get("embedding1")  # $ prefix optional
        >>> registry.shutdown()
        
        >>> # Or use as context manager
        >>> with ServiceRegistry("config/services.yaml") as registry:
        ...     llm = registry.get("llm1")
        ...     # Use services...
        >>> # Automatic cleanup on exit
    """

    def __init__(self, config_path: str = "config/services.yaml"):
        """Initialize service registry from configuration file.
        
        Args:
            config_path: Path to services.yaml configuration
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._instances: Dict[str, Any] = {}  # service_name -> instance
        self._locks = defaultdict(Lock)  # service_name -> lock
        self._validator = ConfigValidator()
        self._factory = ServiceFactory()

        # Load configuration
        self._load_config()

    def _load_config(self) -> None:
        """Load and validate services.yaml configuration."""
        logger.info(f"Loading service registry from: {self.config_path}")

        try:
            # Load YAML
            with open(self.config_path, 'r') as f:
                raw_config = yaml.safe_load(f)

            # Validate schema
            warnings = self._validator.validate_services_yaml(
                raw_config,
                file_path=self.config_path
            )

            # Print warnings
            for warning in warnings:
                logger.warning(warning)

            # Resolve environment variables
            self.config = EnvResolver.resolve(raw_config)

            logger.info(
                f"Service registry loaded: "
                f"{len(self.config.get('services', {}))} services available"
            )

        except FileNotFoundError:
            raise ServiceInstantiationError(
                f"Service registry configuration not found: {self.config_path}"
            )
        except Exception as e:
            raise ServiceInstantiationError(
                f"Failed to load service registry: {e}"
            )

    def get(self, service_ref: str) -> Any:
        """Get or create a service instance.
        
        Args:
            service_ref: Service reference like "$llm1" or "llm1"
            
        Returns:
            Service instance implementing appropriate interface
            
        Raises:
            ServiceNotFoundError: If service not found in registry
            ServiceInstantiationError: If service creation fails
        """
        # Strip $ prefix if present
        service_name = service_ref.lstrip('$')

        # Return cached instance if exists
        if service_name in self._instances:
            logger.debug(f"Service '{service_name}' returned from cache")
            return self._instances[service_name]

        # Thread-safe instantiation
        with self._locks[service_name]:
            # Double-check after acquiring lock
            if service_name in self._instances:
                return self._instances[service_name]

            # Validate service exists
            if 'services' not in self.config:
                raise ServiceNotFoundError(
                    f"No services defined in registry configuration"
                )

            if service_name not in self.config['services']:
                available = list(self.config['services'].keys())
                raise ServiceNotFoundError(
                    f"Service '{service_name}' not found in registry. "
                    f"Available services: {available}"
                )

            # Get service configuration
            service_config = self.config['services'][service_name]

            # Create service instance
            logger.info(f"Instantiating service: {service_name}")
            start_time = time.time()

            try:
                service_instance = self._factory.create_service(
                    service_name,
                    service_config
                )
                instantiation_time = time.time() - start_time

                logger.info(
                    f"Service '{service_name}' instantiated successfully "
                    f"in {instantiation_time:.2f}s"
                )

            except Exception as e:
                logger.error(f"Failed to instantiate service '{service_name}': {e}")
                raise ServiceInstantiationError(
                    f"Service instantiation failed for '{service_name}': {e}"
                )

            # Cache and return
            self._instances[service_name] = service_instance
            return service_instance

    def list_services(self) -> List[str]:
        """List all available service names.
        
        Returns:
            List of service names from configuration
        """
        return list(self.config.get('services', {}).keys())

    def reload(self, service_name: str) -> Any:
        """Force reload a service (useful after config changes).
        
        Closes old instance if it has a close() method, removes from cache,
        and returns newly instantiated service.
        
        Args:
            service_name: Service to reload (without $ prefix)
            
        Returns:
            New service instance
        """
        service_name = service_name.lstrip('$')

        logger.info(f"Reloading service: {service_name}")

        # Clean up old instance
        if service_name in self._instances:
            old_instance = self._instances[service_name]

            # Try to close gracefully
            if hasattr(old_instance, 'close'):
                try:
                    old_instance.close()
                    logger.debug(f"Closed old instance of '{service_name}'")
                except Exception as e:
                    logger.warning(
                        f"Error closing old service instance '{service_name}': {e}"
                    )

            # Remove from cache
            del self._instances[service_name]

        # Reload configuration
        self._load_config()

        # Next get() will create new instance
        return self.get(service_name)

    def shutdown(self) -> None:
        """Close all service instances and cleanup resources.
        
        Calls close() method on all services that support it.
        """
        logger.info("Shutting down service registry")

        for service_name, instance in self._instances.items():
            if hasattr(instance, 'close'):
                try:
                    instance.close()
                    logger.debug(f"Closed service: {service_name}")
                except Exception as e:
                    logger.warning(
                        f"Error closing service '{service_name}': {e}"
                    )

        self._instances.clear()
        logger.info("Service registry shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()
        return False
