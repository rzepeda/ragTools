"""Service Registry for RAG Factory.

This package provides the service registry system for Epic 17:
- ServiceRegistry: Central registry for service definitions and instances
- ServiceFactory: Factory for creating service instances from configurations
- Custom exceptions for registry errors

The registry loads service configurations from YAML and creates/caches service instances.
Multiple strategies can share the same service instance for efficiency.

Example usage:
    >>> from rag_factory.registry import ServiceRegistry
    >>> 
    >>> # Initialize registry
    >>> registry = ServiceRegistry("config/services.yaml")
    >>> 
    >>> # Get services (creates on first call, returns cached on subsequent calls)
    >>> llm = registry.get("$llm1")
    >>> embedding = registry.get("embedding1")  # $ prefix is optional
    >>> 
    >>> # List available services
    >>> print(registry.list_services())
    >>> 
    >>> # Cleanup
    >>> registry.shutdown()
"""

from .service_registry import ServiceRegistry
from .service_factory import ServiceFactory
from .exceptions import (
    ServiceRegistryError,
    ServiceNotFoundError,
    ServiceInstantiationError,
)

__all__ = [
    "ServiceRegistry",
    "ServiceFactory",
    "ServiceRegistryError",
    "ServiceNotFoundError",
    "ServiceInstantiationError",
]
