"""Custom exceptions for service registry."""


class ServiceRegistryError(Exception):
    """Base exception for service registry errors."""
    pass


class ServiceNotFoundError(ServiceRegistryError):
    """Service not found in registry."""
    pass


class ServiceInstantiationError(ServiceRegistryError):
    """Service instantiation failed."""
    pass
