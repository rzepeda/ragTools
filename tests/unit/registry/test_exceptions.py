"""Unit tests for service registry exceptions."""

import pytest
from rag_factory.registry.exceptions import (
    ServiceRegistryError,
    ServiceNotFoundError,
    ServiceInstantiationError,
)


def test_service_registry_error():
    """Test base ServiceRegistryError exception."""
    error = ServiceRegistryError("Test error")
    assert str(error) == "Test error"
    assert isinstance(error, Exception)


def test_service_not_found_error():
    """Test ServiceNotFoundError exception."""
    error = ServiceNotFoundError("Service not found")
    assert str(error) == "Service not found"
    assert isinstance(error, ServiceRegistryError)
    assert isinstance(error, Exception)


def test_service_instantiation_error():
    """Test ServiceInstantiationError exception."""
    error = ServiceInstantiationError("Instantiation failed")
    assert str(error) == "Instantiation failed"
    assert isinstance(error, ServiceRegistryError)
    assert isinstance(error, Exception)
