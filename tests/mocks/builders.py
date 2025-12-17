"""Base builder classes for mock objects.

This module provides base classes and utilities for building consistent
mock objects across the test suite.
"""

from typing import Any, Dict, Optional, Callable
from unittest.mock import Mock, AsyncMock, MagicMock


class MockBuilder:
    """Base class for building mock objects with fluent interface.
    
    Example:
        builder = MockBuilder()
        builder.with_method("foo", return_value=42)
        builder.with_async_method("bar", return_value="async result")
        mock = builder.build()
    """
    
    def __init__(self, spec: Optional[type] = None):
        """Initialize builder.
        
        Args:
            spec: Optional class to use as spec for the mock
        """
        self.spec = spec
        self.methods: Dict[str, Any] = {}
        self.async_methods: Dict[str, Any] = {}
        self.properties: Dict[str, Any] = {}
        self.attributes: Dict[str, Any] = {}
    
    def with_method(
        self, 
        name: str, 
        return_value: Any = None,
        side_effect: Optional[Callable] = None
    ) -> 'MockBuilder':
        """Add a synchronous method to the mock.
        
        Args:
            name: Method name
            return_value: Value to return when called
            side_effect: Optional callable for dynamic behavior
            
        Returns:
            Self for chaining
        """
        self.methods[name] = {
            'return_value': return_value,
            'side_effect': side_effect
        }
        return self
    
    def with_async_method(
        self,
        name: str,
        return_value: Any = None,
        side_effect: Optional[Callable] = None
    ) -> 'MockBuilder':
        """Add an asynchronous method to the mock.
        
        Args:
            name: Method name
            return_value: Value to return when awaited
            side_effect: Optional callable for dynamic behavior
            
        Returns:
            Self for chaining
        """
        self.async_methods[name] = {
            'return_value': return_value,
            'side_effect': side_effect
        }
        return self
    
    def with_property(self, name: str, value: Any) -> 'MockBuilder':
        """Add a property to the mock.
        
        Args:
            name: Property name
            value: Property value
            
        Returns:
            Self for chaining
        """
        self.properties[name] = value
        return self
    
    def with_attribute(self, name: str, value: Any) -> 'MockBuilder':
        """Add an attribute to the mock.
        
        Args:
            name: Attribute name
            value: Attribute value
            
        Returns:
            Self for chaining
        """
        self.attributes[name] = value
        return self
    
    def build(self) -> Mock:
        """Build the mock object.
        
        Returns:
            Configured Mock instance
        """
        if self.spec:
            mock = Mock(spec=self.spec)
        else:
            mock = Mock()
        
        # Add synchronous methods
        for name, config in self.methods.items():
            method_mock = Mock(
                return_value=config['return_value'],
                side_effect=config['side_effect']
            )
            setattr(mock, name, method_mock)
        
        # Add asynchronous methods
        for name, config in self.async_methods.items():
            method_mock = AsyncMock(
                return_value=config['return_value'],
                side_effect=config['side_effect']
            )
            setattr(mock, name, method_mock)
        
        # Add properties
        for name, value in self.properties.items():
            type(mock).name = property(lambda self: value)
        
        # Add attributes
        for name, value in self.attributes.items():
            setattr(mock, name, value)
        
        return mock


class ServiceMockBuilder(MockBuilder):
    """Base builder for service mocks.
    
    Provides common service functionality like initialization and cleanup.
    """
    
    def __init__(self, spec: Optional[type] = None):
        super().__init__(spec)
        # Add common service methods
        self.with_async_method('close', return_value=None)
        self.with_method('__enter__', return_value=None)
        self.with_method('__exit__', return_value=None)


def create_mock_with_context_manager(
    mock_obj: Optional[Mock] = None
) -> Mock:
    """Create a mock that works as a context manager.
    
    Args:
        mock_obj: Optional mock to enhance, creates new if None
        
    Returns:
        Mock configured as context manager
        
    Example:
        mock = create_mock_with_context_manager()
        with mock as ctx:
            # ctx is the same mock
            pass
    """
    if mock_obj is None:
        mock_obj = Mock()
    
    mock_obj.__enter__ = Mock(return_value=mock_obj)
    mock_obj.__exit__ = Mock(return_value=None)
    
    return mock_obj


def create_async_context_manager_mock(
    mock_obj: Optional[Mock] = None
) -> Mock:
    """Create a mock that works as an async context manager.
    
    Args:
        mock_obj: Optional mock to enhance, creates new if None
        
    Returns:
        Mock configured as async context manager
        
    Example:
        mock = create_async_context_manager_mock()
        async with mock as ctx:
            # ctx is the same mock
            pass
    """
    if mock_obj is None:
        mock_obj = Mock()
    
    mock_obj.__aenter__ = AsyncMock(return_value=mock_obj)
    mock_obj.__aexit__ = AsyncMock(return_value=None)
    
    return mock_obj


def configure_mock_call_tracking(mock_obj: Mock) -> Mock:
    """Configure a mock to track all method calls.
    
    Args:
        mock_obj: Mock to configure
        
    Returns:
        Same mock with call tracking enabled
    """
    mock_obj.reset_mock()
    return mock_obj
