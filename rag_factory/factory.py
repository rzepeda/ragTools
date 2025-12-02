"""
RAG Factory for creating and managing RAG strategy instances.

This module provides a factory class for instantiating RAG strategies,
managing strategy registration, and handling configuration-based
strategy creation.

Example usage:
    >>> from rag_factory.factory import RAGFactory, register_rag_strategy
    >>> from rag_factory.strategies.base import IRAGStrategy, StrategyConfig
    >>>
    >>> # Register a strategy
    >>> factory = RAGFactory()
    >>> factory.register_strategy("my_strategy", MyStrategyClass)
    >>>
    >>> # Create a strategy instance
    >>> config = {"chunk_size": 1024, "top_k": 10}
    >>> strategy = factory.create_strategy("my_strategy", config)
    >>>
    >>> # Or use decorator for auto-registration
    >>> @register_rag_strategy("auto_strategy")
    >>> class AutoStrategy(IRAGStrategy):
    ...     pass
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import yaml

from rag_factory.strategies.base import IRAGStrategy, StrategyConfig


class StrategyNotFoundError(Exception):
    """Raised when strategy name not found in registry."""


class ConfigurationError(Exception):
    """Raised when strategy configuration is invalid."""


class RAGFactory:
    """
    Factory for creating RAG strategy instances.

    This factory provides a centralized mechanism for registering and
    creating RAG strategies. It supports:
    - Dynamic strategy registration
    - Configuration-based instantiation
    - File-based configuration (YAML/JSON)
    - Dependency injection
    - Decorator-based auto-registration

    The factory maintains a class-level registry that is shared across
    all factory instances, allowing for global strategy management.

    Example:
        >>> factory = RAGFactory()
        >>> factory.register_strategy("simple", SimpleStrategy)
        >>> strategy = factory.create_strategy("simple", {"chunk_size": 512})
        >>> chunks = strategy.retrieve("query", top_k=5)
    """

    _registry: Dict[str, Type[IRAGStrategy]] = {}
    _dependencies: Dict[str, Any] = {}

    @classmethod
    def register_strategy(
        cls,
        name: str,
        strategy_class: Type[IRAGStrategy],
        override: bool = False,
    ) -> None:
        """
        Register a strategy class with the factory.

        Args:
            name: Unique identifier for the strategy
            strategy_class: The strategy class to register (must implement
                          IRAGStrategy)
            override: If True, allow overriding existing registrations
                     (default: False)

        Raises:
            ValueError: If strategy name already registered and override
                       is False

        Example:
            >>> factory = RAGFactory()
            >>> factory.register_strategy("my_strategy", MyStrategy)
        """
        if name in cls._registry and not override:
            raise ValueError(
                f"Strategy '{name}' already registered. "
                f"Use override=True to replace."
            )
        cls._registry[name] = strategy_class

    @classmethod
    def unregister_strategy(cls, name: str) -> None:
        """
        Remove a strategy from the registry.

        Args:
            name: Name of the strategy to remove

        Raises:
            KeyError: If strategy name not found in registry

        Example:
            >>> factory = RAGFactory()
            >>> factory.unregister_strategy("old_strategy")
        """
        if name not in cls._registry:
            raise KeyError(
                f"Strategy '{name}' not found in registry. "
                f"Available: {list(cls._registry.keys())}"
            )
        del cls._registry[name]

    @classmethod
    def create_strategy(
        cls, name: str, config: Optional[Dict[str, Any]] = None
    ) -> IRAGStrategy:
        """
        Create and initialize a strategy instance.

        Args:
            name: Name of the registered strategy to create
            config: Optional configuration dictionary. If provided,
                   will be used to create a StrategyConfig and
                   initialize the strategy.

        Returns:
            IRAGStrategy: Initialized strategy instance

        Raises:
            StrategyNotFoundError: If strategy name not in registry
            ValueError: If configuration is invalid
            RuntimeError: If strategy initialization fails

        Example:
            >>> factory = RAGFactory()
            >>> config = {"chunk_size": 1024, "top_k": 10}
            >>> strategy = factory.create_strategy("my_strategy", config)
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise StrategyNotFoundError(
                f"Strategy '{name}' not found. "
                f"Available strategies: {available}"
            )

        strategy_class = cls._registry[name]
        strategy = strategy_class()

        if config:
            strategy_config = StrategyConfig(**config)
            strategy.initialize(strategy_config)

        return strategy

    @classmethod
    def create_from_config(cls, config_path: str) -> IRAGStrategy:
        """
        Create strategy from configuration file.

        Supports both YAML and JSON configuration files. The configuration
        file must include a 'strategy_name' field that identifies which
        registered strategy to instantiate.

        Args:
            config_path: Path to YAML or JSON configuration file

        Returns:
            IRAGStrategy: Initialized strategy instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ConfigurationError: If config is missing required fields
            yaml.YAMLError: If YAML parsing fails
            json.JSONDecodeError: If JSON parsing fails

        Example:
            >>> # config.yaml contains:
            >>> # strategy_name: my_strategy
            >>> # chunk_size: 1024
            >>> # top_k: 10
            >>> factory = RAGFactory()
            >>> strategy = factory.create_from_config("config.yaml")
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load config based on file extension
        if config_file.suffix in [".yaml", ".yml"]:
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        elif config_file.suffix == ".json":
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        else:
            raise ConfigurationError(
                f"Unsupported config file format: {config_file.suffix}. "
                f"Use .yaml, .yml, or .json"
            )

        if not config_dict or "strategy_name" not in config_dict:
            raise ConfigurationError(
                "Configuration file must include 'strategy_name' field"
            )

        strategy_name = config_dict.pop("strategy_name")
        return cls.create_strategy(strategy_name, config_dict)

    @classmethod
    def create_strategy_from_config(cls, config_path: str) -> IRAGStrategy:
        """
        Alias for create_from_config for backward compatibility.

        Args:
            config_path: Path to configuration file

        Returns:
            IRAGStrategy: Initialized strategy instance
        """
        return cls.create_from_config(config_path)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        List all registered strategy names.

        Returns:
            List[str]: List of registered strategy names

        Example:
            >>> factory = RAGFactory()
            >>> factory.register_strategy("strategy1", Strategy1)
            >>> factory.register_strategy("strategy2", Strategy2)
            >>> print(factory.list_strategies())
            ['strategy1', 'strategy2']
        """
        return list(cls._registry.keys())

    @classmethod
    def set_dependency(cls, name: str, dependency: Any) -> None:
        """
        Register a dependency for injection into strategies.

        Dependencies can be retrieved later and injected into strategies
        that need external services (e.g., embedding service, LLM client).

        Args:
            name: Unique identifier for the dependency
            dependency: The dependency object to register

        Example:
            >>> factory = RAGFactory()
            >>> embedding_service = EmbeddingService()
            >>> factory.set_dependency("embedding_service", embedding_service)
        """
        cls._dependencies[name] = dependency

    @classmethod
    def get_dependency(cls, name: str) -> Optional[Any]:
        """
        Retrieve a registered dependency.

        Args:
            name: Name of the dependency to retrieve

        Returns:
            Optional[Any]: The dependency object, or None if not found

        Example:
            >>> factory = RAGFactory()
            >>> service = factory.get_dependency("embedding_service")
        """
        return cls._dependencies.get(name)

    @classmethod
    def clear_registry(cls) -> None:
        """
        Clear all registered strategies.

        This method is primarily useful for testing to ensure a clean
        state between tests.

        Warning:
            This will remove all registered strategies globally.

        Example:
            >>> factory = RAGFactory()
            >>> factory.clear_registry()
        """
        cls._registry.clear()

    @classmethod
    def clear_dependencies(cls) -> None:
        """
        Clear all registered dependencies.

        This method is primarily useful for testing to ensure a clean
        state between tests.

        Warning:
            This will remove all registered dependencies globally.

        Example:
            >>> factory = RAGFactory()
            >>> factory.clear_dependencies()
        """
        cls._dependencies.clear()


def register_rag_strategy(name: str) -> Callable[[Type[IRAGStrategy]], Type[IRAGStrategy]]:
    """
    Decorator to auto-register strategy classes.

    This decorator provides a convenient way to register strategies
    at class definition time, without needing to manually call
    register_strategy().

    Args:
        name: Unique identifier for the strategy

    Returns:
        Callable: Decorator function that registers the strategy

    Example:
        >>> @register_rag_strategy("simple_strategy")
        >>> class SimpleStrategy(IRAGStrategy):
        ...     def initialize(self, config):
        ...         pass
        ...     # ... other methods
        >>>
        >>> # Strategy is now automatically registered
        >>> factory = RAGFactory()
        >>> strategy = factory.create_strategy("simple_strategy")
    """

    def decorator(cls: Type[IRAGStrategy]) -> Type[IRAGStrategy]:
        """Register the strategy class and return it unchanged."""
        RAGFactory.register_strategy(name, cls)
        return cls

    return decorator
