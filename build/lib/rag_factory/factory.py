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
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Set

import yaml

from rag_factory.strategies.base import IRAGStrategy, StrategyConfig
from rag_factory.exceptions import StrategyNotFoundError, ConfigurationError
from rag_factory.services.interfaces import (
    ILLMService,
    IEmbeddingService,
    IGraphService,
    IDatabaseService,
    IRerankingService,
)
from rag_factory.services.dependencies import StrategyDependencies
from rag_factory.services.consistency import ConsistencyChecker
from rag_factory.core.capabilities import IndexCapability, ValidationResult
from rag_factory.core.pipeline import IndexingPipeline, RetrievalPipeline


logger = logging.getLogger(__name__)


class RAGFactory:
    """
    Factory for creating RAG strategy instances with dependency injection.

    This factory provides a centralized mechanism for registering and
    creating RAG strategies. It supports:
    - Dynamic strategy registration
    - Configuration-based instantiation
    - File-based configuration (YAML/JSON)
    - Dependency injection for services
    - Decorator-based auto-registration

    The factory maintains a class-level registry that is shared across
    all factory instances, allowing for global strategy management.
    Each factory instance can have its own set of service dependencies.

    Example:
        >>> # Create factory with services
        >>> factory = RAGFactory(
        ...     llm_service=my_llm,
        ...     embedding_service=my_embedder
        ... )
        >>> factory.register_strategy("simple", SimpleStrategy)
        >>> strategy = factory.create_strategy("simple", {"chunk_size": 512})
        >>> chunks = strategy.retrieve("query", top_k=5)
    """

    _registry: Dict[str, Type[IRAGStrategy]] = {}
    _dependencies: Dict[str, Any] = {}  # Legacy class-level dependencies

    def __init__(
        self,
        llm_service: Optional[ILLMService] = None,
        embedding_service: Optional[IEmbeddingService] = None,
        graph_service: Optional[IGraphService] = None,
        database_service: Optional[IDatabaseService] = None,
        reranker_service: Optional[IRerankingService] = None,
    ) -> None:
        """
        Initialize the factory with optional service dependencies.

        Args:
            llm_service: Optional LLM service for text generation
            embedding_service: Optional embedding service for vector generation
            graph_service: Optional graph database service
            database_service: Optional database service for storage
            reranker_service: Optional reranking service

        Example:
            >>> # Create factory with specific services
            >>> from rag_factory.services.implementations.llm import AnthropicLLMService
            >>> from rag_factory.services.implementations.embedding import ONNXEmbeddingService
            >>>
            >>> factory = RAGFactory(
            ...     llm_service=AnthropicLLMService(api_key="..."),
            ...     embedding_service=ONNXEmbeddingService(model_path="...")
            ... )
        """
        self.dependencies = StrategyDependencies(
            llm_service=llm_service,
            embedding_service=embedding_service,
            graph_service=graph_service,
            database_service=database_service,
            reranker_service=reranker_service,
        )
        self.consistency_checker = ConsistencyChecker()

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

    def create_strategy(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        override_deps: Optional[StrategyDependencies] = None,
    ) -> IRAGStrategy:
        """
        Create and initialize a strategy instance with dependency injection.

        Args:
            name: Name of the registered strategy to create
            config: Optional configuration dictionary for the strategy
            override_deps: Optional dependencies to override factory defaults

        Returns:
            IRAGStrategy: Initialized strategy instance

        Raises:
            StrategyNotFoundError: If strategy name not in registry
            ValueError: If required services are missing or config is invalid

        Example:
            >>> factory = RAGFactory(llm_service=my_llm)
            >>> config = {"chunk_size": 1024, "top_k": 10}
            >>> strategy = factory.create_strategy("my_strategy", config)
            >>>
            >>> # Or with override dependencies
            >>> override = StrategyDependencies(llm_service=different_llm)
            >>> strategy = factory.create_strategy("my_strategy", config, override)
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise StrategyNotFoundError(
                f"Strategy '{name}' not found. "
                f"Available strategies: {available}"
            )

        strategy_class = self._registry[name]
        deps = override_deps or self.dependencies

        # Use config dict or empty dict if not provided
        strategy_config = config or {}

        try:
            # Strategy constructor will validate dependencies
            strategy = strategy_class(config=strategy_config, dependencies=deps)
            
            # Check consistency between capabilities and services
            # Note: This requires strategies to implement produces()/requires() methods
            # from Epic 12. If these methods don't exist, we skip consistency checking.
            self._check_strategy_consistency(strategy)
            
        except ValueError as e:
            # Re-raise with context about which strategy failed
            raise ValueError(
                f"Failed to create strategy '{name}': {str(e)}"
            ) from e

        return strategy

    @classmethod
    def create_strategy_legacy(
        cls, name: str, config: Optional[Dict[str, Any]] = None
    ) -> IRAGStrategy:
        """
        Create strategy using legacy initialization pattern (deprecated).

        This method maintains backward compatibility with strategies that
        use the old initialize() pattern instead of dependency injection.

        Args:
            name: Name of the registered strategy to create
            config: Optional configuration dictionary

        Returns:
            IRAGStrategy: Initialized strategy instance

        Raises:
            StrategyNotFoundError: If strategy name not in registry

        Note:
            This method is deprecated and will be removed in a future version.
            Use create_strategy() with dependency injection instead.
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

    def create_from_config(self, config_path: str) -> IRAGStrategy:
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
            >>> factory = RAGFactory(llm_service=my_llm)
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
        return self.create_strategy(strategy_name, config_dict)

    @classmethod
    def create_strategy_from_config(cls, config_path: str) -> IRAGStrategy:
        """
        Alias for create_from_config for backward compatibility.

        This method creates a temporary factory instance with no services
        to maintain backward compatibility with code that calls this as
        a class method.

        Args:
            config_path: Path to configuration file

        Returns:
            IRAGStrategy: Initialized strategy instance

        Note:
            This method is deprecated. Use an instance method instead:
            factory = RAGFactory(services...)
            factory.create_from_config(config_path)
        """
        # Create temporary factory instance with no services
        temp_factory = cls()
        return temp_factory.create_from_config(config_path)

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

    def _check_strategy_consistency(self, strategy: IRAGStrategy) -> None:
        """Check and log consistency warnings for a strategy.
        
        This method checks if the strategy's capabilities are consistent with
        its service dependencies. It logs warnings for any inconsistencies but
        does not prevent strategy creation.
        
        Note: This requires strategies to implement produces()/requires() methods
        from Epic 12. If these methods don't exist, consistency checking is skipped.
        
        Args:
            strategy: The strategy instance to check
        """
        # Check if strategy has produces() method (indexing strategy)
        if hasattr(strategy, 'produces') and callable(getattr(strategy, 'produces')):
            warnings = self.consistency_checker.check_indexing_strategy(strategy)
            for warning in warnings:
                logger.warning(warning)
        
        # Check if strategy has requires() method (retrieval strategy)
        if hasattr(strategy, 'requires') and callable(getattr(strategy, 'requires')):
            warnings = self.consistency_checker.check_retrieval_strategy(strategy)
            for warning in warnings:
                logger.warning(warning)

    def validate_compatibility(
        self,
        indexing_pipeline: IndexingPipeline,
        retrieval_pipeline: RetrievalPipeline
    ) -> ValidationResult:
        """
        Validate capability compatibility between pipelines.
        
        Also checks consistency of strategies (warns, doesn't fail).
        
        Args:
            indexing_pipeline: Indexing pipeline to validate
            retrieval_pipeline: Retrieval pipeline to validate
            
        Returns:
            ValidationResult indicating compatibility
        """
        # Check consistency of strategies (warnings only)
        for strategy in indexing_pipeline.strategies:
            self.consistency_checker.check_indexing_strategy(strategy)
        
        for strategy in retrieval_pipeline.strategies:
            self.consistency_checker.check_retrieval_strategy(strategy)
        
        # Check capability compatibility (can fail)
        capabilities = indexing_pipeline.get_capabilities()
        requirements = retrieval_pipeline.get_requirements()
        
        missing_caps = requirements - capabilities
        
        if missing_caps:
            suggestions = self._generate_suggestions(missing_caps)
            return ValidationResult(
                is_valid=False,
                missing_capabilities=missing_caps,
                missing_services=set(),
                message=f"Missing capabilities: {[c.name for c in missing_caps]}",
                suggestions=suggestions
            )
        
        return ValidationResult(
            is_valid=True,
            missing_capabilities=set(),
            missing_services=set(),
            message="Pipelines are compatible",
            suggestions=[]
        )
    
    def validate_pipeline(
        self,
        indexing_pipeline: IndexingPipeline,
        retrieval_pipeline: RetrievalPipeline
    ) -> ValidationResult:
        """
        Full validation: capabilities AND services.
        
        Also runs consistency checks (warns about suspicious patterns).
        
        Args:
            indexing_pipeline: Indexing pipeline
            retrieval_pipeline: Retrieval pipeline
            
        Returns:
            Complete ValidationResult
        """
        # Check capabilities (includes consistency checking)
        cap_validation = self.validate_compatibility(indexing_pipeline, retrieval_pipeline)
        if not cap_validation.is_valid:
            return cap_validation
        
        # Check services (already validated at pipeline creation, but double-check)
        service_reqs = retrieval_pipeline.get_service_requirements()
        is_valid, missing = self.dependencies.validate_for_strategy(service_reqs)
        
        if not is_valid:
            return ValidationResult(
                is_valid=False,
                missing_capabilities=set(),
                missing_services=set(missing),
                message=f"Missing services: {[s.name for s in missing]}",
                suggestions=[]
            )
        
        return ValidationResult(
            is_valid=True,
            missing_capabilities=set(),
            missing_services=set(),
            message="Pipeline fully valid (capabilities and services)",
            suggestions=[]
        )

    def auto_select_retrieval(
        self,
        indexing_pipeline: IndexingPipeline,
        preferred_strategies: Optional[List[str]] = None
    ) -> List[str]:
        """
        Select compatible retrieval strategies based on indexing capabilities.
        
        Args:
            indexing_pipeline: The indexing pipeline providing capabilities
            preferred_strategies: Optional list of preferred strategy names
            
        Returns:
            List of compatible strategy names
        """
        capabilities = indexing_pipeline.get_capabilities()
        compatible = []
        
        # Filter available strategies
        candidates = []
        if preferred_strategies:
            candidates.extend(preferred_strategies)
        
        # Add others
        for name in self._registry:
            if name not in candidates:
                candidates.append(name)
                
        final_selection = []
        
        for name in candidates:
            if name not in self._registry:
                continue
                
            strategy_cls = self._registry[name]
            
            # Check if it's a retrieval strategy (has requires method)
            # We check the class for the method presence
            if not hasattr(strategy_cls, 'requires'):
                continue
                
            # Try to check compatibility
            try:
                # We need an instance to call requires() as per protocol
                # We'll try to create one with current factory deps
                # This might fail if deps are missing, which is fine (not compatible)
                strategy = self.create_strategy(name)
                reqs = strategy.requires()
                
                if reqs.issubset(capabilities):
                    final_selection.append(name)
            except Exception:
                # If we can't create it, it's not compatible/available
                continue
                
        if not final_selection:
            logger.warning("No compatible retrieval strategies found for current indexing capabilities.")
            
        return final_selection

    def check_all_strategies(
        self,
        strategy_filter: Optional[List[str]] = None,
        type_filter: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Check all registered strategies for consistency issues.
        
        This method iterates through registered strategies and checks for
        consistency between their capabilities and service dependencies.
        It returns warnings without blocking strategy usage.
        
        Args:
            strategy_filter: Optional list of strategy names to check.
                           If None, checks all registered strategies.
            type_filter: Optional filter by type ('indexing', 'retrieval', 'all').
                        Defaults to 'all'.
        
        Returns:
            Dict mapping strategy names to their check results. Each result contains:
            - 'type': 'indexing' or 'retrieval' or 'unknown'
            - 'warnings': List of warning messages
            - 'error': Optional error message if strategy couldn't be checked
        
        Example:
            >>> factory = RAGFactory()
            >>> results = factory.check_all_strategies()
            >>> for name, result in results.items():
            ...     if result['warnings']:
            ...         print(f"{name}: {result['warnings']}")
        """
        results = {}
        
        # Determine which strategies to check
        strategies_to_check = strategy_filter if strategy_filter else list(self._registry.keys())
        
        for name in strategies_to_check:
            if name not in self._registry:
                results[name] = {
                    'type': 'unknown',
                    'warnings': [],
                    'error': f"Strategy '{name}' not found in registry"
                }
                continue
            
            strategy_class = self._registry[name]
            
            try:
                # Try to instantiate the strategy with current dependencies
                # Use empty config for consistency checking
                strategy = self.create_strategy(name, config={})
                
                # Determine strategy type and run appropriate checks
                strategy_type = 'unknown'
                warnings = []
                
                # Check if it's an indexing strategy
                if hasattr(strategy, 'produces') and callable(getattr(strategy, 'produces')):
                    strategy_type = 'indexing'
                    if type_filter in [None, 'all', 'indexing']:
                        warnings = self.consistency_checker.check_indexing_strategy(strategy)
                
                # Check if it's a retrieval strategy
                elif hasattr(strategy, 'requires') and callable(getattr(strategy, 'requires')):
                    strategy_type = 'retrieval'
                    if type_filter in [None, 'all', 'retrieval']:
                        warnings = self.consistency_checker.check_retrieval_strategy(strategy)
                
                # Skip if type doesn't match filter
                if type_filter and type_filter != 'all' and strategy_type != type_filter:
                    continue
                
                results[name] = {
                    'type': strategy_type,
                    'warnings': warnings,
                    'error': None
                }
                
            except Exception as e:
                # If we can't instantiate the strategy, record the error
                results[name] = {
                    'type': 'unknown',
                    'warnings': [],
                    'error': f"Could not instantiate strategy: {str(e)}"
                }
        
        return results

    def _generate_suggestions(self, missing_caps: Set[IndexCapability]) -> List[str]:
        """Generate suggestions for missing capabilities."""
        suggestions = []
        for cap in missing_caps:
            if cap == IndexCapability.VECTORS:
                suggestions.append("Add a VectorEmbeddingIndexing strategy to your indexing pipeline")
            elif cap == IndexCapability.KEYWORDS:
                suggestions.append("Add a KeywordIndexing strategy to your indexing pipeline")
            elif cap == IndexCapability.GRAPH:
                suggestions.append("Add a GraphIndexing strategy to your indexing pipeline")
            elif cap == IndexCapability.DATABASE:
                suggestions.append("Ensure you have a strategy that persists to database")
            else:
                suggestions.append(f"Add a strategy that produces {cap.name}")
        return suggestions


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
