"""
Configuration Management System for RAG Factory.

This module provides centralized configuration management with validation,
environment-specific overrides, hot-reload capability, and a simple access API.

Example usage:
    >>> from rag_factory.config import ConfigManager
    >>>
    >>> # Load configuration
    >>> config = ConfigManager()
    >>> config.load("config.yaml")
    >>>
    >>> # Access configuration
    >>> log_level = config.get("global_settings.log_level")
    >>> strategy_config = config.get_strategy_config("my_strategy")
    >>>
    >>> # Enable hot-reload in development
    >>> config.enable_hot_reload(callback=lambda cfg: print("Config reloaded"))
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast, TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer as WatchdogObserver


class ConfigurationError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""


class StrategyConfigSchema(BaseModel):
    """
    Schema for strategy configuration.

    Attributes:
        chunk_size: Size of text chunks (1-8192)
        chunk_overlap: Overlap between chunks (0-500)
        top_k: Number of results to retrieve (1-100)
        strategy_name: Strategy identifier
        metadata: Additional strategy-specific parameters
    """

    chunk_size: int = Field(
        512,
        ge=1,
        le=8192,
        description="Size of text chunks"
    )
    chunk_overlap: int = Field(
        50,
        ge=0,
        le=500,
        description="Overlap between chunks"
    )
    top_k: int = Field(
        5,
        ge=1,
        le=100,
        description="Number of results to retrieve"
    )
    strategy_name: str = Field(
        "",
        description="Strategy identifier"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters"
    )


class PipelineConfigSchema(BaseModel):
    """
    Schema for pipeline configuration.

    Attributes:
        mode: Execution mode (sequential, parallel, cascade)
        stages: List of pipeline stages
        timeout: Pipeline timeout in seconds
    """

    mode: str = Field(
        "sequential",
        description="Execution mode"
    )
    stages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Pipeline stages"
    )
    timeout: Optional[int] = Field(
        None,
        description="Pipeline timeout in seconds"
    )


class GlobalConfigSchema(BaseModel):
    """
    Schema for global configuration.

    Attributes:
        environment: Environment name (development, test, production)
        log_level: Logging level
        cache_enabled: Enable result caching
        cache_ttl: Cache TTL in seconds
    """

    environment: str = Field(
        "development",
        description="Environment name"
    )
    log_level: str = Field(
        "INFO",
        description="Logging level"
    )
    cache_enabled: bool = Field(
        True,
        description="Enable result caching"
    )
    cache_ttl: int = Field(
        3600,
        ge=0,
        description="Cache TTL in seconds"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(
                f"Invalid log_level: {v}. Must be one of {valid_levels}"
            )
        return v.upper()


class RAGConfigSchema(BaseModel):
    """
    Complete RAG configuration schema.

    Attributes:
        global_settings: Global configuration settings
        strategies: Strategy-specific configurations
        pipeline: Pipeline configuration (optional)
    """

    global_settings: GlobalConfigSchema = Field(
        default_factory=lambda: GlobalConfigSchema()  # type: ignore[call-arg]  # pylint: disable=unnecessary-lambda
    )
    strategies: Dict[str, StrategyConfigSchema] = Field(
        default_factory=dict
    )
    pipeline: Optional[PipelineConfigSchema] = None


class ConfigManager:
    """
    Manages RAG configuration with validation and hot-reload.

    This class implements a singleton pattern to ensure a single
    configuration instance throughout the application. It supports:
    - Loading from YAML/JSON files or dictionaries
    - Schema validation using pydantic
    - Environment-specific configuration overrides
    - Hot-reload capability for development
    - Dot notation access to nested configuration

    Example:
        >>> config = ConfigManager()
        >>> config.load("config.yaml")
        >>> log_level = config.get("global_settings.log_level")
        >>> print(log_level)
        INFO
    """

    _instance: Optional["ConfigManager"] = None
    _config: Optional[RAGConfigSchema] = None
    _config_path: Optional[Path] = None
    _observers: List[Any] = []  # List of watchdog.observers.Observer instances
    _callbacks: List[Callable[[RAGConfigSchema], None]] = []

    def __new__(cls) -> "ConfigManager":
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        environment: Optional[str] = None
    ) -> None:
        """
        Load configuration from file or dictionary.

        Args:
            config_path: Path to configuration file (YAML or JSON)
            config_dict: Configuration as dictionary
            environment: Environment name for overrides

        Raises:
            ConfigurationError: If configuration is invalid or cannot be loaded

        Example:
            >>> config = ConfigManager()
            >>> config.load("config.yaml")
            >>> # Or from dict
            >>> config.load(config_dict={"global_settings": {"log_level": "DEBUG"}})
        """
        if config_path:
            self._config_path = Path(config_path)
            config_data = self._load_file(self._config_path)
        elif config_dict:
            config_data = config_dict
        else:
            config_data = self._get_default_config()

        # Apply environment-specific overrides
        env_name = environment if environment is not None else os.getenv("RAG_ENV", "development")
        config_data = self._apply_environment_overrides(config_data, env_name)

        # Validate and parse
        try:
            self._config = RAGConfigSchema(**config_data)
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration: {e}") from e

    def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load configuration from file.

        Args:
            file_path: Path to configuration file

        Returns:
            Dict: Configuration data

        Raises:
            ConfigurationError: If file not found or has invalid format
        """
        if not file_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {file_path}"
            )

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix in ['.yaml', '.yml']:
                    loaded = yaml.safe_load(f)
                    return cast(Dict[str, Any], loaded if loaded is not None else {})
                if file_path.suffix == '.json':
                    return cast(Dict[str, Any], json.load(f))
                raise ConfigurationError(
                    f"Unsupported file format: {file_path.suffix}. "
                    f"Use .yaml, .yml, or .json"
                )
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML: {e}") from e
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON: {e}") from e

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.

        Returns:
            Dict: Default configuration data
        """
        return {
            "global_settings": {
                "environment": "development",
                "log_level": "INFO",
                "cache_enabled": True,
                "cache_ttl": 3600
            },
            "strategies": {},
            "pipeline": None
        }

    def _apply_environment_overrides(
        self,
        config: Dict[str, Any],
        environment: str
    ) -> Dict[str, Any]:
        """
        Apply environment-specific configuration overrides.

        Args:
            config: Base configuration
            environment: Environment name

        Returns:
            Dict: Merged configuration
        """
        if self._config_path:
            env_config_path = (
                self._config_path.parent /
                f"{self._config_path.stem}.{environment}{self._config_path.suffix}"
            )
            if env_config_path.exists():
                env_config = self._load_file(env_config_path)
                config = self._deep_merge(config, env_config)

        return config

    def _deep_merge(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Dict: Merged dictionary
        """
        result = base.copy()
        for key, value in override.items():
            if (key in result and
                    isinstance(result[key], dict) and
                    isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Any: Configuration value

        Raises:
            ConfigurationError: If configuration not loaded

        Example:
            >>> config.get("global_settings.log_level")
            'INFO'
            >>> config.get("nonexistent.key", "default")
            'default'
        """
        if self._config is None:
            raise ConfigurationError("Configuration not loaded")

        keys = key.split('.')
        value: Any = self._config.model_dump()

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_strategy_config(self, strategy_name: str) -> StrategyConfigSchema:
        """
        Get configuration for a specific strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            StrategyConfigSchema: Strategy configuration

        Raises:
            ConfigurationError: If configuration not loaded

        Example:
            >>> config.get_strategy_config("my_strategy")
            StrategyConfigSchema(chunk_size=512, top_k=5, ...)
        """
        if self._config is None:
            raise ConfigurationError("Configuration not loaded")

        if strategy_name in self._config.strategies:
            return self._config.strategies[strategy_name]

        # Return default config
        return StrategyConfigSchema(strategy_name=strategy_name)  # type: ignore[call-arg]

    def enable_hot_reload(
        self,
        callback: Optional[Callable[[RAGConfigSchema], None]] = None
    ) -> None:
        """
        Enable hot-reloading of configuration files.

        Args:
            callback: Optional callback function called when config reloads

        Raises:
            ConfigurationError: If no config file path available
            ImportError: If watchdog is not installed

        Example:
            >>> def on_reload(new_config):
            ...     print("Configuration reloaded!")
            >>> config.enable_hot_reload(callback=on_reload)
        """
        try:
            from watchdog.observers import Observer as WatchdogObserver
        except ImportError as e:
            raise ImportError(
                "watchdog is required for hot-reload. "
                "Install it with: pip install rag-factory[watch]"
            ) from e

        if not self._config_path:
            raise ConfigurationError(
                "Cannot hot-reload without config file path"
            )

        if callback:
            self._callbacks.append(callback)

        handler = ConfigFileHandler(self)
        observer = WatchdogObserver()
        observer.schedule(
            handler,
            str(self._config_path.parent),
            recursive=False
        )
        observer.start()
        self._observers.append(observer)

    def disable_hot_reload(self) -> None:
        """
        Disable hot-reloading.

        Example:
            >>> config.disable_hot_reload()
        """
        for observer in self._observers:
            observer.stop()
            observer.join()
        self._observers.clear()

    def reload(self) -> None:
        """
        Reload configuration from file.

        Raises:
            ConfigurationError: If no config file path available

        Example:
            >>> config.reload()
        """
        if self._config_path:
            self.load(str(self._config_path))
            # Notify callbacks
            for callback in self._callbacks:
                if self._config:
                    callback(self._config)

    @property
    def config(self) -> Optional[RAGConfigSchema]:
        """
        Get the current configuration.

        Returns:
            Optional[RAGConfigSchema]: Current configuration or None
        """
        return self._config

    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.

        Returns:
            Dict: Configuration as dictionary

        Raises:
            ConfigurationError: If configuration not loaded

        Example:
            >>> config_dict = config.to_dict()
            >>> print(config_dict["global_settings"]["log_level"])
            INFO
        """
        if self._config is None:
            raise ConfigurationError("Configuration not loaded")
        return self._config.model_dump()


def _get_file_handler_base() -> type:
    """Get base class for ConfigFileHandler."""
    try:
        from watchdog.events import FileSystemEventHandler
        return FileSystemEventHandler
    except ImportError:
        # If watchdog not installed, use object as base
        return object  # type: ignore[return-value]


class ConfigFileHandler(_get_file_handler_base()):  # type: ignore[misc]
    """Handle configuration file changes for hot-reload."""

    def __init__(self, config_manager: ConfigManager) -> None:
        """
        Initialize file handler.

        Args:
            config_manager: ConfigManager instance
        """
        try:
            super().__init__()  # type: ignore[misc]
        except TypeError:
            # If watchdog not installed, super() might not work
            pass
        self.config_manager = config_manager

    def on_modified(self, event: Any) -> None:
        """
        Handle file modification event.

        Args:
            event: File system event
        """
        # pylint: disable=protected-access
        if (not event.is_directory and
                str(event.src_path) == str(self.config_manager._config_path)):
            try:
                self.config_manager.reload()
                print(f"Configuration reloaded from {event.src_path}")
            except ConfigurationError as e:
                print(f"Failed to reload configuration: {e}")
