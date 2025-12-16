"""Epic 17 Configuration Schema and Validation System.

This package provides the NEW configuration system for Epic 17:
- JSON Schema validation for service registry and strategy pair configurations
- Environment variable resolution with ${VAR}, ${VAR:-default}, ${VAR:?error} syntax
- Service reference validation
- Plaintext secret detection

NOTE: The legacy ConfigManager is in rag_factory.config (the .py file, not this package).
      Import it as: from rag_factory import config; config.ConfigManager()

Example usage:
    >>> from rag_factory.config.validator import load_yaml_with_validation
    >>> from rag_factory.config.env_resolver import EnvResolver
    >>> 
    >>> # Load and validate service registry
    >>> services = load_yaml_with_validation(
    ...     "config/services.yaml",
    ...     config_type="service_registry"
    ... )
    >>> 
    >>> # Resolve environment variables
    >>> services = EnvResolver.resolve(services)
    >>> 
    >>> # Load strategy pair with service reference validation
    >>> strategy = load_yaml_with_validation(
    ...     "strategies/semantic-pair.yaml",
    ...     config_type="strategy_pair",
    ...     service_registry=services
    ... )
    >>> strategy = EnvResolver.resolve(strategy)
"""

from .validator import (
    ConfigValidator,
    ConfigValidationError,
    load_yaml_with_validation,
)
from .env_resolver import (
    EnvResolver,
    EnvironmentVariableError,
)
from .schemas import (
    SERVICE_REGISTRY_VERSION,
    STRATEGY_PAIR_VERSION,
    VERSION_HISTORY,
    is_compatible,
)

__all__ = [
    # Validator
    "ConfigValidator",
    "ConfigValidationError",
    "load_yaml_with_validation",
    # Environment resolver
    "EnvResolver",
    "EnvironmentVariableError",
    # Schema versions
    "SERVICE_REGISTRY_VERSION",
    "STRATEGY_PAIR_VERSION",
    "VERSION_HISTORY",
    "is_compatible",
]
