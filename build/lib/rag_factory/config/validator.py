"""Configuration validation for service registry and strategy pairs."""

from typing import Dict, Any, List, Optional
import yaml
import jsonschema
import json
from pathlib import Path


class ConfigValidationError(Exception):
    """Configuration validation error with detailed context."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        field: Optional[str] = None
    ):
        """
        Initialize validation error.

        Args:
            message: Error message
            file_path: Path to configuration file
            field: Field path that caused the error
        """
        self.message = message
        self.file_path = file_path
        self.field = field
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with context."""
        parts = [self.message]
        if self.file_path:
            parts.append(f"File: {self.file_path}")
        if self.field:
            parts.append(f"Field: {self.field}")
        return "\n".join(parts)


class ConfigValidator:
    """Validates service registry and strategy pair configurations."""

    def __init__(self, schemas_dir: Optional[str] = None):
        """
        Initialize validator with JSON schemas.

        Args:
            schemas_dir: Directory containing JSON schema files.
                        Defaults to rag_factory/config/schemas/
        """
        if schemas_dir is None:
            schemas_dir = Path(__file__).parent / "schemas"

        self.schemas_dir = Path(schemas_dir)
        self._schemas: Dict[str, Dict] = {}
        self._load_schemas()

    def _load_schemas(self):
        """Load JSON schemas from files."""
        schema_files = {
            "service_registry": "service_registry_schema.json",
            "strategy_pair": "strategy_pair_schema.json"
        }

        for name, filename in schema_files.items():
            schema_path = self.schemas_dir / filename
            if not schema_path.exists():
                raise FileNotFoundError(
                    f"Schema file not found: {schema_path}"
                )
            with open(schema_path, 'r') as f:
                self._schemas[name] = json.load(f)

    def validate_services_yaml(
        self,
        config: Dict[str, Any],
        file_path: Optional[str] = None
    ) -> List[str]:
        """
        Validate services.yaml configuration.

        Args:
            config: Parsed YAML configuration
            file_path: Path to YAML file (for error messages)

        Returns:
            List of warning messages (empty if no warnings)

        Raises:
            ConfigValidationError: If validation fails

        Examples:
            >>> validator = ConfigValidator()
            >>> config = {
            ...     "services": {
            ...         "llm1": {
            ...             "name": "test-llm",
            ...             "type": "llm",
            ...             "url": "http://localhost:1234/v1",
            ...             "api_key": "${API_KEY}",
            ...             "model": "test-model"
            ...         }
            ...     }
            ... }
            >>> warnings = validator.validate_services_yaml(config)
            >>> len(warnings)
            0
        """
        try:
            # JSON Schema validation
            jsonschema.validate(
                instance=config,
                schema=self._schemas["service_registry"]
            )
        except jsonschema.ValidationError as e:
            raise ConfigValidationError(
                message=f"Schema validation failed: {e.message}",
                file_path=file_path,
                field=".".join(str(p) for p in e.path)
            )

        warnings = []

        # Check for plaintext secrets
        warnings.extend(self._check_for_plaintext_secrets(config, file_path))

        # Check for deprecated configurations
        warnings.extend(self._check_deprecated_configs(config))

        return warnings

    def validate_strategy_pair_yaml(
        self,
        config: Dict[str, Any],
        service_registry: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None
    ) -> List[str]:
        """
        Validate strategy-pair.yaml configuration.

        Args:
            config: Parsed YAML configuration
            service_registry: Service registry for reference validation
            file_path: Path to YAML file (for error messages)

        Returns:
            List of warning messages

        Raises:
            ConfigValidationError: If validation fails

        Examples:
            >>> validator = ConfigValidator()
            >>> config = {
            ...     "strategy_name": "test-pair",
            ...     "version": "1.0.0",
            ...     "indexer": {
            ...         "strategy": "TestIndexer",
            ...         "services": {"embedding": "$embedding1"}
            ...     },
            ...     "retriever": {
            ...         "strategy": "TestRetriever",
            ...         "services": {"embedding": "$embedding1"}
            ...     }
            ... }
            >>> service_registry = {
            ...     "services": {
            ...         "embedding1": {"name": "test", "type": "embedding", "provider": "onnx"}
            ...     }
            ... }
            >>> warnings = validator.validate_strategy_pair_yaml(config, service_registry)
            >>> len(warnings)
            0
        """
        try:
            # JSON Schema validation
            jsonschema.validate(
                instance=config,
                schema=self._schemas["strategy_pair"]
            )
        except jsonschema.ValidationError as e:
            raise ConfigValidationError(
                message=f"Schema validation failed: {e.message}",
                file_path=file_path,
                field=".".join(str(p) for p in e.path)
            )

        warnings = []

        # Validate service references if registry provided
        if service_registry:
            warnings.extend(
                self._validate_service_references(config, service_registry, file_path)
            )

        return warnings

    def _check_for_plaintext_secrets(
        self,
        config: Dict[str, Any],
        file_path: Optional[str]
    ) -> List[str]:
        """
        Check for potential plaintext secrets in configuration.

        Args:
            config: Configuration dictionary
            file_path: Path to configuration file

        Returns:
            List of warning messages
        """
        warnings = []
        secret_fields = ["api_key", "password", "secret", "token"]

        def check_dict(d: dict, path: str = ""):
            for key, value in d.items():
                current_path = f"{path}.{key}" if path else key

                if isinstance(value, dict):
                    check_dict(value, current_path)
                elif isinstance(value, str):
                    # Check if this is a secret field
                    if any(field in key.lower() for field in secret_fields):
                        # Check if it uses environment variable syntax
                        if not value.startswith("${"):
                            warnings.append(
                                f"WARNING: Potential plaintext secret in {current_path}. "
                                f"Consider using environment variable: ${{ENV_VAR}}"
                            )

        check_dict(config)
        return warnings

    def _check_deprecated_configs(self, config: Dict[str, Any]) -> List[str]:
        """
        Check for deprecated configuration options.

        Args:
            config: Configuration dictionary

        Returns:
            List of warning messages
        """
        warnings = []
        # Placeholder for future deprecations
        return warnings

    def _validate_service_references(
        self,
        config: Dict[str, Any],
        service_registry: Dict[str, Any],
        file_path: Optional[str]
    ) -> List[str]:
        """
        Validate that service references exist in registry.

        Args:
            config: Strategy pair configuration
            service_registry: Service registry configuration
            file_path: Path to configuration file

        Returns:
            List of warnings

        Raises:
            ConfigValidationError: If referenced service doesn't exist
        """
        warnings = []
        available_services = set(service_registry.get("services", {}).keys())

        # Check indexer and retriever services
        for component in ["indexer", "retriever"]:
            if component not in config:
                continue

            services = config[component].get("services", {})
            for service_type, service_ref in services.items():
                # Check if it's a reference (starts with $)
                if isinstance(service_ref, str) and service_ref.startswith("$"):
                    service_name = service_ref[1:]  # Remove $

                    if service_name not in available_services:
                        raise ConfigValidationError(
                            message=(
                                f"Service reference '${service_name}' not found in registry. "
                                f"Available services: {sorted(available_services)}"
                            ),
                            file_path=file_path,
                            field=f"{component}.services.{service_type}"
                        )

        return warnings


def load_yaml_with_validation(
    file_path: str,
    config_type: str,
    service_registry: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load and validate YAML configuration file.

    Args:
        file_path: Path to YAML file
        config_type: "service_registry" or "strategy_pair"
        service_registry: Service registry for strategy pair validation

    Returns:
        Validated configuration dictionary

    Raises:
        ConfigValidationError: If validation fails
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML parsing fails

    Examples:
        >>> # Load service registry
        >>> config = load_yaml_with_validation(
        ...     "config/services.yaml",
        ...     config_type="service_registry"
        ... )
        >>> # Load strategy pair
        >>> strategy = load_yaml_with_validation(
        ...     "strategies/semantic-pair.yaml",
        ...     config_type="strategy_pair",
        ...     service_registry=config
        ... )
    """
    # Load YAML
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate
    validator = ConfigValidator()

    if config_type == "service_registry":
        warnings = validator.validate_services_yaml(config, file_path)
    elif config_type == "strategy_pair":
        warnings = validator.validate_strategy_pair_yaml(
            config,
            service_registry=service_registry,
            file_path=file_path
        )
    else:
        raise ValueError(f"Unknown config type: {config_type}")

    # Print warnings
    for warning in warnings:
        print(f"WARNING: {warning}")

    return config
