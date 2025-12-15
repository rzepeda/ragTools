import os
import yaml
import json
import logging
import importlib.resources
from typing import Dict, Any, Union
from pathlib import Path
from jsonschema import validate, ValidationError as JsonSchemaError

logger = logging.getLogger(__name__)

class StrategyLoaderError(Exception):
    """Base exception for strategy loading errors."""
    pass

class ConfigurationError(StrategyLoaderError):
    """Raised when configuration is invalid."""
    pass

class StrategyPairLoader:
    """
    Loads and validates strategy pair configurations from YAML files.
    """
    
    def __init__(self, schemas_dir: Union[str, Path, None] = None):
        """
        Initialize the loader.
        
        Args:
            schemas_dir: Directory containing JSON schemas. If None, uses default package location.
        """
        if schemas_dir:
            self.schemas_dir = Path(schemas_dir)
        else:
            # Determine default location based on package structure
            # user path: rag_factory/config/schemas
            current_file = Path(__file__)
            self.schemas_dir = current_file.parent / "schemas"
            
        self.schema_file = "strategy_pair_schema.json"
        self._schema = self._load_schema()

    def _load_schema(self) -> Dict[str, Any]:
        """Load the JSON schema for validation."""
        schema_path = self.schemas_dir / self.schema_file
        try:
            with open(schema_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback for when running in environments where relative paths might slightly differ
            # or if strictly using importlib
            try:
                from rag_factory.config import schemas
                with importlib.resources.open_text(schemas, self.schema_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Could not find schema file at {schema_path} or via importlib: {e}")
                raise ConfigurationError(f"Schema file {self.schema_file} not found.")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in schema file: {e}")

    def load_config(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and validate a strategy pair configuration file.
        
        Args:
            file_path: Path to the YAML configuration file.
            
        Returns:
            Validated configuration dictionary.
            
        Raises:
            ConfigurationError: If file not found or invalid YAML/Schema.
        """
        path = Path(file_path)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
            
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML file {file_path}: {e}")
            
        if not config:
            raise ConfigurationError(f"Empty configuration file: {file_path}")
            
        self.validate_config(config)
        return config

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration against the schema.
        
        Args:
            config: Configuration dictionary to validate.
            
        Raises:
            ConfigurationError: If validation fails.
        """
        try:
            validate(instance=config, schema=self._schema)
        except JsonSchemaError as e:
            raise ConfigurationError(f"Configuration validation failed: {e.message} at path {e.path}")

