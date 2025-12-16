"""Validation utilities for CLI inputs."""

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from rich.console import Console

from rag_factory.factory import RAGFactory

console = Console()


def validate_path_exists(path: str, must_be_file: bool = False, must_be_dir: bool = False) -> Path:
    """
    Validate that a path exists.

    Args:
        path: Path to validate
        must_be_file: If True, path must be a file
        must_be_dir: If True, path must be a directory

    Returns:
        Path: Validated path object

    Raises:
        ValueError: If path doesn't exist or doesn't meet requirements
    """
    path_obj = Path(path)

    if not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")

    if must_be_file and not path_obj.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if must_be_dir and not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    return path_obj


def validate_strategy_name(strategy_name: str) -> str:
    """
    Validate that a strategy name is registered.

    Args:
        strategy_name: Name of the strategy to validate

    Returns:
        str: The validated strategy name

    Raises:
        ValueError: If strategy is not registered
    """
    available_strategies = RAGFactory.list_strategies()

    if strategy_name not in available_strategies:
        # Try to suggest similar strategies
        suggestions = [s for s in available_strategies if strategy_name.lower() in s.lower()]

        error_msg = f"Strategy '{strategy_name}' not found."
        if suggestions:
            error_msg += "\n\nDid you mean one of these?\n"
            for suggestion in suggestions[:3]:
                error_msg += f"  - {suggestion}\n"
        else:
            error_msg += "\n\nAvailable strategies:\n"
            for strategy in available_strategies[:5]:
                error_msg += f"  - {strategy}\n"
            if len(available_strategies) > 5:
                error_msg += f"  ... and {len(available_strategies) - 5} more\n"

        raise ValueError(error_msg)

    return strategy_name


def validate_config_file(config_path: str) -> Dict[str, Any]:
    """
    Validate and load a configuration file.

    Args:
        config_path: Path to the configuration file (YAML or JSON)

    Returns:
        Dict[str, Any]: Loaded configuration dictionary

    Raises:
        ValueError: If file doesn't exist, is invalid format, or has errors
    """
    path = validate_path_exists(config_path, must_be_file=True)

    try:
        if path.suffix in [".yaml", ".yml"]:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise ValueError(
                f"Unsupported config file format: {path.suffix}. "
                f"Use .yaml, .yml, or .json"
            )

        if not isinstance(config, dict):
            raise ValueError(f"Configuration file must contain a dictionary, got {type(config)}")

        return config

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file:\n{e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file:\n{e}") from e
    except Exception as e:
        raise ValueError(f"Error loading configuration file: {e}") from e


def parse_strategy_list(strategies_str: str) -> List[str]:
    """
    Parse comma-separated strategy names.

    Args:
        strategies_str: Comma-separated strategy names

    Returns:
        List[str]: List of strategy names

    Example:
        >>> parse_strategy_list("strategy1,strategy2, strategy3")
        ['strategy1', 'strategy2', 'strategy3']
    """
    return [s.strip() for s in strategies_str.split(",") if s.strip()]
