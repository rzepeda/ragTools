"""Environment variable resolution for configuration files."""

import os
import re
from typing import Any, Dict, List, Union


class EnvironmentVariableError(Exception):
    """Error resolving environment variable."""
    pass


class EnvResolver:
    """Resolves environment variables in configuration."""

    # Pattern: ${VAR}, ${VAR:-default}, ${VAR:?error}
    ENV_VAR_PATTERN = re.compile(r'\$\{([^}:]+)(?::-([^}]+)|:\?([^}]+))?\}')

    @classmethod
    def resolve(cls, value: Any) -> Any:
        """
        Recursively resolve environment variables in value.

        Supports the following syntax:
        - ${VAR}: Required variable, raises error if not set
        - ${VAR:-default}: Optional variable with default value
        - ${VAR:?error message}: Required variable with custom error message

        Args:
            value: Value to resolve (can be str, dict, list, etc.)

        Returns:
            Value with environment variables resolved

        Raises:
            EnvironmentVariableError: If required variable is missing

        Examples:
            >>> os.environ['DB_HOST'] = 'localhost'
            >>> EnvResolver.resolve('${DB_HOST}')
            'localhost'
            >>> EnvResolver.resolve('${MISSING:-default}')
            'default'
            >>> EnvResolver.resolve({'host': '${DB_HOST}', 'port': 5432})
            {'host': 'localhost', 'port': 5432}
        """
        if isinstance(value, str):
            return cls._resolve_string(value)
        elif isinstance(value, dict):
            return {k: cls.resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [cls.resolve(item) for item in value]
        else:
            return value

    @classmethod
    def _resolve_string(cls, value: str) -> str:
        """
        Resolve environment variables in a string.

        Args:
            value: String potentially containing environment variable references

        Returns:
            String with all environment variables resolved

        Raises:
            EnvironmentVariableError: If required variable is missing
        """
        def replacer(match):
            var_name = match.group(1)
            default = match.group(2)
            error_msg = match.group(3)

            env_value = os.getenv(var_name)

            if env_value is not None:
                return env_value
            elif default is not None:
                return default
            elif error_msg is not None:
                raise EnvironmentVariableError(
                    f"Environment variable ${{{var_name}}}: {error_msg}"
                )
            else:
                raise EnvironmentVariableError(
                    f"Required environment variable ${{{var_name}}} is not set"
                )

        return cls.ENV_VAR_PATTERN.sub(replacer, value)

    @classmethod
    def validate_no_injection(cls, value: str) -> bool:
        """
        Validate that environment variable name doesn't contain injection attempts.

        Args:
            value: Environment variable name to validate

        Returns:
            True if valid, False otherwise

        Examples:
            >>> EnvResolver.validate_no_injection('DB_HOST')
            True
            >>> EnvResolver.validate_no_injection('DB_HOST; rm -rf /')
            False
        """
        # Only allow alphanumeric characters and underscores
        return bool(re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', value))

    @classmethod
    def extract_variable_names(cls, value: Any) -> List[str]:
        """
        Extract all environment variable names from a value.

        Args:
            value: Value to extract variable names from

        Returns:
            List of unique environment variable names

        Examples:
            >>> EnvResolver.extract_variable_names('${HOST}:${PORT}')
            ['HOST', 'PORT']
            >>> EnvResolver.extract_variable_names({'a': '${VAR1}', 'b': '${VAR2}'})
            ['VAR1', 'VAR2']
        """
        variables = set()

        def extract_from_string(s: str):
            for match in cls.ENV_VAR_PATTERN.finditer(s):
                var_name = match.group(1)
                variables.add(var_name)

        def extract_recursive(val: Any):
            if isinstance(val, str):
                extract_from_string(val)
            elif isinstance(val, dict):
                for v in val.values():
                    extract_recursive(v)
            elif isinstance(val, list):
                for item in val:
                    extract_recursive(item)

        extract_recursive(value)
        return sorted(list(variables))
