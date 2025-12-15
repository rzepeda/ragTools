"""Unit tests for environment variable resolver."""

import pytest
import os
from rag_factory.config.env_resolver import EnvResolver, EnvironmentVariableError


class TestEnvResolverBasic:
    """Test basic environment variable resolution."""

    def test_resolve_simple_variable(self):
        """Test resolving simple environment variable."""
        os.environ["TEST_VAR"] = "test_value"
        try:
            result = EnvResolver.resolve("${TEST_VAR}")
            assert result == "test_value"
        finally:
            del os.environ["TEST_VAR"]

    def test_resolve_with_default(self):
        """Test resolving with default value."""
        # Variable doesn't exist
        result = EnvResolver.resolve("${NONEXISTENT:-default_value}")
        assert result == "default_value"

    def test_resolve_with_default_when_exists(self):
        """Test that existing variable takes precedence over default."""
        os.environ["TEST_VAR"] = "actual_value"
        try:
            result = EnvResolver.resolve("${TEST_VAR:-default_value}")
            assert result == "actual_value"
        finally:
            del os.environ["TEST_VAR"]

    def test_resolve_required_missing(self):
        """Test error for missing required variable."""
        with pytest.raises(EnvironmentVariableError) as exc_info:
            EnvResolver.resolve("${REQUIRED_VAR}")

        assert "REQUIRED_VAR" in str(exc_info.value)
        assert "not set" in str(exc_info.value)

    def test_resolve_with_custom_error(self):
        """Test custom error message."""
        with pytest.raises(EnvironmentVariableError) as exc_info:
            EnvResolver.resolve("${REQUIRED_VAR:?Custom error message}")

        assert "Custom error message" in str(exc_info.value)
        assert "REQUIRED_VAR" in str(exc_info.value)


class TestEnvResolverRecursive:
    """Test recursive resolution in complex structures."""

    def test_resolve_in_dict(self):
        """Test recursive resolution in dictionary."""
        os.environ["KEY1"] = "value1"
        os.environ["KEY2"] = "value2"

        try:
            config = {
                "field1": "${KEY1}",
                "field2": "${KEY2}",
                "field3": "normal_value",
                "nested": {
                    "field4": "${KEY1}"
                }
            }

            result = EnvResolver.resolve(config)

            assert result["field1"] == "value1"
            assert result["field2"] == "value2"
            assert result["field3"] == "normal_value"
            assert result["nested"]["field4"] == "value1"
        finally:
            del os.environ["KEY1"]
            del os.environ["KEY2"]

    def test_resolve_in_list(self):
        """Test resolution in list."""
        os.environ["ITEM"] = "item_value"

        try:
            config = ["${ITEM}", "normal", "${ITEM}"]
            result = EnvResolver.resolve(config)

            assert result == ["item_value", "normal", "item_value"]
        finally:
            del os.environ["ITEM"]

    def test_resolve_deeply_nested(self):
        """Test resolution in deeply nested structures."""
        os.environ["VAR"] = "resolved"

        try:
            config = {
                "level1": {
                    "level2": {
                        "level3": {
                            "value": "${VAR}"
                        }
                    }
                }
            }

            result = EnvResolver.resolve(config)
            assert result["level1"]["level2"]["level3"]["value"] == "resolved"
        finally:
            del os.environ["VAR"]


class TestEnvResolverPartialReplacement:
    """Test partial string replacement."""

    def test_partial_string_replacement(self):
        """Test partial string replacement."""
        os.environ["HOST"] = "localhost"
        os.environ["PORT"] = "5432"

        try:
            connection_string = "postgresql://user:pass@${HOST}:${PORT}/db"
            result = EnvResolver.resolve(connection_string)

            assert result == "postgresql://user:pass@localhost:5432/db"
        finally:
            del os.environ["HOST"]
            del os.environ["PORT"]

    def test_multiple_variables_in_string(self):
        """Test multiple variables in single string."""
        os.environ["USER"] = "admin"
        os.environ["PASS"] = "secret"
        os.environ["DB"] = "mydb"

        try:
            result = EnvResolver.resolve("${USER}:${PASS}@${DB}")
            assert result == "admin:secret@mydb"
        finally:
            del os.environ["USER"]
            del os.environ["PASS"]
            del os.environ["DB"]

    def test_mixed_variables_and_defaults(self):
        """Test mixing variables and defaults in string."""
        os.environ["HOST"] = "localhost"

        try:
            result = EnvResolver.resolve("${HOST}:${PORT:-5432}")
            assert result == "localhost:5432"
        finally:
            del os.environ["HOST"]


class TestEnvResolverValidation:
    """Test environment variable validation."""

    def test_validate_no_injection_valid(self):
        """Test validation accepts valid variable names."""
        assert EnvResolver.validate_no_injection("DB_HOST")
        assert EnvResolver.validate_no_injection("API_KEY")
        assert EnvResolver.validate_no_injection("_PRIVATE")
        assert EnvResolver.validate_no_injection("VAR123")

    def test_validate_no_injection_invalid(self):
        """Test validation rejects invalid variable names."""
        assert not EnvResolver.validate_no_injection("DB_HOST; rm -rf /")
        assert not EnvResolver.validate_no_injection("VAR && echo hack")
        assert not EnvResolver.validate_no_injection("123VAR")  # Can't start with number
        assert not EnvResolver.validate_no_injection("VAR-NAME")  # No hyphens


class TestEnvResolverExtraction:
    """Test variable name extraction."""

    def test_extract_variable_names_simple(self):
        """Test extracting variable names from simple string."""
        result = EnvResolver.extract_variable_names("${HOST}:${PORT}")
        assert result == ["HOST", "PORT"]

    def test_extract_variable_names_dict(self):
        """Test extracting variable names from dictionary."""
        config = {
            "a": "${VAR1}",
            "b": "${VAR2}",
            "c": "normal"
        }
        result = EnvResolver.extract_variable_names(config)
        assert result == ["VAR1", "VAR2"]

    def test_extract_variable_names_with_defaults(self):
        """Test extracting variable names with defaults."""
        result = EnvResolver.extract_variable_names("${HOST:-localhost}:${PORT:-5432}")
        assert result == ["HOST", "PORT"]

    def test_extract_variable_names_duplicates(self):
        """Test that duplicates are removed."""
        config = {
            "a": "${VAR}",
            "b": "${VAR}",
            "c": "${VAR}"
        }
        result = EnvResolver.extract_variable_names(config)
        assert result == ["VAR"]

    def test_extract_variable_names_nested(self):
        """Test extracting from nested structures."""
        config = {
            "db": {
                "host": "${DB_HOST}",
                "port": "${DB_PORT}"
            },
            "api": {
                "key": "${API_KEY}"
            }
        }
        result = EnvResolver.extract_variable_names(config)
        assert result == ["API_KEY", "DB_HOST", "DB_PORT"]


class TestEnvResolverEdgeCases:
    """Test edge cases and error handling."""

    def test_resolve_non_string_values(self):
        """Test that non-string values are returned unchanged."""
        assert EnvResolver.resolve(123) == 123
        assert EnvResolver.resolve(45.67) == 45.67
        assert EnvResolver.resolve(True) is True
        assert EnvResolver.resolve(None) is None

    def test_resolve_empty_string(self):
        """Test resolving empty string."""
        assert EnvResolver.resolve("") == ""

    def test_resolve_string_without_variables(self):
        """Test string without variables."""
        assert EnvResolver.resolve("normal string") == "normal string"

    def test_resolve_empty_dict(self):
        """Test resolving empty dictionary."""
        assert EnvResolver.resolve({}) == {}

    def test_resolve_empty_list(self):
        """Test resolving empty list."""
        assert EnvResolver.resolve([]) == []

    def test_resolve_mixed_types_in_list(self):
        """Test list with mixed types."""
        os.environ["VAR"] = "value"
        try:
            config = ["${VAR}", 123, True, None]
            result = EnvResolver.resolve(config)
            assert result == ["value", 123, True, None]
        finally:
            del os.environ["VAR"]
