"""Unit tests for validation utilities."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from rag_factory.cli.utils.validation import (
    parse_strategy_list,
    validate_config_file,
    validate_path_exists,
)


class TestValidatePathExists:
    """Tests for path validation."""

    def test_validate_existing_file(self, tmp_path):
        """Test validation of existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = validate_path_exists(str(test_file), must_be_file=True)
        assert result == test_file

    def test_validate_existing_directory(self, tmp_path):
        """Test validation of existing directory."""
        result = validate_path_exists(str(tmp_path), must_be_dir=True)
        assert result == tmp_path

    def test_validate_nonexistent_path(self):
        """Test validation of non-existent path raises error."""
        with pytest.raises(ValueError, match="Path does not exist"):
            validate_path_exists("/nonexistent/path")

    def test_validate_file_when_directory(self, tmp_path):
        """Test validation fails when expecting file but got directory."""
        with pytest.raises(ValueError, match="Path is not a file"):
            validate_path_exists(str(tmp_path), must_be_file=True)

    def test_validate_directory_when_file(self, tmp_path):
        """Test validation fails when expecting directory but got file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with pytest.raises(ValueError, match="Path is not a directory"):
            validate_path_exists(str(test_file), must_be_dir=True)


class TestValidateConfigFile:
    """Tests for configuration file validation."""

    def test_validate_yaml_config(self, tmp_path):
        """Test validation of valid YAML config."""
        config_file = tmp_path / "config.yaml"
        config_data = {"strategy_name": "test", "chunk_size": 512}
        config_file.write_text(yaml.dump(config_data))

        result = validate_config_file(str(config_file))
        assert result == config_data

    def test_validate_json_config(self, tmp_path):
        """Test validation of valid JSON config."""
        config_file = tmp_path / "config.json"
        config_data = {"strategy_name": "test", "chunk_size": 512}
        config_file.write_text(json.dumps(config_data))

        result = validate_config_file(str(config_file))
        assert result == config_data

    def test_validate_invalid_yaml(self, tmp_path):
        """Test validation of invalid YAML raises error."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content:")

        with pytest.raises(ValueError, match="Invalid YAML"):
            validate_config_file(str(config_file))

    def test_validate_invalid_json(self, tmp_path):
        """Test validation of invalid JSON raises error."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{invalid json}")

        with pytest.raises(ValueError, match="Invalid JSON"):
            validate_config_file(str(config_file))

    def test_validate_unsupported_format(self, tmp_path):
        """Test validation of unsupported file format raises error."""
        config_file = tmp_path / "config.txt"
        config_file.write_text("content")

        with pytest.raises(ValueError, match="Unsupported config file format"):
            validate_config_file(str(config_file))

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file raises error."""
        with pytest.raises(ValueError, match="Path does not exist"):
            validate_config_file("/nonexistent/config.yaml")

    def test_validate_non_dict_config(self, tmp_path):
        """Test validation fails for non-dictionary config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("- item1\n- item2")

        with pytest.raises(ValueError, match="must contain a dictionary"):
            validate_config_file(str(config_file))


class TestParseStrategyList:
    """Tests for strategy list parsing."""

    def test_parse_single_strategy(self):
        """Test parsing single strategy."""
        result = parse_strategy_list("strategy1")
        assert result == ["strategy1"]

    def test_parse_multiple_strategies(self):
        """Test parsing multiple comma-separated strategies."""
        result = parse_strategy_list("strategy1,strategy2,strategy3")
        assert result == ["strategy1", "strategy2", "strategy3"]

    def test_parse_strategies_with_spaces(self):
        """Test parsing strategies with spaces around commas."""
        result = parse_strategy_list("strategy1, strategy2 , strategy3")
        assert result == ["strategy1", "strategy2", "strategy3"]

    def test_parse_empty_string(self):
        """Test parsing empty string returns empty list."""
        result = parse_strategy_list("")
        assert result == []

    def test_parse_only_commas(self):
        """Test parsing string with only commas returns empty list."""
        result = parse_strategy_list(",,,")
        assert result == []
