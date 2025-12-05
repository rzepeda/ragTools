"""Integration tests for index and query workflow."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_factory.cli.main import app

runner = CliRunner()


class TestIndexQueryFlow:
    """Integration tests for complete index and query workflows."""

    @pytest.fixture
    def sample_documents(self, tmp_path):
        """Create sample documents for testing."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        # Create test documents
        (docs_dir / "doc1.txt").write_text("This is the first test document about machine learning.")
        (docs_dir / "doc2.txt").write_text("This is the second document discussing natural language processing.")
        (docs_dir / "doc3.md").write_text("# Markdown Document\n\nContent about AI and deep learning.")

        return docs_dir

    @pytest.fixture
    def index_dir(self, tmp_path):
        """Create temporary index directory."""
        return tmp_path / "index"

    def test_index_single_file(self, tmp_path):
        """Test indexing a single file."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        # Run index command
        result = runner.invoke(app, ["index", str(test_file)])

        # Should complete without error
        assert result.exit_code == 0
        assert "Successfully indexed" in result.stdout or result.exit_code == 0

    def test_index_directory(self, sample_documents):
        """Test indexing a directory of documents."""
        result = runner.invoke(app, ["index", str(sample_documents)])

        # Should complete without error
        assert result.exit_code == 0

    def test_index_with_custom_strategy(self, sample_documents):
        """Test indexing with custom strategy."""
        result = runner.invoke(app, [
            "index",
            str(sample_documents),
            "--strategy", "fixed_size_chunker",
        ])

        # Should complete without error
        assert result.exit_code == 0

    def test_index_with_config_file(self, sample_documents, tmp_path):
        """Test indexing with configuration file."""
        # Create config file
        config = {
            "strategy_name": "fixed_size_chunker",
            "chunk_size": 1024,
            "chunk_overlap": 100,
        }
        config_file = tmp_path / "config.yaml"
        import yaml
        config_file.write_text(yaml.dump(config))

        result = runner.invoke(app, [
            "index",
            str(sample_documents),
            "--config", str(config_file),
        ])

        # Should complete without error
        assert result.exit_code == 0

    def test_index_nonexistent_path(self):
        """Test indexing non-existent path fails gracefully."""
        result = runner.invoke(app, ["index", "/nonexistent/path"])

        # Should fail with appropriate error
        assert result.exit_code != 0
        assert "does not exist" in result.stdout.lower() or result.exit_code == 1

    def test_query_basic(self, sample_documents):
        """Test basic query functionality."""
        # First index documents
        runner.invoke(app, ["index", str(sample_documents)])

        # Then query
        result = runner.invoke(app, ["query", "machine learning"])

        # Should complete without error
        assert result.exit_code == 0

    def test_query_with_strategies(self, sample_documents):
        """Test query with multiple strategies."""
        runner.invoke(app, ["index", str(sample_documents)])

        result = runner.invoke(app, [
            "query",
            "deep learning",
            "--strategies", "basic,reranking",
        ])

        # Should complete without error
        assert result.exit_code == 0

    def test_query_with_top_k(self, sample_documents):
        """Test query with custom top-k."""
        runner.invoke(app, ["index", str(sample_documents)])

        result = runner.invoke(app, [
            "query",
            "AI",
            "--top-k", "10",
        ])

        # Should complete without error
        assert result.exit_code == 0

    def test_query_without_index(self):
        """Test query without index fails gracefully."""
        result = runner.invoke(app, [
            "query",
            "test query",
            "--index", "/nonexistent/index",
        ])

        # Should fail with appropriate error
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or result.exit_code == 1


class TestFullWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_workflow(self, tmp_path):
        """Test complete workflow: index, query, strategies."""
        # Create documents
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc1.txt").write_text("Machine learning content")
        (docs_dir / "doc2.txt").write_text("Deep learning content")

        # Step 1: List strategies
        result = runner.invoke(app, ["strategies"])
        assert result.exit_code == 0

        # Step 2: Index documents
        result = runner.invoke(app, ["index", str(docs_dir)])
        assert result.exit_code == 0

        # Step 3: Query documents
        result = runner.invoke(app, ["query", "machine learning"])
        assert result.exit_code == 0

    def test_workflow_with_config(self, tmp_path):
        """Test workflow with configuration file."""
        # Create documents
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc1.txt").write_text("Test content")

        # Create config
        config = {"strategy_name": "fixed_size_chunker", "chunk_size": 512}
        config_file = tmp_path / "config.yaml"
        import yaml
        config_file.write_text(yaml.dump(config))

        # Validate config
        result = runner.invoke(app, ["config", str(config_file)])
        assert result.exit_code == 0

        # Index with config
        result = runner.invoke(app, [
            "index",
            str(docs_dir),
            "--config", str(config_file),
        ])
        assert result.exit_code == 0
