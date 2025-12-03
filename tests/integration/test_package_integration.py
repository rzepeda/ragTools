"""Integration tests for package structure and distribution."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
class TestPackageInstallation:
    """Test package installation in clean environment."""

    def test_package_installable(self, tmp_path: Path) -> None:
        """Test package can be installed in clean environment."""
        # Create virtual environment
        venv_dir = tmp_path / "venv"
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)], check=True
        )

        # Determine pip and python executables based on OS
        if sys.platform == "win32":
            pip_executable = venv_dir / "Scripts" / "pip"
            python_executable = venv_dir / "Scripts" / "python"
        else:
            pip_executable = venv_dir / "bin" / "pip"
            python_executable = venv_dir / "bin" / "python"

        # Install package in editable mode
        project_root = Path(__file__).parent.parent.parent
        subprocess.run(
            [str(pip_executable), "install", "-e", str(project_root)],
            check=True,
        )

        # Test import in the venv
        result = subprocess.run(
            [
                str(python_executable),
                "-c",
                "import rag_factory; print(rag_factory.__version__)",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0


@pytest.mark.integration
class TestSmokeTest:
    """Smoke tests for basic package functionality."""

    def test_basic_usage_smoke_test(self) -> None:
        """Test basic usage works after import."""
        from rag_factory import RAGFactory, StrategyPipeline, ConfigManager
        from rag_factory.strategies import IRAGStrategy, Chunk, StrategyConfig

        # Can instantiate basic objects
        factory = RAGFactory()
        assert factory is not None

        pipeline = StrategyPipeline()
        assert pipeline is not None

        config = ConfigManager()
        assert config is not None

        # Verify IRAGStrategy interface is accessible
        assert IRAGStrategy is not None

        # Can create basic data structures
        chunk = Chunk("text", {}, 0.9, "doc1", "chunk1")
        assert chunk.text == "text"

        strategy_config = StrategyConfig(chunk_size=512)
        assert strategy_config.chunk_size == 512


@pytest.mark.integration
class TestFullWorkflow:
    """Test complete workflow using installed package."""

    def test_full_workflow_with_installed_package(self) -> None:
        """Test complete workflow using installed package."""
        from rag_factory import RAGFactory, StrategyPipeline
        from rag_factory.strategies import IRAGStrategy, Chunk
        from typing import List, Any, Optional

        # Define a test strategy
        class TestStrategy(IRAGStrategy):
            """Test strategy implementation."""

            def initialize(
                self, config: Optional[dict[str, Any]] = None
            ) -> None:
                """Initialize the strategy."""
                self.config = config

            def prepare_data(self, documents: List[str]) -> Any:
                """Prepare data for retrieval."""
                return {"prepared": True}

            def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
                """Retrieve chunks."""
                return [
                    Chunk(
                        f"Result {i}",
                        {},
                        0.9,
                        f"doc{i}",
                        f"chunk{i}",
                    )
                    for i in range(top_k)
                ]

            async def aretrieve(
                self, query: str, top_k: int = 5
            ) -> List[Chunk]:
                """Asynchronously retrieve chunks."""
                return self.retrieve(query, top_k)

            def process_query(
                self, query: str, context: List[Chunk]
            ) -> str:
                """Process query with context."""
                return f"Processed: {query} with {len(context)} chunks"

        # Register strategy
        factory = RAGFactory()
        factory.register_strategy("test", TestStrategy)

        # Create strategy
        strategy = factory.create_strategy("test")

        # Use strategy in pipeline
        pipeline = StrategyPipeline()
        pipeline.add_stage(strategy, "test_stage")

        result = pipeline.execute("test query", top_k=3)

        assert len(result.final_results) == 3


@pytest.mark.integration
class TestBuildAndDistribution:
    """Test package building for distribution."""

    @pytest.mark.skip(reason="Requires build package to be installed")
    def test_package_can_be_built(self) -> None:
        """Test package can be built for distribution."""
        import os

        # Build package
        result = subprocess.run(
            [sys.executable, "-m", "build"],
            capture_output=True,
            text=True,
            check=False,
        )

        # Should succeed
        assert result.returncode == 0

        # Check dist directory created
        assert os.path.exists("dist")

        # Check wheel and sdist created
        dist_files = os.listdir("dist")
        assert any(f.endswith(".whl") for f in dist_files)
        assert any(f.endswith(".tar.gz") for f in dist_files)
