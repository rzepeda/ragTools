"""Unit tests for package structure and imports."""

import re
import pytest


class TestImports:
    """Test package imports work correctly."""

    def test_import_main_package(self) -> None:
        """Test main package can be imported."""
        import rag_factory

        assert hasattr(rag_factory, "__version__")

    def test_import_factory(self) -> None:
        """Test RAGFactory can be imported."""
        from rag_factory import RAGFactory

        assert RAGFactory is not None

    def test_import_pipeline(self) -> None:
        """Test StrategyPipeline can be imported."""
        from rag_factory import StrategyPipeline

        assert StrategyPipeline is not None

    def test_import_config(self) -> None:
        """Test ConfigManager can be imported."""
        from rag_factory import ConfigManager

        assert ConfigManager is not None

    def test_import_base_strategy(self) -> None:
        """Test IRAGStrategy can be imported."""
        from rag_factory.strategies import IRAGStrategy

        assert IRAGStrategy is not None

    def test_import_all_exports(self) -> None:
        """Test all items in __all__ can be imported."""
        import rag_factory

        for name in rag_factory.__all__:
            assert hasattr(rag_factory, name), f"Missing export: {name}"


class TestVersion:
    """Test version information."""

    def test_version_format(self) -> None:
        """Test version follows semantic versioning."""
        from rag_factory import __version__

        # Should match X.Y.Z format
        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$"
        assert re.match(
            pattern, __version__
        ), f"Invalid version format: {__version__}"

    def test_version_accessible(self) -> None:
        """Test version accessible from package."""
        import rag_factory

        assert hasattr(rag_factory, "__version__")
        assert isinstance(rag_factory.__version__, str)


class TestPackageStructure:
    """Test package structure is correct."""

    def test_strategies_subpackage_exists(self) -> None:
        """Test strategies subpackage exists and is accessible."""
        from rag_factory import strategies

        assert hasattr(strategies, "IRAGStrategy")

    def test_no_circular_imports(self) -> None:
        """Test importing doesn't cause circular import errors."""
        try:
            from rag_factory import RAGFactory
            from rag_factory import StrategyPipeline
            from rag_factory import ConfigManager
            from rag_factory.strategies import IRAGStrategy

            # If we get here, no circular imports
            assert RAGFactory is not None
            assert StrategyPipeline is not None
            assert ConfigManager is not None
            assert IRAGStrategy is not None
        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")


class TestDependencies:
    """Test package dependencies."""

    def test_required_dependencies_installed(self) -> None:
        """Test all required dependencies are available."""
        required_packages = ["pydantic", "yaml"]

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                pytest.fail(f"Required dependency not installed: {package}")

    def test_optional_dependencies_handled(self) -> None:
        """Test package works without optional dependencies."""
        # Should not fail if optional dependencies missing
        from rag_factory import RAGFactory

        assert RAGFactory is not None
