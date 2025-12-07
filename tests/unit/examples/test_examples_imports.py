"""Tests for example imports."""

import pytest
from pathlib import Path
import ast
import re


class TestExamplesImports:
    """Test that examples import from rag_factory correctly."""

    @pytest.fixture
    def examples_root(self):
        """Get examples directory path."""
        return Path(__file__).parent.parent.parent.parent / "examples"

    def extract_imports(self, py_file: Path):
        """Extract import statements from Python file."""
        with open(py_file, encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                return []

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return imports

    def test_simple_example_imports_rag_factory(self, examples_root):
        """Test that simple example imports from rag_factory."""
        simple_example = examples_root / "simple" / "basic_retrieval.py"
        
        if not simple_example.exists():
            pytest.skip("Simple example not created yet")
            
        imports = self.extract_imports(simple_example)

        assert any("rag_factory" in imp for imp in imports), \
            "Simple example doesn't import from rag_factory"

    def test_medium_example_imports(self, examples_root):
        """Test that medium example has proper imports."""
        medium_example = examples_root / "medium" / "strategy_pipeline.py"
        
        if not medium_example.exists():
            pytest.skip("Medium example not created yet")

        with open(medium_example, encoding="utf-8") as f:
            content = f.read()

        # Should import from rag_factory
        assert "rag_factory" in content, \
            "Medium example doesn't import from rag_factory"
        
        # Should import yaml for configuration
        assert "yaml" in content, \
            "Medium example doesn't import yaml"

    def test_all_examples_import_from_rag_factory(self, examples_root):
        """Test that main examples import from rag_factory."""
        main_examples = [
            examples_root / "simple" / "basic_retrieval.py",
            examples_root / "medium" / "strategy_pipeline.py",
            examples_root / "advanced" / "full_system.py"
        ]

        for example in main_examples:
            if not example.exists():
                continue

            with open(example, encoding="utf-8") as f:
                content = f.read()

            assert "rag_factory" in content, \
                f"{example.name} doesn't import from rag_factory"

    def test_no_relative_imports_in_examples(self, examples_root):
        """Test that examples don't use relative imports."""
        python_files = list(examples_root.rglob("*.py"))

        relative_imports = []
        for py_file in python_files:
            if "__pycache__" in str(py_file) or "test_" in py_file.name:
                continue
                
            with open(py_file, encoding="utf-8") as f:
                content = f.read()

            # Check for relative imports (from . or from ..)
            if re.search(r'from\s+\.+\s+import', content):
                relative_imports.append(str(py_file))

        assert len(relative_imports) == 0, \
            f"Examples using relative imports:\n" + "\n".join(relative_imports)

    def test_examples_use_absolute_imports(self, examples_root):
        """Test that examples use absolute imports for rag_factory."""
        python_files = list(examples_root.rglob("*.py"))

        for py_file in python_files:
            if "__pycache__" in str(py_file) or "test_" in py_file.name:
                continue
                
            imports = self.extract_imports(py_file)
            
            # If importing from rag_factory, should be absolute
            rag_imports = [imp for imp in imports if "rag_factory" in imp]
            for imp in rag_imports:
                assert imp.startswith("rag_factory"), \
                    f"{py_file} uses non-absolute import: {imp}"

    def test_no_wildcard_imports(self, examples_root):
        """Test that examples don't use wildcard imports."""
        python_files = list(examples_root.rglob("*.py"))

        wildcard_imports = []
        for py_file in python_files:
            if "__pycache__" in str(py_file) or "test_" in py_file.name:
                continue
                
            with open(py_file, encoding="utf-8") as f:
                content = f.read()

            # Check for wildcard imports
            if re.search(r'from\s+\S+\s+import\s+\*', content):
                wildcard_imports.append(str(py_file))

        assert len(wildcard_imports) == 0, \
            f"Examples using wildcard imports:\n" + "\n".join(wildcard_imports)

    def test_imports_are_organized(self, examples_root):
        """Test that imports follow standard organization."""
        python_files = list(examples_root.rglob("*.py"))

        for py_file in python_files:
            if "__pycache__" in str(py_file) or "test_" in py_file.name:
                continue
                
            with open(py_file, encoding="utf-8") as f:
                lines = f.readlines()

            # Find import section
            import_lines = []
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    import_lines.append(i)
                elif import_lines and stripped and not stripped.startswith("#"):
                    # End of import section
                    break

            # Imports should be at the top (after docstring and comments)
            if import_lines:
                # Allow for module docstring and comments
                first_import = import_lines[0]
                assert first_import < 50, \
                    f"{py_file}: Imports should be near the top of the file"
