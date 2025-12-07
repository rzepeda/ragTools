"""Tests for example file syntax and structure."""

import pytest
from pathlib import Path
import ast


class TestExamplesSyntax:
    """Test that all example files have valid syntax."""

    @pytest.fixture
    def examples_root(self):
        """Get examples directory path."""
        return Path(__file__).parent.parent.parent.parent / "examples"

    def get_python_files(self, examples_root: Path):
        """Get all Python example files."""
        return list(examples_root.rglob("*.py"))

    def test_all_examples_valid_syntax(self, examples_root):
        """Test that all example Python files have valid syntax."""
        python_files = self.get_python_files(examples_root)
        
        assert len(python_files) > 0, "No Python files found in examples directory"

        errors = []
        for py_file in python_files:
            # Skip __pycache__ and other non-example files
            if "__pycache__" in str(py_file) or "test_" in py_file.name:
                continue
                
            try:
                with open(py_file, encoding="utf-8") as f:
                    ast.parse(f.read())
            except SyntaxError as e:
                errors.append(f"{py_file}: {e}")

        assert len(errors) == 0, f"Syntax errors in examples:\n" + "\n".join(errors)

    def test_simple_example_exists(self, examples_root):
        """Test that simple example exists."""
        simple_example = examples_root / "simple" / "basic_retrieval.py"
        assert simple_example.exists(), "Simple example missing"

    def test_medium_example_exists(self, examples_root):
        """Test that medium example exists."""
        medium_example = examples_root / "medium" / "strategy_pipeline.py"
        assert medium_example.exists(), "Medium example missing"

    def test_advanced_example_exists(self, examples_root):
        """Test that advanced example exists."""
        advanced_example = examples_root / "advanced" / "full_system.py"
        assert advanced_example.exists(), "Advanced example missing"

    def test_all_examples_have_docstrings(self, examples_root):
        """Test that all examples have module docstrings."""
        python_files = self.get_python_files(examples_root)

        missing_docs = []
        for py_file in python_files:
            # Skip non-example files
            if "__pycache__" in str(py_file) or "test_" in py_file.name:
                continue
                
            with open(py_file, encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read())
                    docstring = ast.get_docstring(tree)
                    if not docstring:
                        missing_docs.append(str(py_file))
                except SyntaxError:
                    # Already caught by syntax test
                    pass

        assert len(missing_docs) == 0, \
            f"Examples missing docstrings:\n" + "\n".join(missing_docs)

    def test_all_examples_have_main_function(self, examples_root):
        """Test that runnable examples have main() function."""
        # Only check new examples in specific directories
        check_dirs = ["simple", "medium", "advanced"]
        
        python_files = []
        for dir_name in check_dirs:
            dir_path = examples_root / dir_name
            if dir_path.exists():
                python_files.extend(list(dir_path.rglob("*.py")))

        # Exclude utility files
        runnable_files = [
            f for f in python_files
            if "utils" not in str(f) 
            and "models" not in str(f)
            and "__pycache__" not in str(f)
            and "test_" not in f.name
        ]

        missing_main = []
        for py_file in runnable_files:
            with open(py_file, encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read())
                    has_main = any(
                        isinstance(node, ast.FunctionDef) and node.name == "main"
                        for node in ast.walk(tree)
                    )

                    if not has_main:
                        missing_main.append(str(py_file))
                except SyntaxError:
                    # Already caught by syntax test
                    pass

        assert len(missing_main) == 0, \
            f"Examples missing main() function:\n" + "\n".join(missing_main)

    def test_examples_have_proper_encoding(self, examples_root):
        """Test that all examples can be read with UTF-8 encoding."""
        python_files = self.get_python_files(examples_root)

        encoding_errors = []
        for py_file in python_files:
            if "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, encoding="utf-8") as f:
                    f.read()
            except UnicodeDecodeError as e:
                encoding_errors.append(f"{py_file}: {e}")

        assert len(encoding_errors) == 0, \
            f"Encoding errors in examples:\n" + "\n".join(encoding_errors)

    def test_examples_not_too_long(self, examples_root):
        """Test that examples are reasonable length."""
        python_files = self.get_python_files(examples_root)

        too_long = []
        max_lines = 500  # Reasonable limit for examples

        for py_file in python_files:
            if "__pycache__" in str(py_file) or "test_" in py_file.name:
                continue
                
            with open(py_file, encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) > max_lines:
                    too_long.append(f"{py_file}: {len(lines)} lines")

        assert len(too_long) == 0, \
            f"Examples too long (>{max_lines} lines):\n" + "\n".join(too_long)
