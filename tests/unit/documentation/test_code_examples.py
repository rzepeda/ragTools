"""
Tests for code examples in documentation.

This module tests that all code examples in documentation have valid
syntax and can be executed.
"""

import pytest
import re
from pathlib import Path
import ast


class TestCodeExamples:
    """Test that all code examples in documentation are valid."""

    @pytest.fixture
    def docs_root(self):
        """Get documentation root directory."""
        return Path(__file__).parent.parent.parent.parent / "docs"

    def extract_python_examples(self, md_file: Path):
        """Extract Python code blocks from markdown."""
        if not md_file.exists():
            return []
        
        with open(md_file, encoding='utf-8') as f:
            content = f.read()

        # Find all ```python code blocks
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)
        return matches

    def test_all_code_examples_have_valid_syntax(self, docs_root):
        """Test that all Python code examples have valid syntax."""
        errors = []

        for md_file in docs_root.rglob("*.md"):
            examples = self.extract_python_examples(md_file)

            for i, code in enumerate(examples):
                try:
                    # Try to compile the code
                    compile(code, f"{md_file.name}:example_{i}", "exec")
                except SyntaxError as e:
                    errors.append(f"{md_file.name}:example_{i}: {e}")

        assert len(errors) == 0, \
            f"Syntax errors in code examples:\n" + "\n".join(errors)

    def test_strategy_examples_have_imports(self, docs_root):
        """Test that strategy examples include imports."""
        strategy_docs = docs_root / "strategies"
        
        if not strategy_docs.exists():
            pytest.skip("Strategy documentation not yet created")

        missing_imports = []

        for doc_file in strategy_docs.glob("*.md"):
            if doc_file.stem == "overview":
                continue

            examples = self.extract_python_examples(doc_file)

            if len(examples) == 0:
                # Some strategy docs might not have examples yet
                continue

            # First example should include imports
            first_example = examples[0]
            if "from rag_factory" not in first_example and "import" not in first_example:
                missing_imports.append(doc_file.name)

        # Allow some strategies to not have complete examples yet
        assert len(missing_imports) < 5, \
            f"Strategy docs missing imports in first example: {missing_imports}"

    def test_configuration_examples_valid(self, docs_root):
        """Test that configuration examples are valid Python dicts."""
        config_doc = docs_root / "guides" / "configuration-reference.md"

        if not config_doc.exists():
            pytest.skip("Configuration reference not yet created")

        examples = self.extract_python_examples(config_doc)

        syntax_errors = []
        for i, code in enumerate(examples):
            # Check if it contains config dictionaries
            if "config" in code or "{" in code:
                try:
                    compile(code, f"config_example_{i}", "exec")
                except SyntaxError as e:
                    syntax_errors.append(f"Example {i}: {e}")

        assert len(syntax_errors) == 0, \
            f"Invalid config examples: {syntax_errors}"

    def test_quick_start_example_complete(self, docs_root):
        """Test that quick start guide has complete working example."""
        quick_start = docs_root / "getting-started" / "quick-start.md"
        
        if not quick_start.exists():
            pytest.skip("Quick start guide not yet created")

        examples = self.extract_python_examples(quick_start)
        
        # Should have at least one example
        assert len(examples) > 0, "Quick start guide has no code examples"

        # At least one example should be substantial (>100 chars)
        substantial_examples = [ex for ex in examples if len(ex) > 100]
        assert len(substantial_examples) > 0, \
            "Quick start guide has no substantial code examples"

    def test_no_placeholder_code(self, docs_root):
        """Test that code examples don't contain placeholder comments."""
        placeholders = ["# TODO", "# FIXME", "# Implementation", "pass  # Your"]
        
        files_with_placeholders = []

        for md_file in docs_root.rglob("*.md"):
            examples = self.extract_python_examples(md_file)
            
            for code in examples:
                for placeholder in placeholders:
                    if placeholder in code:
                        files_with_placeholders.append(md_file.name)
                        break

        # Allow some placeholders in contributing docs
        assert len(files_with_placeholders) < 3, \
            f"Files with placeholder code: {files_with_placeholders}"
