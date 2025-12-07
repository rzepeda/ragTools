"""Tests for example requirements files."""

import pytest
from pathlib import Path


class TestRequirements:
    """Test that each example has requirements.txt."""

    @pytest.fixture
    def examples_root(self):
        """Get examples directory path."""
        return Path(__file__).parent.parent.parent.parent / "examples"

    def test_simple_has_requirements(self, examples_root):
        """Test that simple example has requirements.txt."""
        req_file = examples_root / "simple" / "requirements.txt"
        assert req_file.exists(), "Simple example missing requirements.txt"

    def test_medium_has_requirements(self, examples_root):
        """Test that medium example has requirements.txt."""
        req_file = examples_root / "medium" / "requirements.txt"
        assert req_file.exists(), "Medium example missing requirements.txt"

    def test_advanced_has_requirements(self, examples_root):
        """Test that advanced example has requirements.txt."""
        req_file = examples_root / "advanced" / "requirements.txt"
        assert req_file.exists(), "Advanced example missing requirements.txt"

    def test_all_example_dirs_have_requirements(self, examples_root):
        """Test that all main example directories have requirements.txt."""
        # Main example directories that should have requirements
        required_dirs = ["simple", "medium", "advanced"]
        
        missing_requirements = []
        for dir_name in required_dirs:
            example_dir = examples_root / dir_name
            if not example_dir.exists():
                continue
                
            req_file = example_dir / "requirements.txt"
            if not req_file.exists():
                missing_requirements.append(dir_name)

        assert len(missing_requirements) == 0, \
            f"Directories missing requirements.txt: {missing_requirements}"

    def test_requirements_have_rag_factory(self, examples_root):
        """Test that requirements include rag-factory."""
        req_files = list(examples_root.rglob("requirements.txt"))
        
        # Filter out non-example requirements
        example_req_files = [
            f for f in req_files 
            if "simple" in str(f) or "medium" in str(f) or "advanced" in str(f)
        ]

        for req_file in example_req_files:
            with open(req_file, encoding="utf-8") as f:
                content = f.read()

            assert "rag-factory" in content or "rag_factory" in content, \
                f"{req_file} missing rag-factory dependency"

    def test_requirements_are_valid_format(self, examples_root):
        """Test that requirements files have valid format."""
        req_files = list(examples_root.rglob("requirements.txt"))

        for req_file in req_files:
            with open(req_file, encoding="utf-8") as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Basic validation: should have package name
                assert len(line) > 0, \
                    f"{req_file}:{i} - Empty requirement line"
                
                # Should not have spaces around operators
                if "==" in line or ">=" in line or "<=" in line:
                    parts = line.split("==") if "==" in line else \
                           line.split(">=") if ">=" in line else \
                           line.split("<=")
                    
                    for part in parts:
                        assert part == part.strip(), \
                            f"{req_file}:{i} - Whitespace around operator"

    def test_no_duplicate_requirements(self, examples_root):
        """Test that requirements files don't have duplicate packages."""
        req_files = list(examples_root.rglob("requirements.txt"))

        for req_file in req_files:
            with open(req_file, encoding="utf-8") as f:
                lines = f.readlines()

            packages = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Extract package name (before ==, >=, etc.)
                    package = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
                    packages.append(package)

            duplicates = [p for p in packages if packages.count(p) > 1]
            assert len(set(duplicates)) == 0, \
                f"{req_file} has duplicate packages: {set(duplicates)}"

    def test_requirements_have_versions(self, examples_root):
        """Test that requirements specify versions or constraints."""
        req_files = list(examples_root.rglob("requirements.txt"))

        for req_file in req_files:
            with open(req_file, encoding="utf-8") as f:
                lines = f.readlines()

            unversioned = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Check if it has version specifier
                    if not any(op in line for op in ["==", ">=", "<=", ">", "<", "~="]):
                        # Allow rag-factory without version (local package)
                        if "rag-factory" not in line and "rag_factory" not in line:
                            unversioned.append(line)

            # It's okay to have some unversioned (like local packages)
            # but warn if there are many
            assert len(unversioned) < 3, \
                f"{req_file} has many unversioned requirements: {unversioned}"
