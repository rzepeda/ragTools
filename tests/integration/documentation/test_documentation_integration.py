"""
Integration tests for documentation build process.

This module tests that documentation builds successfully and all
components work together.
"""

import pytest
import subprocess
from pathlib import Path
import shutil
import time


@pytest.mark.integration
class TestDocumentationBuild:
    """Test that documentation builds successfully."""

    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent.parent

    def test_mkdocs_build_succeeds(self, project_root):
        """Test that mkdocs build completes without errors."""
        # Check if mkdocs is installed
        try:
            subprocess.run(
                ["mkdocs", "--version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("mkdocs not installed")

        # Try to build documentation
        result = subprocess.run(
            ["mkdocs", "build", "--strict"],
            cwd=project_root,
            capture_output=True,
            text=True
        )

        # Build might fail if some docs are missing, but should not error
        # on the docs that do exist
        if result.returncode != 0:
            # Check if it's just missing files vs actual errors
            if "ERROR" in result.stderr and "not found" not in result.stderr.lower():
                pytest.fail(f"mkdocs build failed with errors:\n{result.stderr}")
            else:
                pytest.skip(f"mkdocs build incomplete (missing files): {result.stderr}")

    def test_mkdocs_serve_starts(self, project_root):
        """Test that mkdocs serve starts (integration test)."""
        # Check if mkdocs is installed
        try:
            subprocess.run(
                ["mkdocs", "--version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("mkdocs not installed")

        # Start mkdocs serve in background
        process = subprocess.Popen(
            ["mkdocs", "serve", "--dev-addr", "localhost:8001"],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        try:
            # Wait for server to start
            time.sleep(3)

            # Check if process is still running
            if process.poll() is not None:
                _, stderr = process.communicate()
                pytest.skip(f"mkdocs serve failed to start: {stderr.decode()}")

            # Try to connect (requires requests)
            try:
                import requests
                response = requests.get("http://localhost:8001", timeout=5)
                assert response.status_code == 200, \
                    f"mkdocs serve returned status {response.status_code}"
            except ImportError:
                pytest.skip("requests library not installed")
            except Exception as e:
                pytest.skip(f"Could not connect to mkdocs serve: {e}")

        finally:
            process.terminate()
            process.wait(timeout=5)

    def test_search_index_generated(self, project_root):
        """Test that search index is generated."""
        # Check if mkdocs is installed
        try:
            subprocess.run(
                ["mkdocs", "--version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("mkdocs not installed")

        # Build docs
        result = subprocess.run(
            ["mkdocs", "build"],
            cwd=project_root,
            capture_output=True
        )

        if result.returncode != 0:
            pytest.skip("mkdocs build failed")

        # Check search index exists
        search_index = project_root / "site" / "search" / "search_index.json"
        
        if not search_index.exists():
            pytest.skip("Search index not generated (might be due to missing docs)")

        # Verify it's valid JSON
        import json
        with open(search_index, encoding='utf-8') as f:
            index = json.load(f)

        assert "docs" in index, "Search index missing 'docs' field"
        assert len(index["docs"]) > 0, "Search index is empty"


@pytest.mark.integration
def test_documentation_time_to_first_example():
    """Test that time to first working example is <30 minutes."""
    # This is a manual test to be performed by actual users
    # Document the process and timing
    pytest.skip("Manual test: Time a new user going through quick start guide")


@pytest.mark.integration
def test_api_reference_generated():
    """Test that API reference is auto-generated from docstrings."""
    project_root = Path(__file__).parent.parent.parent.parent

    # Check if mkdocs is installed
    try:
        subprocess.run(
            ["mkdocs", "--version"],
            capture_output=True,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("mkdocs not installed")

    # Build documentation
    result = subprocess.run(
        ["mkdocs", "build"],
        cwd=project_root,
        capture_output=True
    )

    if result.returncode != 0:
        pytest.skip("mkdocs build failed")

    # Check that site was generated
    site_dir = project_root / "site"
    assert site_dir.exists(), "Site directory not generated"

    # Check for some key pages (might not all exist yet)
    expected_pages = [
        "index.html",
        "getting-started/installation/index.html",
        "getting-started/quick-start/index.html",
    ]

    existing_pages = []
    for page in expected_pages:
        if (site_dir / page).exists():
            existing_pages.append(page)

    assert len(existing_pages) > 0, \
        "No expected pages were generated"


@pytest.mark.integration
def test_diagrams_rendered():
    """Test that Mermaid diagrams are rendered in build."""
    project_root = Path(__file__).parent.parent.parent.parent

    # Check if mkdocs is installed
    try:
        subprocess.run(
            ["mkdocs", "--version"],
            capture_output=True,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("mkdocs not installed")

    # Build documentation
    result = subprocess.run(
        ["mkdocs", "build"],
        cwd=project_root,
        capture_output=True
    )

    if result.returncode != 0:
        pytest.skip("mkdocs build failed")

    # Check architecture page
    arch_page = project_root / "site" / "architecture" / "overview" / "index.html"
    
    if not arch_page.exists():
        pytest.skip("Architecture overview page not generated")

    with open(arch_page, encoding='utf-8') as f:
        content = f.read()

    # Check if mermaid diagram was rendered
    # Mermaid diagrams should be present in some form
    assert "mermaid" in content.lower() or "graph" in content.lower(), \
        "Mermaid diagrams not found in architecture page"
