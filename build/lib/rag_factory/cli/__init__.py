"""RAG Factory CLI - Development tool for testing strategies.

This module provides a command-line interface for interacting with the
RAG Factory library. It enables developers to:
- Index documents with various strategies
- Query indexed documents
- List available strategies
- Validate configurations
- Run benchmarks
- Interactive REPL mode
"""

from rag_factory.cli.main import app

__all__ = ["app"]
