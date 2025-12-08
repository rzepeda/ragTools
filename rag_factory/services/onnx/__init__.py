"""ONNX-based service implementations.

This package provides service implementations using ONNX Runtime
for local, lightweight inference without heavy ML frameworks.
"""

from rag_factory.services.onnx.embedding import ONNXEmbeddingService

__all__ = [
    "ONNXEmbeddingService",
]
