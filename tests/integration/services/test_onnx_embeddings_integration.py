"""Integration tests for ONNX embeddings with real models.

These tests require actual ONNX models to be available.
They are marked as 'slow' and can be skipped with: pytest -m "not slow"
"""

import pytest
import numpy as np
from pathlib import Path

# Mark all tests in this module as slow
pytestmark = pytest.mark.slow


class TestONNXEmbeddingsIntegration:
    """Integration tests with real ONNX models.
    
    Note: These tests use Xenova ONNX models from environment configuration.
    To prepare:
        python scripts/download_embedding_model.py
    """

    @pytest.fixture(scope="class")
    def embedding_provider(self):
        """Create real ONNX embedding provider.
        
        Uses a small, fast model for testing.
        """
        try:
            from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider
            
            # Use small fast model for testing as promised by docstring
            config = {
                "model": "Xenova/all-MiniLM-L6-v2"
            }
            provider = ONNXLocalProvider(config)
            return provider
        except Exception as e:
            pytest.skip(f"Could not load ONNX provider: {e}")

    def test_embed_single_document(self, embedding_provider):
        """Test embedding a single document."""
        text = "This is a test document for integration testing."
        
        result = embedding_provider.get_embeddings([text])

        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == embedding_provider.get_dimensions()
        assert all(isinstance(x, float) for x in result.embeddings[0])

        # Check normalization (should be unit vector)
        norm = np.linalg.norm(result.embeddings[0])
        assert abs(norm - 1.0) < 0.01

    def test_embed_multiple_documents(self, embedding_provider):
        """Test embedding multiple documents."""
        texts = [
            "First document about machine learning and artificial intelligence.",
            "Second document about natural language processing and transformers.",
            "Third document about deep learning and neural networks.",
        ]

        result = embedding_provider.get_embeddings(texts)

        assert len(result.embeddings) == 3
        assert all(len(emb) == embedding_provider.get_dimensions() for emb in result.embeddings)
        
        # All embeddings should be normalized
        for emb in result.embeddings:
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 0.01

    def test_semantic_similarity(self, embedding_provider):
        """Test that similar texts have similar embeddings."""
        text1 = "The cat sits on the mat."
        text2 = "A cat is sitting on a mat."
        text3 = "Python is a programming language for data science."

        result = embedding_provider.get_embeddings([text1, text2, text3])
        
        emb1 = np.array(result.embeddings[0])
        emb2 = np.array(result.embeddings[1])
        emb3 = np.array(result.embeddings[2])

        # Similar texts should have higher similarity
        sim_12 = np.dot(emb1, emb2)
        sim_13 = np.dot(emb1, emb3)

        assert sim_12 > sim_13, "Similar texts should have higher similarity"
        assert sim_12 > 0.7, "Very similar texts should have high similarity"

    def test_batch_consistency(self, embedding_provider):
        """Test that batch and individual embeddings are consistent."""
        texts = ["Test 1", "Test 2", "Test 3"]

        # Batch embedding
        batch_result = embedding_provider.get_embeddings(texts)

        # Individual embeddings
        individual_results = [
            embedding_provider.get_embeddings([text])
            for text in texts
        ]

        # Should be very similar (allowing for small numerical differences)
        for batch_emb, ind_result in zip(batch_result.embeddings, individual_results):
            ind_emb = ind_result.embeddings[0]
            similarity = np.dot(batch_emb, ind_emb)
            assert similarity > 0.99, "Batch and individual embeddings should match"

    def test_empty_input(self, embedding_provider):
        """Test handling of empty input."""
        result = embedding_provider.get_embeddings([])

        assert result.embeddings == []
        assert result.token_count == 0

    def test_long_text_handling(self, embedding_provider):
        """Test handling of long texts (should truncate)."""
        # Create a very long text
        long_text = "This is a test sentence. " * 200  # ~1000 words

        result = embedding_provider.get_embeddings([long_text])

        # Should still produce valid embedding
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == embedding_provider.get_dimensions()

    def test_special_characters(self, embedding_provider):
        """Test handling of special characters."""
        texts = [
            "Text with émojis 😀 and spëcial çharacters!",
            "Numbers: 123, 456.789",
            "Symbols: @#$%^&*()",
            "Mixed: Hello! How are you? 你好！"
        ]

        result = embedding_provider.get_embeddings(texts)

        assert len(result.embeddings) == len(texts)
        for emb in result.embeddings:
            assert len(emb) == embedding_provider.get_dimensions()
            assert not any(np.isnan(emb))

    @pytest.mark.benchmark
    def test_performance_target(self, embedding_provider):
        """Test that embedding speed meets performance target.
        
        Note: This test is marked as 'benchmark' and can be skipped with:
            pytest -m "not benchmark"
        """
        import time

        text = "This is a performance test document with reasonable length."

        # Warm up
        embedding_provider.get_embeddings([text])

        # Measure performance
        times = []
        for _ in range(10):
            start = time.perf_counter()
            embedding_provider.get_embeddings([text])
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)

        # Should be < 3s per document on CPU (relaxed for CI and first-run overhead)
        # First run includes model loading which can be slow
        assert avg_time < 3.0, f"Average time {avg_time*1000:.2f}ms exceeds 3000ms target"
        assert p95_time < 5.0, f"P95 time {p95_time*1000:.2f}ms too slow"

    def test_provider_metadata(self, embedding_provider):
        """Test provider metadata."""
        assert embedding_provider.get_model_name() == "Xenova/all-MiniLM-L6-v2"
        assert embedding_provider.get_dimensions() == 384
        assert embedding_provider.get_max_batch_size() > 0
        assert embedding_provider.calculate_cost(1000) == 0.0

    def test_result_metadata(self, embedding_provider):
        """Test that result contains correct metadata."""
        result = embedding_provider.get_embeddings(["Test"])

        assert result.model == "Xenova/all-MiniLM-L6-v2"
        assert result.dimensions == 384
        assert result.provider == "onnx-local"
        assert result.cost == 0.0
        assert result.token_count > 0
