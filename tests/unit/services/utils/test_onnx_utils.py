"""Unit tests for ONNX utilities."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from rag_factory.services.utils.onnx_utils import (
    mean_pooling,
    normalize_embeddings,
    cosine_similarity,
    get_model_metadata,
)


class TestMeanPooling:
    """Test mean pooling function."""

    def test_basic_pooling(self):
        """Test basic mean pooling."""
        # Create token embeddings: [batch=2, seq_len=3, dim=2]
        embeddings = np.array([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # Batch 1
            [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]   # Batch 2
        ], dtype=np.float32)
        
        # Attention mask: all tokens valid
        attention_mask = np.array([
            [1, 1, 1],
            [1, 1, 1]
        ], dtype=np.int64)

        result = mean_pooling(embeddings, attention_mask)

        # Expected: mean of all tokens
        expected = np.array([
            [3.0, 4.0],  # (1+3+5)/3, (2+4+6)/3
            [4.0, 5.0]   # (2+4+6)/3, (3+5+7)/3
        ], dtype=np.float32)

        np.testing.assert_array_almost_equal(result, expected)

    def test_pooling_with_padding(self):
        """Test mean pooling with padding tokens."""
        embeddings = np.array([
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],  # Last token is padding
            [[2.0, 3.0], [0.0, 0.0], [0.0, 0.0]]   # Last two tokens are padding
        ], dtype=np.float32)
        
        attention_mask = np.array([
            [1, 1, 0],  # First two tokens valid
            [1, 0, 0]   # Only first token valid
        ], dtype=np.int64)

        result = mean_pooling(embeddings, attention_mask)

        # Expected: mean of only valid tokens
        expected = np.array([
            [2.0, 3.0],  # (1+3)/2, (2+4)/2
            [2.0, 3.0]   # 2/1, 3/1
        ], dtype=np.float32)

        np.testing.assert_array_almost_equal(result, expected)

    def test_pooling_single_token(self):
        """Test pooling with single valid token."""
        embeddings = np.array([
            [[5.0, 10.0], [0.0, 0.0], [0.0, 0.0]]
        ], dtype=np.float32)
        
        attention_mask = np.array([[1, 0, 0]], dtype=np.int64)

        result = mean_pooling(embeddings, attention_mask)

        expected = np.array([[5.0, 10.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)


class TestNormalizeEmbeddings:
    """Test embedding normalization."""

    def test_normalize_2d_array(self):
        """Test normalizing 2D array."""
        embeddings = np.array([
            [3.0, 4.0],      # Length 5
            [1.0, 0.0],      # Length 1
            [0.0, 1.0]       # Length 1
        ], dtype=np.float32)

        result = normalize_embeddings(embeddings)

        # Check all vectors have unit length
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0, 1.0])

        # Check values
        expected = np.array([
            [0.6, 0.8],
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_1d_array(self):
        """Test normalizing 1D array."""
        embedding = np.array([3.0, 4.0], dtype=np.float32)

        result = normalize_embeddings(embedding)

        # Check unit length
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-6

        # Check values
        expected = np.array([0.6, 0.8], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_zero_vector(self):
        """Test normalizing zero vector (should not crash)."""
        embeddings = np.array([[0.0, 0.0]], dtype=np.float32)

        result = normalize_embeddings(embeddings)

        # Should handle gracefully (clipping prevents division by zero)
        assert result.shape == embeddings.shape
        assert not np.any(np.isnan(result))

    def test_normalize_already_normalized(self):
        """Test normalizing already normalized vectors."""
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.707, 0.707]
        ], dtype=np.float32)

        result = normalize_embeddings(embeddings)

        # Should remain approximately the same
        np.testing.assert_array_almost_equal(result, embeddings, decimal=2)


class TestCosineSimilarity:
    """Test cosine similarity calculation."""

    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])

        similarity = cosine_similarity(a, b)

        assert abs(similarity - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])

        similarity = cosine_similarity(a, b)

        assert abs(similarity - 0.0) < 1e-6

    def test_opposite_vectors(self):
        """Test similarity of opposite vectors."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])

        similarity = cosine_similarity(a, b)

        assert abs(similarity - (-1.0)) < 1e-6

    def test_similar_vectors(self):
        """Test similarity of similar but not identical vectors."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 2.9])

        similarity = cosine_similarity(a, b)

        assert 0.99 < similarity < 1.0

    def test_zero_vector(self):
        """Test similarity with zero vector."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.0, 0.0, 0.0])

        similarity = cosine_similarity(a, b)

        assert similarity == 0.0

    def test_2d_arrays(self):
        """Test that function handles 2D arrays (flattens them)."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[1.0, 2.0], [3.0, 4.0]])

        similarity = cosine_similarity(a, b)

        assert abs(similarity - 1.0) < 1e-6


class TestGetModelMetadata:
    """Test model metadata extraction."""

    def test_metadata_extraction(self):
        """Test extracting metadata from ONNX session."""
        # Mock session
        session = Mock()
        
        # Mock inputs
        input1 = Mock()
        input1.name = "input_ids"
        input1.shape = [1, 512]
        
        input2 = Mock()
        input2.name = "attention_mask"
        input2.shape = [1, 512]
        
        session.get_inputs.return_value = [input1, input2]
        
        # Mock outputs
        output = Mock()
        output.name = "last_hidden_state"
        output.shape = [1, 512, 384]
        
        session.get_outputs.return_value = [output]

        metadata = get_model_metadata(session)

        assert metadata["input_names"] == ["input_ids", "attention_mask"]
        assert metadata["output_names"] == ["last_hidden_state"]
        assert metadata["input_shapes"]["input_ids"] == [1, 512]
        assert metadata["output_shapes"]["last_hidden_state"] == [1, 512, 384]
        assert metadata["embedding_dim"] == 384

    def test_metadata_with_2d_output(self):
        """Test metadata extraction with 2D output shape."""
        session = Mock()
        
        session.get_inputs.return_value = []
        
        output = Mock()
        output.name = "output"
        output.shape = [1, 768]  # 2D output
        
        session.get_outputs.return_value = [output]

        metadata = get_model_metadata(session)

        assert metadata["embedding_dim"] == 768

    def test_metadata_without_embedding_dim(self):
        """Test metadata extraction when embedding dim cannot be determined."""
        session = Mock()
        
        session.get_inputs.return_value = []
        
        output = Mock()
        output.name = "output"
        output.shape = [1]  # 1D output (unusual)
        
        session.get_outputs.return_value = [output]

        metadata = get_model_metadata(session)

        assert "embedding_dim" not in metadata

    def test_metadata_with_dynamic_shapes(self):
        """Test metadata extraction with dynamic shapes."""
        session = Mock()
        
        session.get_inputs.return_value = []
        
        output = Mock()
        output.name = "output"
        output.shape = ["batch", "seq_len", 384]  # Dynamic batch and seq_len
        
        session.get_outputs.return_value = [output]

        metadata = get_model_metadata(session)

        # Should still extract embedding dimension
        assert metadata["embedding_dim"] == 384
