"""
Unit tests for document embedder with ONNX.
"""

import os
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from rag_factory.strategies.late_chunking.document_embedder import DocumentEmbedder
from rag_factory.strategies.late_chunking.models import LateChunkingConfig

# Get embedding model from environment or use ONNX-compatible default
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "Xenova/all-MiniLM-L6-v2")


@pytest.fixture
def embedder_config():
    """Create test configuration for embedder."""
    return LateChunkingConfig(
        model_name=EMBEDDING_MODEL,
        max_document_tokens=512
    )


@pytest.fixture
def mock_session():
    """Mock ONNX session."""
    session = Mock()
    # Mock output metadata
    output_meta = Mock()
    output_meta.shape = [1, 512, 384]  # [batch, seq, dim]
    session.get_outputs.return_value = [output_meta]
    # Mock inputs for token_type_ids check
    mock_input = Mock()
    mock_input.name = "input_ids"
    session.get_inputs.return_value = [mock_input]
    return session


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer."""
    tokenizer = Mock()
    
    # Default token IDs
    default_token_ids = [1, 2, 3, 4, 5]
    tokenizer.encode = Mock(return_value=default_token_ids)
    
    # Mock the __call__ method to return encoded dict based on encode() call
    def mock_call(*args, **kwargs):
        # Call encode to get token IDs (supports both return_value and side_effect)
        token_ids = tokenizer.encode(*args, **kwargs)
        
        # Handle truncation if specified
        if kwargs.get('truncation', False) and 'max_length' in kwargs:
            max_length = kwargs['max_length']
            token_ids = token_ids[:max_length]
        
        num_tokens = len(token_ids)
        return {
            "input_ids": np.array([token_ids], dtype=np.int64),
            "attention_mask": np.array([[1] * num_tokens], dtype=np.int64)
        }
    tokenizer.side_effect = mock_call
    tokenizer.decode.side_effect = lambda ids, **kwargs: " ".join([f"token_{i}" for i in ids])
    tokenizer.model_max_length = 512  # Add model_max_length attribute
    return tokenizer


@pytest.fixture
def document_embedder(embedder_config, mock_session, mock_tokenizer):
    """Create document embedder instance with mocked dependencies."""
    with patch("rag_factory.strategies.late_chunking.document_embedder.download_onnx_model") as mock_download:
        with patch("rag_factory.strategies.late_chunking.document_embedder.create_onnx_session") as mock_create:
            with patch("rag_factory.strategies.late_chunking.document_embedder.get_model_metadata") as mock_metadata:
                with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_class:
                    from pathlib import Path
                    mock_download.return_value = Path("/fake/path/model.onnx")
                    mock_create.return_value = mock_session
                    mock_metadata.return_value = {"embedding_dim": 384}
                    mock_tok_class.return_value = mock_tokenizer
                    
                    embedder = DocumentEmbedder(embedder_config)
                    embedder.session = mock_session
                    embedder.tokenizer = mock_tokenizer
                    
                    return embedder


def test_document_embedding_basic(document_embedder, mock_session, mock_tokenizer):
    """Test basic document embedding."""
    text = "This is a test document with multiple sentences."
    
    # Mock ONNX output
    mock_embeddings = np.random.randn(1, 5, 384).astype(np.float32)
    mock_session.run.return_value = [mock_embeddings]
    
    doc_emb = document_embedder.embed_document(text, "test_doc")

    assert doc_emb.document_id == "test_doc"
    assert doc_emb.text == text
    assert len(doc_emb.full_embedding) == 384
    assert len(doc_emb.token_embeddings) == 5
    assert doc_emb.token_count == 5
    assert doc_emb.embedding_dim == 384


def test_token_embeddings_extracted(document_embedder, mock_session, mock_tokenizer):
    """Test that token-level embeddings are extracted."""
    text = "Hello world"
    
    mock_tokenizer.encode.return_value = [1, 2]
    mock_embeddings = np.random.randn(1, 2, 384).astype(np.float32)
    mock_session.run.return_value = [mock_embeddings]
    
    doc_emb = document_embedder.embed_document(text, "test_doc")

    # Should have token embeddings
    assert len(doc_emb.token_embeddings) == 2

    # Each token should have embedding
    for token_emb in doc_emb.token_embeddings:
        assert len(token_emb.embedding) == 384
        assert token_emb.start_char >= 0
        assert token_emb.end_char >= token_emb.start_char
        assert token_emb.position >= 0


def test_mean_pooling(document_embedder):
    """Test mean pooling for document embedding."""
    # Create fake token embeddings
    token_embeddings = np.random.randn(5, 384).astype(np.float32)
    attention_mask = np.ones(5, dtype=np.int64)

    mean_emb = document_embedder._mean_pooling(token_embeddings, attention_mask)

    assert mean_emb.shape == (384,)
    # Mean should be average of token embeddings
    expected_mean = token_embeddings.mean(axis=0)
    np.testing.assert_array_almost_equal(mean_emb, expected_mean, decimal=5)


def test_mean_pooling_with_mask(document_embedder):
    """Test mean pooling with attention mask."""
    # Create fake token embeddings with some masked tokens
    token_embeddings = np.random.randn(5, 384).astype(np.float32)
    attention_mask = np.array([1, 1, 1, 0, 0], dtype=np.int64)

    mean_emb = document_embedder._mean_pooling(token_embeddings, attention_mask)

    assert mean_emb.shape == (384,)
    # Mean should only include first 3 tokens
    expected_mean = token_embeddings[:3].mean(axis=0)
    np.testing.assert_array_almost_equal(mean_emb, expected_mean, decimal=5)


def test_long_document_truncation(document_embedder, mock_session, mock_tokenizer):
    """Test that long documents are truncated."""
    # Create very long text
    long_text = "This is a sentence. " * 1000
    
    # Mock tokenizer to return many tokens
    long_token_ids = list(range(1000))
    mock_tokenizer.encode.return_value = long_token_ids
    
    # Mock ONNX output for truncated length
    truncated_length = document_embedder.max_length
    mock_embeddings = np.random.randn(1, truncated_length, 384).astype(np.float32)
    mock_session.run.return_value = [mock_embeddings]

    doc_emb = document_embedder.embed_document(long_text, "long_doc")

    # Should be truncated to max_length
    assert doc_emb.token_count <= document_embedder.max_length


def test_batch_processing(document_embedder, mock_session, mock_tokenizer):
    """Test batch processing of multiple documents."""
    documents = [
        {"text": "First document", "document_id": "doc1"},
        {"text": "Second document", "document_id": "doc2"},
        {"text": "Third document", "document_id": "doc3"}
    ]
    
    # Mock ONNX output
    mock_embeddings = np.random.randn(1, 5, 384).astype(np.float32)
    mock_session.run.return_value = [mock_embeddings]

    doc_embeddings = document_embedder.embed_documents_batch(documents)

    assert len(doc_embeddings) == 3
    assert doc_embeddings[0].document_id == "doc1"
    assert doc_embeddings[1].document_id == "doc2"
    assert doc_embeddings[2].document_id == "doc3"

    for doc_emb in doc_embeddings:
        assert len(doc_emb.full_embedding) > 0
        assert len(doc_emb.token_embeddings) > 0


def test_embedding_dimensions_consistent(document_embedder, mock_session, mock_tokenizer):
    """Test that embedding dimensions are consistent."""
    # Mock different token counts
    mock_embeddings1 = np.random.randn(1, 3, 384).astype(np.float32)
    mock_embeddings2 = np.random.randn(1, 7, 384).astype(np.float32)
    
    # encode() is called twice per document: once from tokenizer() and once for truncation check
    mock_tokenizer.encode.side_effect = [
        [1, 2, 3],  # First call for doc1 (from tokenizer)
        [1, 2, 3],  # Second call for doc1 (truncation check)
        [1, 2, 3, 4, 5, 6, 7],  # First call for doc2 (from tokenizer)
        [1, 2, 3, 4, 5, 6, 7]   # Second call for doc2 (truncation check)
    ]
    mock_session.run.side_effect = [[mock_embeddings1], [mock_embeddings2]]
    
    text1 = "Short text"
    text2 = "This is a longer text with more words"

    doc_emb1 = document_embedder.embed_document(text1, "doc1")
    doc_emb2 = document_embedder.embed_document(text2, "doc2")

    # Embedding dimensions should be the same
    assert doc_emb1.embedding_dim == doc_emb2.embedding_dim
    assert len(doc_emb1.full_embedding) == len(doc_emb2.full_embedding)


def test_model_name_stored(document_embedder, mock_session, mock_tokenizer):
    """Test that model name is stored in document embedding."""
    text = "Test document"
    
    mock_embeddings = np.random.randn(1, 5, 384).astype(np.float32)
    mock_session.run.return_value = [mock_embeddings]
    
    doc_emb = document_embedder.embed_document(text, "test_doc")

    assert doc_emb.model_name == document_embedder.config.model_name


def test_chunk_embeddings(document_embedder):
    """Test chunking of token embeddings."""
    # Create mock embeddings
    token_embeddings = np.random.randn(100, 384).astype(np.float32)
    tokens = [f"token_{i}" for i in range(100)]

    chunks = document_embedder.chunk_embeddings(
        token_embeddings,
        tokens,
        chunk_size=30,
        overlap=5
    )

    assert len(chunks) > 1
    for chunk_emb, chunk_tok, start, end in chunks:
        assert chunk_emb.shape[0] == len(chunk_tok)
        assert chunk_emb.shape[1] == 384
        assert end - start == len(chunk_tok)


def test_pool_embeddings_mean(document_embedder):
    """Test mean pooling."""
    token_embeddings = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float32)

    pooled = document_embedder.pool_embeddings(token_embeddings, method="mean")

    expected = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(pooled, expected)


def test_pool_embeddings_max(document_embedder):
    """Test max pooling."""
    token_embeddings = np.array([
        [1.0, 5.0, 3.0],
        [4.0, 2.0, 6.0],
        [7.0, 8.0, 1.0]
    ], dtype=np.float32)

    pooled = document_embedder.pool_embeddings(token_embeddings, method="max")

    expected = np.array([7.0, 8.0, 6.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(pooled, expected)


def test_pool_embeddings_first(document_embedder):
    """Test first token pooling."""
    token_embeddings = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float32)

    pooled = document_embedder.pool_embeddings(token_embeddings, method="first")

    expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(pooled, expected)


def test_decode_tokens(document_embedder, mock_tokenizer):
    """Test token decoding."""
    token_ids = [1, 2, 3, 4, 5]
    
    tokens = document_embedder._decode_tokens(token_ids)
    
    assert len(tokens) == 5
    assert all(isinstance(t, str) for t in tokens)
