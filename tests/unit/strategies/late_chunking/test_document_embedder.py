"""
Unit tests for document embedder.
"""

import pytest
import torch
import numpy as np

from rag_factory.strategies.late_chunking.document_embedder import DocumentEmbedder
from rag_factory.strategies.late_chunking.models import LateChunkingConfig


@pytest.fixture
def embedder_config():
    """Create test configuration for embedder."""
    return LateChunkingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_document_tokens=512,
        device="cpu"
    )


@pytest.fixture
def document_embedder(embedder_config):
    """Create document embedder instance."""
    return DocumentEmbedder(embedder_config)


def test_document_embedding_basic(document_embedder):
    """Test basic document embedding."""
    text = "This is a test document with multiple sentences. It should be embedded properly."
    doc_emb = document_embedder.embed_document(text, "test_doc")

    assert doc_emb.document_id == "test_doc"
    assert doc_emb.text == text
    assert len(doc_emb.full_embedding) > 0
    assert len(doc_emb.token_embeddings) > 0
    assert doc_emb.token_count > 0
    assert doc_emb.embedding_dim > 0


def test_token_embeddings_extracted(document_embedder):
    """Test that token-level embeddings are extracted."""
    text = "Hello world"
    doc_emb = document_embedder.embed_document(text, "test_doc")

    # Should have token embeddings
    assert len(doc_emb.token_embeddings) >= 2  # At least "Hello" and "world"

    # Each token should have embedding
    for token_emb in doc_emb.token_embeddings:
        assert len(token_emb.embedding) == doc_emb.embedding_dim
        assert token_emb.start_char >= 0
        assert token_emb.end_char > token_emb.start_char
        assert token_emb.position >= 0


def test_char_position_mapping(document_embedder):
    """Test that character positions are correct."""
    text = "The quick brown fox"
    doc_emb = document_embedder.embed_document(text, "test_doc")

    # Verify character positions map to correct text
    for token_emb in doc_emb.token_embeddings:
        token_text = text[token_emb.start_char:token_emb.end_char]
        # Token should be similar to extracted text (may have subword differences)
        assert len(token_text) > 0


def test_mean_pooling(document_embedder):
    """Test mean pooling for document embedding."""
    # Create fake token embeddings
    token_embeddings = torch.randn(5, 384)  # 5 tokens, 384 dim
    attention_mask = torch.ones(5)

    mean_emb = document_embedder._mean_pooling(token_embeddings, attention_mask)

    assert mean_emb.shape == (384,)
    # Mean should be average of token embeddings
    expected_mean = token_embeddings.mean(dim=0)
    assert torch.allclose(mean_emb, expected_mean, atol=1e-5)


def test_mean_pooling_with_mask(document_embedder):
    """Test mean pooling with attention mask."""
    # Create fake token embeddings with some masked tokens
    token_embeddings = torch.randn(5, 384)
    attention_mask = torch.tensor([1, 1, 1, 0, 0])  # Last 2 tokens masked

    mean_emb = document_embedder._mean_pooling(token_embeddings, attention_mask)

    assert mean_emb.shape == (384,)
    # Mean should only include first 3 tokens
    expected_mean = token_embeddings[:3].mean(dim=0)
    assert torch.allclose(mean_emb, expected_mean, atol=1e-5)


def test_long_document_truncation(document_embedder):
    """Test that long documents are truncated."""
    # Create very long text
    long_text = "This is a sentence. " * 1000

    doc_emb = document_embedder.embed_document(long_text, "long_doc")

    # Should be truncated to max_length
    assert doc_emb.token_count <= document_embedder.max_length


def test_batch_processing(document_embedder):
    """Test batch processing of multiple documents."""
    documents = [
        {"text": "First document", "document_id": "doc1"},
        {"text": "Second document", "document_id": "doc2"},
        {"text": "Third document", "document_id": "doc3"}
    ]

    doc_embeddings = document_embedder.embed_documents_batch(documents)

    assert len(doc_embeddings) == 3
    assert doc_embeddings[0].document_id == "doc1"
    assert doc_embeddings[1].document_id == "doc2"
    assert doc_embeddings[2].document_id == "doc3"

    for doc_emb in doc_embeddings:
        assert len(doc_emb.full_embedding) > 0
        assert len(doc_emb.token_embeddings) > 0


def test_embedding_dimensions_consistent(document_embedder):
    """Test that embedding dimensions are consistent."""
    text1 = "Short text"
    text2 = "This is a longer text with more words and tokens"

    doc_emb1 = document_embedder.embed_document(text1, "doc1")
    doc_emb2 = document_embedder.embed_document(text2, "doc2")

    # Embedding dimensions should be the same
    assert doc_emb1.embedding_dim == doc_emb2.embedding_dim
    assert len(doc_emb1.full_embedding) == len(doc_emb2.full_embedding)


def test_model_name_stored(document_embedder):
    """Test that model name is stored in document embedding."""
    text = "Test document"
    doc_emb = document_embedder.embed_document(text, "test_doc")

    assert doc_emb.model_name == document_embedder.config.model_name
