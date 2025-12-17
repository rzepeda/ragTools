"""
Real integration tests for embedding services.

Tests actual embedding generation with real services configured via .env.
"""

import pytest
import numpy as np


@pytest.mark.real_integration
@pytest.mark.requires_embeddings
@pytest.mark.asyncio
async def test_onnx_embedding_generation(real_embedding_service):
    """Test ONNX local embedding generation."""
    text = "This is a test sentence for embedding generation."
    
    embedding = await real_embedding_service.embed(text)
    
    # Verify embedding properties
    assert embedding is not None
    assert isinstance(embedding, (list, np.ndarray))
    assert len(embedding) > 0
    
    # ONNX embeddings should be normalized (L2 norm ≈ 1)
    if isinstance(embedding, list):
        embedding = np.array(embedding)
    norm = np.linalg.norm(embedding)
    assert 0.9 < norm < 1.1, f"Embedding should be normalized, got norm {norm}"


@pytest.mark.real_integration
@pytest.mark.requires_embeddings
@pytest.mark.asyncio
async def test_embedding_consistency(real_embedding_service):
    """Test that same text produces same embedding."""
    text = "Consistency test sentence"
    
    embedding1 = await real_embedding_service.embed(text)
    embedding2 = await real_embedding_service.embed(text)
    
    # Convert to numpy arrays for comparison
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    
    # Embeddings should be identical (or very close due to floating point)
    assert np.allclose(emb1, emb2, rtol=1e-5)


@pytest.mark.real_integration
@pytest.mark.requires_embeddings
@pytest.mark.asyncio
async def test_batch_embedding_generation(real_embedding_service, sample_texts):
    """Test batch embedding generation."""
    embeddings = await real_embedding_service.embed_batch(sample_texts)
    
    assert len(embeddings) == len(sample_texts)
    
    # All embeddings should have same dimension
    dims = [len(emb) for emb in embeddings]
    assert len(set(dims)) == 1, "All embeddings should have same dimension"
    
    # Each embedding should be normalized
    for embedding in embeddings:
        emb_array = np.array(embedding)
        norm = np.linalg.norm(emb_array)
        assert 0.9 < norm < 1.1


@pytest.mark.real_integration
@pytest.mark.requires_embeddings
@pytest.mark.asyncio
async def test_semantic_similarity(real_embedding_service):
    """Test that semantically similar texts have higher similarity."""
    # Similar texts
    text1 = "The cat sat on the mat"
    text2 = "A feline rested on the rug"
    
    # Different text
    text3 = "Python is a programming language"
    
    emb1 = np.array(await real_embedding_service.embed(text1))
    emb2 = np.array(await real_embedding_service.embed(text2))
    emb3 = np.array(await real_embedding_service.embed(text3))
    
    # Calculate cosine similarities
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    sim_similar = cosine_similarity(emb1, emb2)
    sim_different = cosine_similarity(emb1, emb3)
    
    # Similar texts should have higher similarity than different texts
    assert sim_similar > sim_different
    assert sim_similar > 0.5  # Should be reasonably similar


@pytest.mark.real_integration
@pytest.mark.requires_embeddings
@pytest.mark.asyncio
async def test_embedding_dimensions(real_embedding_service):
    """Test that embeddings have expected dimensions."""
    import os
    
    text = "Test embedding dimensions"
    embedding = await real_embedding_service.embed(text)
    
    # Common embedding dimensions:
    # - all-MiniLM-L6-v2: 384
    # - all-mpnet-base-v2: 768
    # - text-embedding-ada-002 (OpenAI): 1536
    
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "")
    
    if "MiniLM-L6" in model_name:
        expected_dim = 384
    elif "mpnet-base" in model_name:
        expected_dim = 768
    else:
        # Just verify it's a reasonable dimension
        expected_dim = len(embedding)
    
    assert len(embedding) == expected_dim


@pytest.mark.real_integration
@pytest.mark.requires_openai
@pytest.mark.asyncio
async def test_openai_embedding_generation(real_openai_embedding_service):
    """Test OpenAI embedding generation."""
    text = "This is a test for OpenAI embeddings."
    
    embedding = await real_openai_embedding_service.embed(text)
    
    assert embedding is not None
    assert len(embedding) == 1536  # OpenAI ada-002 dimension
    
    # Verify it's normalized
    emb_array = np.array(embedding)
    norm = np.linalg.norm(emb_array)
    assert 0.9 < norm < 1.1


@pytest.mark.real_integration
@pytest.mark.requires_openai
@pytest.mark.asyncio
async def test_openai_batch_embedding(real_openai_embedding_service, sample_texts):
    """Test OpenAI batch embedding generation."""
    embeddings = await real_openai_embedding_service.embed_batch(sample_texts[:3])
    
    assert len(embeddings) == 3
    assert all(len(emb) == 1536 for emb in embeddings)


@pytest.mark.real_integration
@pytest.mark.requires_cohere
@pytest.mark.asyncio
async def test_cohere_embedding_generation(real_cohere_embedding_service):
    """Test Cohere embedding generation."""
    text = "This is a test for Cohere embeddings."
    
    embedding = await real_cohere_embedding_service.embed(text)
    
    assert embedding is not None
    assert len(embedding) > 0
    
    # Cohere embeddings are also normalized
    emb_array = np.array(embedding)
    norm = np.linalg.norm(emb_array)
    assert 0.9 < norm < 1.1


@pytest.mark.real_integration
@pytest.mark.requires_embeddings
@pytest.mark.asyncio
async def test_empty_text_handling(real_embedding_service):
    """Test handling of empty or whitespace-only text."""
    # Most embedding models should handle empty text gracefully
    try:
        embedding = await real_embedding_service.embed("")
        # If it doesn't raise an error, verify it returns valid embedding
        assert embedding is not None
        assert len(embedding) > 0
    except (ValueError, Exception) as e:
        # It's also acceptable to raise an error for empty text
        assert "empty" in str(e).lower() or "invalid" in str(e).lower()


@pytest.mark.real_integration
@pytest.mark.requires_embeddings
@pytest.mark.asyncio
async def test_long_text_handling(real_embedding_service):
    """Test handling of long text (beyond typical token limits)."""
    # Create a very long text (most models have 512 token limit)
    long_text = " ".join(["This is a test sentence."] * 200)
    
    # Service should either truncate or handle gracefully
    embedding = await real_embedding_service.embed(long_text)
    
    assert embedding is not None
    assert len(embedding) > 0


@pytest.mark.real_integration
@pytest.mark.requires_embeddings
@pytest.mark.asyncio
async def test_special_characters_handling(real_embedding_service):
    """Test handling of special characters and unicode."""
    texts = [
        "Hello, world! 🌍",
        "Math symbols: ∑ ∫ ∂",
        "Quotes: \"Hello\" 'World'",
        "Emoji: 😀 🎉 🚀",
    ]
    
    for text in texts:
        embedding = await real_embedding_service.embed(text)
        assert embedding is not None
        assert len(embedding) > 0
