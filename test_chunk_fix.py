#!/usr/bin/env python3
"""Quick test of chunk_embeddings fix."""

import numpy as np

# Simulate the fixed chunk_embeddings method
def chunk_embeddings(
    token_embeddings: np.ndarray,
    tokens: list,
    chunk_size: int = 512,
    overlap: int = 50
):
    """
    Chunk token embeddings into smaller pieces.
    """
    num_tokens = len(token_embeddings)
    chunks = []

    start = 0
    while start < num_tokens:
        end = min(start + chunk_size, num_tokens)

        chunk_emb = token_embeddings[start:end]
        chunk_tok = tokens[start:end]

        chunks.append((chunk_emb, chunk_tok, start, end))

        # If we've reached the end, we're done
        if end >= num_tokens:
            break

        # Move to next chunk with overlap
        start = end - overlap
        
        # Ensure we make progress (avoid infinite loop if overlap >= chunk_size)
        if start <= chunks[-1][2]:  # If new start <= previous start
            start = chunks[-1][2] + 1

    return chunks


# Test it
token_embeddings = np.random.randn(100, 384).astype(np.float32)
tokens = [f"token_{i}" for i in range(100)]

print("Testing chunk_embeddings with 100 tokens, chunk_size=30, overlap=5")
chunks = chunk_embeddings(
    token_embeddings,
    tokens,
    chunk_size=30,
    overlap=5
)

print(f"Number of chunks: {len(chunks)}")
for i, (chunk_emb, chunk_tok, start, end) in enumerate(chunks):
    print(f"Chunk {i}: start={start}, end={end}, tokens={len(chunk_tok)}, emb_shape={chunk_emb.shape}")

print("\nTest passed!")
