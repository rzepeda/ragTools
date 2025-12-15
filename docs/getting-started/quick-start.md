# Quick Start

Get up and running with RAG Factory in under 5 minutes! This guide will walk you through creating your first retrieval system.

---

## Step 1: Install RAG Factory

If you haven't already, install RAG Factory:

```bash
pip install rag-factory
```

See the [Installation Guide](installation.md) for detailed setup instructions.

---

## Step 2: Set Up Database Connection

Create a configuration file or set environment variables:

```python
import os

# Set database connection
os.environ["RAG_DB_HOST"] = "localhost"
os.environ["RAG_DB_PORT"] = "5432"
os.environ["RAG_DB_NAME"] = "ragdb"
os.environ["RAG_DB_USER"] = "your_username"
os.environ["RAG_DB_PASSWORD"] = "your_password"
```

---

## Step 3: Index Your First Document

Let's index a simple document:

```python
from rag_factory.repositories.document import DocumentRepository
from rag_factory.repositories.chunk import ChunkRepository
from rag_factory.database.connection import get_connection

# Sample document
document_text = """
Machine learning is a subset of artificial intelligence that focuses on 
developing systems that can learn from and make decisions based on data. 
It involves training algorithms on large datasets to recognize patterns 
and make predictions without being explicitly programmed for every scenario.
"""

# Get database connection
conn = get_connection()

# Create repositories
doc_repo = DocumentRepository(conn)
chunk_repo = ChunkRepository(conn)

# Create document
document = doc_repo.create(
    content=document_text,
    metadata={"source": "quick_start", "topic": "machine_learning"}
)

print(f"Document created with ID: {document.id}")
```

---

## Step 4: Create Chunks

Split the document into chunks for retrieval:

```python
from rag_factory.strategies.chunking.strategy import ChunkingStrategy
from rag_factory.strategies.chunking.config import ChunkingConfig

# Configure chunking
config = ChunkingConfig(
    chunk_size=200,
    chunk_overlap=50,
    chunking_method="fixed"
)

# Create chunking strategy
chunker = ChunkingStrategy(config)

# Chunk the document
chunks = chunker.chunk_document(document_text, document.id)

# Store chunks
for chunk in chunks:
    chunk_repo.create(
        document_id=document.id,
        content=chunk.content,
        chunk_index=chunk.chunk_index,
        metadata=chunk.metadata
    )

print(f"Created {len(chunks)} chunks")
```

---

## Step 5: Generate Embeddings

Generate embeddings for the chunks:

```python
from rag_factory.services.embedding_service import EmbeddingService

# Initialize embedding service
embedding_service = EmbeddingService(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Get all chunks for the document
stored_chunks = chunk_repo.get_by_document_id(document.id)

# Generate and store embeddings
for chunk in stored_chunks:
    embedding = embedding_service.embed_text(chunk.content)
    chunk_repo.update_embedding(chunk.id, embedding)

print("Embeddings generated and stored")
```

---

## Step 6: Perform Your First Query

Now let's query the system:

```python
# Query the system
query = "What is machine learning?"

# Generate query embedding
query_embedding = embedding_service.embed_text(query)

# Search for similar chunks
results = chunk_repo.search_similar(
    query_embedding=query_embedding,
    top_k=3
)

# Display results
print(f"\nQuery: {query}\n")
print("Results:")
for i, result in enumerate(results, 1):
    print(f"\n{i}. Score: {result.score:.4f}")
    print(f"   Content: {result.content[:100]}...")
```

**Expected Output:**

```
Query: What is machine learning?

Results:

1. Score: 0.8523
   Content: Machine learning is a subset of artificial intelligence that focuses on 
   developing systems...

2. Score: 0.7234
   Content: It involves training algorithms on large datasets to recognize patterns 
   and make predictions...
```

---

## Step 7: Use a RAG Strategy

Let's use a more advanced strategy - Contextual Retrieval:

```python
from rag_factory.strategies.contextual.strategy import ContextualRetrievalStrategy
from rag_factory.strategies.contextual.config import ContextualRetrievalConfig

# Configure contextual retrieval
config = ContextualRetrievalConfig(
    chunk_size=200,
    context_window=1,  # Include 1 chunk before and after
    top_k=3
)

# Create strategy
strategy = ContextualRetrievalStrategy(config)

# Perform retrieval with context
results = strategy.retrieve(
    query="What is machine learning?",
    document_id=document.id
)

# Display results with context
print("\nContextual Retrieval Results:")
for i, result in enumerate(results, 1):
    print(f"\n{i}. Score: {result.score:.4f}")
    print(f"   Content: {result.content}")
    if result.context:
        print(f"   Context: {result.context}")
```

---

## Complete Example

Here's the complete code in one script:

```python
import os
from rag_factory.repositories.document import DocumentRepository
from rag_factory.repositories.chunk import ChunkRepository
from rag_factory.database.connection import get_connection
from rag_factory.services.embedding_service import EmbeddingService
from rag_factory.strategies.chunking.strategy import ChunkingStrategy
from rag_factory.strategies.chunking.config import ChunkingConfig

# Set up database connection
os.environ["RAG_DB_HOST"] = "localhost"
os.environ["RAG_DB_NAME"] = "ragdb"

# Sample document
document_text = """
Machine learning is a subset of artificial intelligence that focuses on 
developing systems that can learn from and make decisions based on data.
"""

# Initialize services
conn = get_connection()
doc_repo = DocumentRepository(conn)
chunk_repo = ChunkRepository(conn)
embedding_service = EmbeddingService()

# Create and chunk document
document = doc_repo.create(content=document_text, metadata={})
chunker = ChunkingStrategy(ChunkingConfig(chunk_size=200))
chunks = chunker.chunk_document(document_text, document.id)

# Store chunks with embeddings
for chunk in chunks:
    stored_chunk = chunk_repo.create(
        document_id=document.id,
        content=chunk.content,
        chunk_index=chunk.chunk_index
    )
    embedding = embedding_service.embed_text(chunk.content)
    chunk_repo.update_embedding(stored_chunk.id, embedding)

# Query
query = "What is machine learning?"
query_embedding = embedding_service.embed_text(query)
results = chunk_repo.search_similar(query_embedding, top_k=3)

# Display results
for i, result in enumerate(results, 1):
    print(f"{i}. {result.content[:100]}... (Score: {result.score:.4f})")
```

---

## Next Steps

Congratulations! You've created your first RAG retrieval system. Here's what to explore next:

- **[Learn About Configuration](configuration.md)** - Customize your setup
- **[Explore Strategies](../strategies/overview.md)** - Discover all 10 RAG strategies
- **<!-- BROKEN LINK: Build Pipelines <!-- (broken link to: ../tutorials/pipeline-setup.md) --> --> Build Pipelines** - Combine multiple strategies
- **<!-- BROKEN LINK: Production Deployment <!-- (broken link to: ../tutorials/production-deployment.md) --> --> Production Deployment** - Deploy to production

---

## Need Help?

- Check the [FAQ](../troubleshooting/faq.md) for common questions
- See <!-- BROKEN LINK: Troubleshooting <!-- (broken link to: ../troubleshooting/common-errors.md) --> --> Troubleshooting for common issues
- Open an issue on [GitHub](https://github.com/yourusername/rag-factory/issues)
