# Simple RAG Factory Example

This is the simplest possible example to get started with RAG Factory. It demonstrates basic document chunking.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the example:
```bash
python basic_retrieval.py
```

## What This Example Does

1. **Imports RAG Factory**: Shows how to import the library
2. **Creates a Chunker**: Initializes a fixed-size chunking strategy
3. **Processes a Document**: Chunks a sample text document
4. **Displays Results**: Shows the created chunks with metadata

## Expected Output

```
Created 3 chunks:

Chunk 1:
  Text: RAG Factory is a comprehensive library for building RAG applications.
  Tokens: 12

Chunk 2:
  Text: It provides multiple strategies for chunking, retrieval, and generation.
  Tokens: 11

Chunk 3:
  Text: You can easily switch between different approaches to find what works best.
  Tokens: 13
```

## Next Steps

- Check out the [medium example](../medium/) for pipeline usage
- Explore the [advanced example](../advanced/) for production patterns
- Read the [full documentation](../../docs/)

## Troubleshooting

**ImportError: No module named 'rag_factory'**
- Make sure you've installed the requirements: `pip install -r requirements.txt`
- If developing locally, install in editable mode from the project root: `pip install -e .`

**Other Issues**
- Check that you're using Python 3.8 or higher: `python --version`
- Try creating a fresh virtual environment
