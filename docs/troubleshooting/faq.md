# Frequently Asked Questions (FAQ)

Common questions and answers about RAG Factory.

---

## General Questions

### What is RAG Factory?

RAG Factory is a flexible factory system for combining multiple Retrieval-Augmented Generation (RAG) strategies. It provides 10 production-ready strategies that can be used individually or combined in pipelines.

### What are the system requirements?

- Python 3.8+
- PostgreSQL 12+ with pgvector extension
- (Optional) Redis for caching

### Is RAG Factory production-ready?

Yes! RAG Factory is designed for production use with built-in observability, performance optimization, and comprehensive testing.

---

## Installation Questions

### How do I install RAG Factory?

```bash
pip install rag-factory
```

See the [Installation Guide](../getting-started/installation.md) for detailed instructions.

### Do I need to install PostgreSQL?

Yes, RAG Factory requires PostgreSQL with the pgvector extension for vector similarity search.

### Can I use a different database?

Currently, RAG Factory only supports PostgreSQL with pgvector. Support for other vector databases may be added in the future.

---

## Strategy Selection Questions

### Which strategy should I use?

It depends on your use case. See the [Strategy Selection Guide](../guides/strategy-selection.md) for a decision tree and comparison matrix.

### Can I combine multiple strategies?

Yes! Use the `StrategyPipeline` to combine multiple strategies. See the <!-- BROKEN LINK: Pipeline Tutorial <!-- (broken link to: ../tutorials/pipeline-setup.md) --> --> Pipeline Tutorial.

### What's the difference between Contextual and Hierarchical RAG?

- **Contextual RAG**: Adds context to chunks before embedding
- **Hierarchical RAG**: Preserves document hierarchy (sections, paragraphs)

Both improve context, but Hierarchical RAG is better for structured documents.

---

## Performance Questions

### How can I improve query performance?

- Use caching (Redis)
- Add database indexes
- Reduce `top_k` value
- Use metadata filtering
- Enable batch processing

See the <!-- BROKEN LINK: Performance Tuning Guide <!-- (broken link to: ../guides/performance-tuning.md) --> --> Performance Tuning Guide.

### Why are my queries slow?

Common causes:
- Large `top_k` value
- No database indexes
- Expensive strategies (e.g., Agentic RAG, Self-Reflective)
- LLM API latency

### How much does it cost to run?

Costs depend on:
- LLM API usage (OpenAI, Anthropic)
- Embedding API calls
- Database hosting
- Redis hosting (if used)

Strategies like Reranking and Query Expansion are more cost-effective than Agentic RAG or Self-Reflective.

---

## Configuration Questions

### How do I configure RAG Factory?

You can use:
- Python dictionaries
- YAML files
- JSON files
- Environment variables

See the [Configuration Guide](../getting-started/configuration.md).

### Where do I put my API keys?

Use environment variables:

```bash
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
```

### Can I use custom embedding models?

Yes! RAG Factory supports any Sentence-Transformers model or custom embedding services.

---

## Deployment Questions

### How do I deploy to production?

See the <!-- BROKEN LINK: Production Deployment Guide <!-- (broken link to: ../tutorials/production-deployment.md) --> --> Production Deployment Guide for best practices.

### Does RAG Factory support horizontal scaling?

Yes! RAG Factory is stateless and can run on multiple instances with shared PostgreSQL and Redis.

### What monitoring should I set up?

- Query latency metrics
- LLM API costs
- Database performance
- Error rates

RAG Factory has built-in observability features.

---

## Troubleshooting Questions

### I'm getting import errors

Make sure you've installed RAG Factory and activated your virtual environment:

```bash
source venv/bin/activate
pip install rag-factory
```

### My embeddings aren't being generated

Check that:
- Embedding service is configured
- API keys are set (if using API-based embeddings)
- Model is downloaded (for local models)

### Database connection fails

Verify:
- PostgreSQL is running
- Connection credentials are correct
- pgvector extension is installed
- Database exists

---

## Development Questions

### How do I add a custom strategy?

See the <!-- BROKEN LINK: Adding Strategies Guide <!-- (broken link to: ../contributing/adding-strategies.md) --> --> Adding Strategies Guide for step-by-step instructions.

### How do I run tests?

```bash
pytest
```

See the <!-- BROKEN LINK: Testing Guide <!-- (broken link to: ../contributing/testing.md) --> --> Testing Guide.

### How do I build documentation?

```bash
mkdocs serve
```

See the <!-- BROKEN LINK: Documentation Guide <!-- (broken link to: ../contributing/documentation.md) --> --> Documentation Guide.

---

## Support

### Where can I get help?

- [GitHub Issues](#/issues) - Bug reports
- [GitHub Discussions](#/discussions) - Questions
- [Documentation](../index.md) - Guides and tutorials

### How do I report a bug?

Open an issue on GitHub with:
- Clear description
- Steps to reproduce
- System information
- Error messages

### Can I contribute?

Yes! See the [Contributing Guide](../contributing/index.md).

---

## Still have questions?

Open a [GitHub Discussion](#/discussions) and we'll be happy to help!
