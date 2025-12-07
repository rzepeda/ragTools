# RAG Factory Examples

Welcome to the RAG Factory examples! This directory contains comprehensive examples demonstrating how to use the RAG Factory library, from simple getting-started examples to advanced production patterns.

## 📚 Example Categories

### 🚀 Getting Started

#### [Simple Example](./simple/)
**Complexity:** Beginner  
**Time:** 5 minutes  
**What you'll learn:** Basic chunking with RAG Factory

The simplest possible example to get you started. Shows how to:
- Import the library
- Create a chunking strategy
- Process a document
- Display results

```bash
cd simple
pip install -r requirements.txt
python basic_retrieval.py
```

#### [Medium Example](./medium/)
**Complexity:** Intermediate  
**Time:** 15 minutes  
**What you'll learn:** Multiple strategies and performance comparison

Demonstrates using multiple chunking strategies together:
- Structural chunking (respects document structure)
- Fixed-size chunking (fast baseline)
- Hybrid chunking (combines approaches)
- Configuration via YAML
- Performance timing and comparison

```bash
cd medium
pip install -r requirements.txt
python strategy_pipeline.py
```

#### [Advanced Example](./advanced/)
**Complexity:** Advanced  
**Time:** 30 minutes  
**What you'll learn:** Production-ready patterns

Full-featured RAG system with:
- Multiple strategies with retry logic
- Comprehensive logging and metrics
- Batch processing
- Error handling
- Configuration management
- Performance optimization

```bash
cd advanced
pip install -r requirements.txt
python full_system.py
```

### 🎯 Domain-Specific Examples

Real-world use cases showing how to apply RAG Factory to specific domains:

- **[Legal](./domain/legal/)** - Contract analysis, case law retrieval
- **[Medical](./domain/medical/)** - Clinical notes, research papers
- **[Customer Support](./domain/support/)** - Ticket resolution, knowledge base

### 🔌 Framework Integrations

Examples showing how to integrate RAG Factory with popular frameworks:

- **[FastAPI](./integrations/fastapi/)** - REST API with async support
- **[Flask](./integrations/flask/)** - Web application
- **[LangChain](./integrations/langchain/)** - Custom retriever integration
- **[Streamlit](./integrations/streamlit/)** - Interactive UI
- **[CLI](./integrations/cli/)** - Command-line tool

### 🐳 Docker Setup

**[Docker Compose](./docker/)** - Complete development environment

One-command setup with:
- PostgreSQL with pgvector
- Redis for caching
- Application container
- Sample data

```bash
cd docker
cp .env.example .env
# Edit .env with your API keys
docker-compose up
```

### 📓 Jupyter Notebooks

Interactive notebooks for exploration and experimentation:

- **[01_exploration.ipynb](./notebooks/)** - Basic usage and visualization
- **[02_performance.ipynb](./notebooks/)** - Performance benchmarking
- **[03_experimentation.ipynb](./notebooks/)** - Parameter tuning

```bash
cd notebooks
pip install -r requirements.txt
jupyter notebook
```

### 🛠️ Legacy CLI Examples

Original CLI examples (see subdirectory README for details):

- `cli_config_example.yaml` - CLI configuration
- `benchmark_dataset_example.json` - Benchmark datasets
- Various strategy examples

## 🎓 Learning Path

We recommend following this progression:

1. **Start Here:** [Simple Example](./simple/) - Get familiar with basic concepts
2. **Next:** [Medium Example](./medium/) - Learn about multiple strategies
3. **Then:** [Advanced Example](./advanced/) - See production patterns
4. **Explore:** Pick a [domain example](./domain/) relevant to your use case
5. **Integrate:** Check out [framework integrations](./integrations/)
6. **Deploy:** Use the [Docker setup](./docker/) for deployment

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone <repository>
cd rag_factory
pip install -e .

# 2. Run the simple example
cd examples/simple
python basic_retrieval.py

# 3. Try the medium example
cd ../medium
python strategy_pipeline.py

# 4. Explore advanced features
cd ../advanced
python full_system.py
```

## 📖 Documentation

Each example directory contains:
- **README.md** - Detailed setup instructions and explanation
- **requirements.txt** - Python dependencies
- **Sample data** - Example documents to process
- **Configuration files** - YAML configs where applicable

## 🧪 Testing

All examples are tested automatically. Run the test suite:

```bash
# Test syntax and structure
pytest tests/unit/examples/

# Test that examples run
pytest tests/integration/examples/
```

## 💡 Tips and Best Practices

1. **Start Simple:** Begin with the simple example before moving to complex ones
2. **Read the READMEs:** Each example has detailed documentation
3. **Experiment:** Modify the examples to fit your use case
4. **Use Virtual Environments:** Keep dependencies isolated
5. **Check Logs:** Enable logging to understand what's happening

## 🐛 Troubleshooting

### Import Errors

```bash
# Install in editable mode from project root
pip install -e .
```

### Missing Dependencies

```bash
# Install example-specific requirements
cd examples/simple  # or medium, advanced, etc.
pip install -r requirements.txt
```

### Database Connection Issues

```bash
# Use Docker for easy database setup
cd examples/docker
docker-compose up postgres
```

## 🤝 Contributing

Found a bug or want to add an example? See [CONTRIBUTING.md](../CONTRIBUTING.md)

## 📚 Additional Resources

- [Full Documentation](../docs/)
- [API Reference](../docs/api/)
- [Strategy Guide](../docs/guides/strategy-selection.md)
- [CLI User Guide](../docs/CLI-USER-GUIDE.md)

## 📄 License

See [LICENSE](../LICENSE) for details.
