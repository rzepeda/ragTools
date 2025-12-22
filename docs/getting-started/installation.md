# Installation

This guide will walk you through installing RAG Factory and its dependencies.

---

## Prerequisites

Before installing RAG Factory, ensure you have:

- **Python 3.8+** installed
- **PostgreSQL 12+** with the **pgvector** extension
- **pip** package manager
- (Optional) **virtualenv** or **conda** for environment management

---

## Installation Steps

### 1. Create a Virtual Environment

We recommend using a virtual environment to isolate dependencies:

=== "venv"

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

=== "conda"

    ```bash
    conda create -n rag-factory python=3.9
    conda activate rag-factory
    ```

### 2. Install RAG Factory

Install from PyPI:

```bash
pip install rag-factory
```

Or install from source:

```bash
git clone #.git
cd rag-factory
pip install -e .
```

### 3. Install Optional Dependencies

For CLI support:

```bash
pip install rag-factory[cli]
```

For development:

```bash
pip install rag-factory[dev]
```

For all features:

```bash
pip install rag-factory[all]
```

---

## Database Setup

RAG Factory requires PostgreSQL with the pgvector extension for vector similarity search.

### Install PostgreSQL

=== "Ubuntu/Debian"

    ```bash
    sudo apt update
    sudo apt install postgresql postgresql-contrib
    ```

=== "macOS"

    ```bash
    brew install postgresql
    brew services start postgresql
    ```

=== "Windows"

    Download and install from [postgresql.org](https://www.postgresql.org/download/windows/)

### Install pgvector Extension

1. Install pgvector:

    === "Ubuntu/Debian"
    
        ```bash
        sudo apt install postgresql-server-dev-all
        cd /tmp
        git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
        cd pgvector
        make
        sudo make install
        ```
    
    === "macOS"
    
        ```bash
        brew install pgvector
        ```

2. Enable the extension in your database:

    ```sql
    CREATE EXTENSION IF NOT EXISTS vector;
    ```

### Create Database

Create a database for RAG Factory:

```bash
createdb ragdb
```

Or using SQL:

```sql
CREATE DATABASE ragdb;
```

---

## Environment Variables

Set up your environment variables for database connection:

```bash
export RAG_DB_HOST=localhost
export RAG_DB_PORT=5432
export RAG_DB_NAME=ragdb
export RAG_DB_USER=your_username
export RAG_DB_PASSWORD=your_password
```

Or create a `.env` file:

```env
RAG_DB_HOST=localhost
RAG_DB_PORT=5432
RAG_DB_NAME=ragdb
RAG_DB_USER=your_username
RAG_DB_PASSWORD=your_password
```

---

## Run Database Migrations

Initialize the database schema:

```bash
# Using Alembic migrations
alembic upgrade head
```

Or using the CLI:

```bash
rag-factory db init
```

---

## Verify Installation

Verify that RAG Factory is installed correctly:

```python
import rag_factory
print(rag_factory.__version__)
```

Test database connection:

```python
from rag_factory.database.connection import get_connection

# Test connection
conn = get_connection()
print("Database connection successful!")
```

---

## LLM Provider Setup (Optional)

If you plan to use strategies that require LLM providers (e.g., Contextual Retrieval, Self-Reflective), set up API keys:

### OpenAI

```bash
export OPENAI_API_KEY=your_openai_api_key
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=your_anthropic_api_key
```

---

## Next Steps

Now that you have RAG Factory installed, you can:

- [Follow the Quick Start Guide](quick-start.md) to create your first retrieval system
- [Learn about Configuration](configuration.md) options
- [Explore Available Strategies](../strategies/overview.md)

---

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'rag_factory'`**

Solution: Ensure you've activated your virtual environment and installed the package.

**Issue: `psycopg2` installation fails**

Solution: Install PostgreSQL development headers:

```bash
# Ubuntu/Debian
sudo apt install libpq-dev

# macOS
brew install postgresql
```

**Issue: pgvector extension not found**

Solution: Ensure pgvector is properly installed and the extension is created in your database.

For more troubleshooting help, see the <!-- BROKEN LINK: Troubleshooting Guide <!-- (broken link to: ../troubleshooting/common-errors.md) --> --> Troubleshooting Guide.
