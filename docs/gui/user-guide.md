# RAG Factory GUI - User Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Strategy Selection](#strategy-selection)
4. [Indexing Documents](#indexing-documents)
5. [Querying](#querying)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

---

## Installation

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Python**: 3.8 or higher
- **Database**: PostgreSQL 12 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Disk Space**: 500MB for application + database storage

### Step 1: Install Python
Download and install Python from [python.org](https://www.python.org/downloads/).

Verify installation:
```bash
python --version  # Should show 3.8 or higher
```

### Step 2: Install PostgreSQL
Download and install PostgreSQL from [postgresql.org](https://www.postgresql.org/download/).

Create a database for RAG Factory:
```bash
createdb rag_factory
```

### Step 3: Install RAG Factory
```bash
# Clone or download the repository
cd rag-factory

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Configure Database
Edit `config/services.yaml`:
```yaml
services:
  db_main:
    type: postgresql
    host: localhost
    port: 5432
    database: rag_factory
    username: your_username
    password: your_password
```

### Step 5: Run Migrations
```bash
alembic upgrade head
```

### Step 6: Launch GUI
```bash
rag-factory gui
```

---

## Quick Start

### 1. Launch the Application
```bash
rag-factory gui
```

The main window will appear centered on your screen.

### 2. Select a Strategy
- Click the **Strategy** dropdown
- Select a strategy pair (e.g., "semantic-local-pair")
- Wait for "Strategy loaded successfully" message

### 3. Index Your First Document
**Option A: Index Text**
1. Type or paste text into the "Text to Index" box
2. Click **Index Text**
3. Wait for confirmation: "Indexed 1 document in X.XXs"

**Option B: Index File**
1. Click **Browse** next to "File to Index"
2. Select a text file (.txt, .md, etc.)
3. Click **Index File**
4. Wait for confirmation

### 4. Run Your First Query
1. Enter a query in the "Query" field (e.g., "What is machine learning?")
2. Select Top-K value (default: 5)
3. Click **Retrieve**
4. View results in the results panel

**Congratulations!** You've completed your first RAG workflow.

---

## Strategy Selection

### Understanding Strategy Pairs
A strategy pair consists of:
- **Indexing Strategy**: How documents are processed and stored
- **Retrieval Strategy**: How queries are matched to documents

### Available Strategies
Strategies are defined in the `strategies/` directory. Common strategies:
- `semantic-local-pair`: Local semantic search
- `semantic-reranking-pair`: Semantic search with reranking
- `hierarchical-rag-pair`: Hierarchical retrieval

### Loading a Strategy
1. Click the **Strategy** dropdown
2. Select your desired strategy
3. View configuration in the preview panel
4. Wait for "Strategy loaded successfully"

### Configuration Preview
The configuration preview shows:
- Strategy name and version
- Indexing configuration
- Retrieval configuration
- Dependencies

---

## Indexing Documents

### Text Indexing
**Best for**: Quick testing, short documents, code snippets

**Steps**:
1. Enter or paste text in the "Text to Index" box
2. Click **Index Text** (or press `Ctrl+I` then `Enter`)
3. Wait for completion
4. Check status bar for document/chunk counts

**Tips**:
- Text is chunked automatically based on strategy
- Each indexing operation creates a new document
- Clear the textbox after indexing (optional)

### File Indexing
**Best for**: Longer documents, batch processing

**Supported Formats**:
- Plain text (.txt)
- Markdown (.md)
- UTF-8 encoded files

**Steps**:
1. Click **Browse** (or press `Ctrl+F`)
2. Select a file
3. Click **Index File**
4. Wait for completion

**Tips**:
- Files are read with UTF-8 encoding (fallback to latin-1)
- Binary files are not supported
- Large files may take longer to process

### Monitoring Progress
- **Status Bar**: Shows current operation
- **Document Count**: Total documents indexed
- **Chunk Count**: Total chunks created
- **Button State**: Disabled during indexing

---

## Querying

### Entering Queries
1. Type your question in the "Query" field
2. Use natural language (e.g., "What is machine learning?")
3. Press `Ctrl+Q` to focus query field

### Adjusting Top-K
**Top-K** determines how many results to return:
- **1-3**: Very focused results
- **5**: Default, balanced
- **10+**: Broader coverage

### Understanding Results
Results are displayed with:
- **Rank**: [1], [2], [3], etc.
- **Score**: Relevance score (0.0000 to 1.0000)
- **Content**: Preview of matched content (200 chars)
- **Source**: Origin of the content

**Example Result**:
```
[1] Score: 0.8923
    Machine learning is a subset of artificial intelligence...
    Source: ml_basics.txt
```

### Result Interpretation
- **Score > 0.8**: Highly relevant
- **Score 0.6-0.8**: Moderately relevant
- **Score < 0.6**: Less relevant

### No Results Found
If you see "No results found":
1. Check that documents are indexed
2. Try different keywords
3. Try broader search terms
4. Check spelling

---

## Troubleshooting

### "Missing migrations" Error
**Cause**: Database schema is not up to date

**Solution**:
```bash
alembic upgrade head
```

### "Missing services" Error
**Cause**: `config/services.yaml` not found or invalid

**Solution**:
1. Check file exists: `config/services.yaml`
2. Verify YAML syntax
3. Ensure `db_main` is configured

### "Database connection failed" Error
**Cause**: Cannot connect to PostgreSQL

**Solutions**:
1. Verify PostgreSQL is running:
   ```bash
   # On Linux/macOS:
   pg_ctl status
   
   # On Windows:
   # Check Services for "PostgreSQL"
   ```
2. Check connection settings in `config/services.yaml`
3. Verify database exists:
   ```bash
   psql -l | grep rag_factory
   ```

### "No results found" (with indexed data)
**Possible Causes**:
- Query doesn't match indexed content
- Wrong strategy selected
- Indexing failed silently

**Solutions**:
1. View logs: Tools → View Logs
2. Try exact phrase from indexed document
3. Reload strategy and re-index

### "File encoding error"
**Cause**: File is binary or uses unsupported encoding

**Solution**:
- Use UTF-8 encoded text files
- Convert file to UTF-8:
  ```bash
  iconv -f ISO-8859-1 -t UTF-8 input.txt > output.txt
  ```

### Application Freezes
**Cause**: Long-running operation or threading issue

**Solutions**:
1. Wait for operation to complete
2. Check logs for errors
3. Restart application
4. Report issue with logs

---

## FAQ

### How do I add a new strategy?
1. Create a YAML file in `strategies/` directory
2. Define indexing and retrieval configurations
3. Click "Reload Configs" in File menu
4. Select new strategy from dropdown

### How do I clear all data?
1. Click "Clear All Data" button (or press `Ctrl+K`)
2. Confirm the action
3. Data is removed from database
4. Counters reset to zero

**Note**: This action cannot be undone!

### How do I view logs?
1. Click Tools → View Logs
2. Click "Refresh" to update
3. Click "Clear Buffer" to clear logs
4. Logs are also written to application log files

### What file formats are supported?
**Supported**:
- Plain text (.txt)
- Markdown (.md)
- Any UTF-8 encoded text file

**Not Supported**:
- PDF (requires preprocessing)
- Word documents (.docx)
- Binary files
- Images

### Can I index multiple files at once?
Currently, files must be indexed one at a time. For batch processing, use the CLI:
```bash
rag-factory index --strategy semantic-local-pair --files *.txt
```

### How do I change the database?
Edit `config/services.yaml`:
```yaml
services:
  db_main:
    type: postgresql
    host: your_host
    port: 5432
    database: your_database
```

Then restart the application.

### What are keyboard shortcuts?
- `Ctrl+L`: Focus strategy dropdown
- `Ctrl+I`: Focus text to index
- `Ctrl+F`: Browse for file
- `Ctrl+Q`: Focus query entry
- `Ctrl+R`: Retrieve
- `Ctrl+K`: Clear all data
- `Ctrl+H` or `F1`: Show help
- `Ctrl+W`: Close window

### How do I report a bug?
1. View logs: Tools → View Logs
2. Copy relevant log entries
3. Create an issue on GitHub
4. Include: OS, Python version, error message, logs

---

## Getting Help

- **Help Dialog**: Press `F1` or Help → Help
- **About**: Help → About
- **Logs**: Tools → View Logs
- **Documentation**: [GitHub Repository](#)

---

**Version**: 1.0.0  
**Last Updated**: 2024-12-19
