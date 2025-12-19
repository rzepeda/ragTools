# RAG Factory GUI - Developer Guide

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Component Structure](#component-structure)
3. [Threading Model](#threading-model)
4. [Adding Features](#adding-features)
5. [Testing Strategy](#testing-strategy)
6. [Debugging Tips](#debugging-tips)

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RAGFactoryGUI                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Main Window (Tkinter)                │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  Strategy Selection                         │  │  │
│  │  │  Configuration Preview                      │  │  │
│  │  │  Text Indexing                              │  │  │
│  │  │  File Indexing                              │  │  │
│  │  │  Query Interface                            │  │  │
│  │  │  Results Display                            │  │  │
│  │  │  Status Bar                                 │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
│                          │                              │
│                          ▼                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │           Backend Integration Layer               │  │
│  │  ┌─────────────────┐  ┌──────────────────────┐   │  │
│  │  │ ServiceRegistry │  │ StrategyPairManager  │   │  │
│  │  └─────────────────┘  └──────────────────────┘   │  │
│  └───────────────────────────────────────────────────┘  │
│                          │                              │
│                          ▼                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Strategy Execution                   │  │
│  │  ┌──────────────────┐  ┌─────────────────────┐   │  │
│  │  │ IndexingPipeline │  │ RetrievalPipeline   │   │  │
│  │  └──────────────────┘  └─────────────────────┘   │  │
│  └───────────────────────────────────────────────────┘  │
│                          │                              │
│                          ▼                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │                Database Layer                     │  │
│  │              (PostgreSQL + pgvector)              │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Key Components

1. **RAGFactoryGUI**: Main application class
   - Manages window and widgets
   - Coordinates backend operations
   - Handles user interactions

2. **ServiceRegistry**: Backend service management
   - Initializes database connections
   - Manages service lifecycle
   - Provides service access

3. **StrategyPairManager**: Strategy management
   - Loads strategy configurations
   - Instantiates strategy pairs
   - Manages strategy lifecycle

4. **Pipelines**: Core RAG operations
   - IndexingPipeline: Document processing
   - RetrievalPipeline: Query processing

### Data Flow

**Indexing Flow**:
```
User Input → GUI → IndexingPipeline → Database
```

**Retrieval Flow**:
```
User Query → GUI → RetrievalPipeline → Database → Results → GUI
```

---

## Component Structure

### Main Window Class

**File**: `rag_factory/gui/main_window.py`

**Key Attributes**:
```python
class RAGFactoryGUI:
    # Configuration
    config_path: Path
    strategies_dir: Path
    alembic_config: str
    
    # Backend
    service_registry: ServiceRegistry
    strategy_manager: StrategyPairManager
    
    # Strategy state
    current_strategy_name: str
    indexing_strategy: IIndexingStrategy
    retrieval_strategy: IRetrievalStrategy
    
    # Data tracking
    indexed_documents: List[Dict]
    log_buffer: List[str]
    
    # UI components
    root: tk.Tk
    strategy_dropdown: ttk.Combobox
    text_input: ScrolledText
    results_display: ScrolledText
    status_bar: StatusBar
```

### Widget Organization

**Sections** (top to bottom):
1. **Menu Bar**: File, Tools, Help
2. **Strategy Section**: Dropdown and load button
3. **Config Preview**: Shows loaded strategy config
4. **Text Indexing**: Text input and index button
5. **File Indexing**: File browser and index button
6. **Query Section**: Query input, Top-K, retrieve button
7. **Results Display**: Scrollable results
8. **Status Bar**: Status, document count, chunk count

### Event Handling

**Event Flow**:
```
User Action → Event Handler → Background Thread → Callback → GUI Update
```

**Example**:
```python
# User clicks "Index Text"
def _index_text(self):
    # 1. Validate input
    # 2. Disable button
    # 3. Start background thread
    thread = threading.Thread(target=index_operation, daemon=True)
    thread.start()

def index_operation():
    # 4. Perform indexing
    # 5. Call GUI update via safe_gui_update
    safe_gui_update(self.root, self._on_index_complete, ...)
```

### State Management

**Button States**:
- Managed by `_update_button_states()`
- Called after every operation
- Based on current state (strategy loaded, text present, etc.)

**Data State**:
- `indexed_documents`: List of indexed documents
- `current_strategy_name`: Currently loaded strategy
- Counters: Tracked in status bar

---

## Threading Model

### Background Operations

**Why Threading?**
- Prevents GUI freezing during long operations
- Indexing and retrieval can take seconds
- Database operations are I/O bound

**Threading Pattern**:
```python
def _index_text(self):
    def index_operation():
        try:
            # Perform async operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.indexing_strategy.process(...)
                )
            finally:
                loop.close()
            
            # Update GUI from main thread
            safe_gui_update(
                self.root,
                self._on_index_complete,
                result
            )
        except Exception as e:
            safe_gui_update(
                self.root,
                self._on_index_error,
                str(e)
            )
    
    # Start daemon thread
    thread = threading.Thread(target=index_operation, daemon=True)
    thread.start()
```

### GUI Updates from Threads

**CRITICAL**: Never update GUI directly from background thread!

**Wrong**:
```python
# DON'T DO THIS!
def background_operation():
    result = do_work()
    self.results_display.set_text(result)  # WRONG!
```

**Correct**:
```python
# DO THIS!
def background_operation():
    result = do_work()
    safe_gui_update(
        self.root,
        self._on_complete,
        result
    )

def _on_complete(self, result):
    # This runs on main thread
    self.results_display.set_text(result)
```

### Thread Safety Patterns

**safe_gui_update Utility**:
```python
def safe_gui_update(root, callback, *args, **kwargs):
    """Schedule GUI update on main thread."""
    root.after(0, lambda: callback(*args, **kwargs))
```

**Async Execution**:
```python
# Create new event loop for each operation
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    result = loop.run_until_complete(async_operation())
finally:
    loop.close()  # Always cleanup
```

### Common Pitfalls

1. **Updating GUI from background thread**: Use `safe_gui_update`
2. **Sharing event loops**: Create new loop per operation
3. **Not cleaning up loops**: Always use try/finally
4. **Blocking main thread**: Move long operations to background
5. **Race conditions**: Use proper state management

---

## Adding Features

### Adding a New Button

**Step 1**: Add button to UI
```python
def _create_my_section(self, parent, row):
    frame = ttk.LabelFrame(parent, text="My Feature", padding="10")
    frame.grid(row=row, column=0, sticky="ew", pady=(10, 0))
    
    self.my_button = ttk.Button(
        frame,
        text="Do Something",
        command=self._do_something
    )
    self.my_button.pack(side=tk.LEFT)
    
    # Add tooltip
    self._create_tooltip(self.my_button, "Does something useful")
    
    return row + 1
```

**Step 2**: Implement handler
```python
def _do_something(self):
    # Disable button
    self.my_button.config(state=tk.DISABLED)
    
    # Start background operation
    def operation():
        try:
            result = perform_work()
            safe_gui_update(
                self.root,
                self._on_something_complete,
                result
            )
        except Exception as e:
            safe_gui_update(
                self.root,
                self._on_something_error,
                str(e)
            )
    
    thread = threading.Thread(target=operation, daemon=True)
    thread.start()
```

**Step 3**: Add completion handlers
```python
def _on_something_complete(self, result):
    self.status_bar.set_status("success", "Operation complete")
    self._update_button_states()

def _on_something_error(self, error_msg):
    self.status_bar.set_status("error", "Operation failed")
    messagebox.showerror("Error", error_msg)
    self._update_button_states()
```

### Adding a New Workflow

**Example**: Add document export feature

1. **Add UI**: Export button in utility section
2. **Add Handler**: `_export_documents()`
3. **Add Logic**: Gather documents, format, save to file
4. **Add Tests**: Unit and integration tests
5. **Update Docs**: Add to user guide

### Extension Guidelines

**Best Practices**:
1. Follow existing patterns
2. Use threading for long operations
3. Add tooltips to new controls
4. Update button states appropriately
5. Add comprehensive error handling
6. Write tests for new features
7. Update documentation

**Code Style**:
- Follow PEP 8
- Use type hints
- Add docstrings
- Keep methods focused
- Use descriptive names

---

## Testing Strategy

### Unit Testing Approach

**What to Test**:
- Widget creation
- Event handlers
- Button state logic
- Error handling
- Utility functions

**Mocking Strategy**:
```python
@pytest.fixture
def gui_app(temp_config_files):
    with patch('rag_factory.gui.main_window.ServiceRegistry'), \
         patch('rag_factory.gui.main_window.StrategyPairManager'):
        app = RAGFactoryGUI(**temp_config_files)
        yield app
        app.root.destroy()
```

**Example Test**:
```python
def test_index_text_with_empty_input(gui_app):
    app = gui_app
    app.text_input.clear()
    
    initial_count = len(app.indexed_documents)
    app._index_text()
    
    assert len(app.indexed_documents) == initial_count
```

### Integration Testing Approach

**What to Test**:
- End-to-end workflows
- Real backend integration
- Threading behavior
- Error scenarios

**Example Test**:
```python
@pytest.mark.integration
def test_complete_workflow(gui_app):
    # Load strategy
    app._load_strategy("test-strategy")
    
    # Index document
    app.text_input.set_text("Test content")
    app._index_text()
    time.sleep(0.5)  # Wait for thread
    
    # Query
    app.query_var.set("test")
    app._retrieve()
    time.sleep(0.5)
    
    # Verify results
    assert len(app.results_display.get_text()) > 0
```

### Test Fixtures

**Common Fixtures**:
- `temp_config_files`: Temporary config files
- `gui_app`: GUI app with mocked backend
- `gui_app_with_strategy`: GUI with loaded strategy

### Running Tests

```bash
# All GUI tests
pytest tests/unit/gui/ tests/integration/gui/ -v

# Unit tests only
pytest tests/unit/gui/ -v

# Integration tests only
pytest tests/integration/gui/ -v -m integration

# With coverage
pytest tests/unit/gui/ --cov=rag_factory.gui --cov-report=html
```

---

## Debugging Tips

### Viewing Logs

**GUI Log Viewer**:
1. Tools → View Logs
2. Click "Refresh" to update
3. Look for ERROR or WARNING entries

**Log Files**:
- Application logs: Check console output
- Database logs: Check PostgreSQL logs

### Common Issues

**GUI Not Updating**:
- Check if using `safe_gui_update`
- Verify callback is called
- Check for exceptions in background thread

**Threading Deadlock**:
- Check for blocking operations on main thread
- Verify event loop cleanup
- Look for circular dependencies

**Memory Leaks**:
- Check for unclosed windows
- Verify thread cleanup
- Monitor with memory profiler:
  ```bash
  python -m memory_profiler gui_script.py
  ```

### Debugging Threading

**Add Debug Logging**:
```python
import threading

def background_operation():
    logger.debug(f"Thread {threading.current_thread().name} started")
    # ... operation ...
    logger.debug(f"Thread {threading.current_thread().name} completed")
```

**Check Thread Count**:
```python
import threading
print(f"Active threads: {threading.active_count()}")
print(f"Thread list: {threading.enumerate()}")
```

### Performance Profiling

**Profile Indexing**:
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Perform indexing
app._index_text()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Debugging Tips

1. **Enable Debug Logging**:
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Use Breakpoints**:
   ```python
   import pdb; pdb.set_trace()
   ```

3. **Check Event Loop**:
   ```python
   print(f"Event loop: {asyncio.get_event_loop()}")
   ```

4. **Monitor Database**:
   ```sql
   SELECT * FROM pg_stat_activity;
   ```

5. **Test in Isolation**:
   - Test components separately
   - Use mocks to isolate issues
   - Add print statements liberally

---

## Contributing

### Code Review Checklist

- [ ] Follows existing patterns
- [ ] Includes tests
- [ ] Updates documentation
- [ ] Handles errors gracefully
- [ ] Uses threading correctly
- [ ] Adds tooltips to new controls
- [ ] Updates button states
- [ ] Passes all tests

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Documentation
- [ ] User guide updated
- [ ] Developer guide updated
- [ ] Code comments added
```

---

**Version**: 1.0.0  
**Last Updated**: 2024-12-19
