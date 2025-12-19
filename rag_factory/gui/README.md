# RAG Factory GUI Module

## Overview

The GUI module provides a minimal tkinter-based graphical user interface for testing RAG strategy pairs. It allows developers to visually interact with the RAG Factory system without writing code.

## Features

- **Strategy Selection**: Load and preview strategy pair configurations from YAML files
- **Text Indexing**: Index text content directly from the GUI
- **File Indexing**: Browse and index files from the filesystem
- **Query & Retrieval**: Query indexed data and view formatted results
- **Status Monitoring**: Real-time status updates, document/chunk counts, and operation timing
- **Data Management**: Clear indexed data with confirmation dialogs

## Usage

### Launching the GUI

From the command line:

```bash
# Activate virtual environment
source venv/bin/activate

# Launch with default configuration
rag-factory gui

# Launch with custom configuration
rag-factory gui --config my_config.yaml --strategies my_strategies/
```

### Basic Workflow

1. **Select a Strategy**: Choose a strategy pair from the dropdown menu
2. **Index Data**: Use either the text input or file browser to index content
3. **Query**: Enter a query and click "Retrieve" to search indexed data
4. **View Results**: Results appear in the results panel with scores and sources

## Component Architecture

### Main Window (`main_window.py`)

The `RAGFactoryGUI` class implements the complete window layout with 6 main sections:

1. **Strategy Selection**: Dropdown and reload button
2. **Configuration Preview**: Read-only YAML display
3. **Text Indexing**: Multiline text input and index button
4. **File Indexing**: File path entry, browse button, and index button
5. **Query & Retrieval**: Query input, top-k selector, and results display
6. **Status Bar**: Status indicator, counters, and timestamp

### Reusable Components (`components/`)

- **StatusBar**: Displays operation status with color-coded indicators
- **ScrolledText**: Text widget with automatic scrollbars

### Utilities (`utils/`)

- **threading.py**: Functions for running async operations in background threads
- **formatters.py**: Functions for formatting YAML, results, and errors

## Threading Model

All backend operations (strategy loading, indexing, retrieval) run in background threads to prevent GUI freezing. The threading utilities ensure:

- Async operations are wrapped with `asyncio.run()` in threads
- GUI updates from background threads use `root.after(0, callback)`
- Thread-safe state management

## Extension Guidelines

### Adding New UI Components

1. Create component class in `components/` directory
2. Inherit from `ttk.Frame` for consistency
3. Export from `components/__init__.py`
4. Use in `main_window.py`

### Adding New Utility Functions

1. Add function to appropriate module in `utils/`
2. Export from `utils/__init__.py`
3. Import in `main_window.py`

### Modifying the Layout

The main window uses tkinter's grid layout manager. To modify:

1. Locate the relevant `_create_*_section()` method in `main_window.py`
2. Adjust grid row/column positions
3. Update grid weights for proper resizing behavior
4. Test window resizing to ensure minimum size constraints work

## Best Practices

### Maintaining Consistency

- Use ttk widgets (not tk widgets) for modern appearance
- Follow the existing padding and spacing patterns (10px sections, 5px widgets)
- Use monospace fonts for code/data display
- Keep button states synchronized with application state

### Error Handling

- Always show errors in both status bar and popup dialogs
- Use appropriate status indicators (⚫ Ready / 🟢 Success / 🔴 Error)
- Provide clear, actionable error messages

### Performance

- Run all I/O operations in background threads
- Use `safe_gui_update()` for all GUI updates from threads
- Keep GUI responsive by avoiding blocking operations in main thread

## Keyboard Shortcuts

- **Ctrl+Q**: Quit application
- **F1**: Show help dialog

## Known Limitations

- Currently uses simulated indexing/retrieval (not connected to real pipelines)
- Log viewer is a placeholder
- Settings dialog is a placeholder
- No undo/redo for data operations

## Future Enhancements

See Story 18.2 and beyond for planned enhancements:
- Real pipeline integration
- Progress bars for long operations
- Log viewer implementation
- Settings dialog for configuration
- Export/import functionality
- Batch operations support
