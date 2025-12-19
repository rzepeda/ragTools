# Story 18.2: Implement Core GUI Framework (tkinter)

**Story ID:** 18.2  
**Epic:** Epic 18 - Minimal GUI for RAG Strategy Testing  
**Story Points:** 13  
**Priority:** Critical  
**Dependencies:** Story 18.1 (Design GUI Layout)

---

## User Story

**As a** developer  
**I want** the basic GUI window and component layout  
**So that** I have a foundation for adding functionality

---

## Detailed Requirements

### Functional Requirements

1. **Main Window Creation**
   - Create main window with title "RAG Factory - Strategy Pair Tester"
   - Set default dimensions (900x800px)
   - Set minimum dimensions (800x600px)
   - Window is resizable
   - Window can be closed cleanly

2. **UI Component Implementation**
   - Strategy dropdown (Combobox)
   - Reload Configs button
   - Configuration preview textbox (read-only, scrollable)
   - Text indexing textbox (editable, scrollable)
   - Index Text button
   - File path entry field
   - Browse button
   - Index File button
   - Query entry field
   - Top K dropdown
   - Retrieve button
   - Results textbox (read-only, scrollable)
   - Status bar label
   - Utility buttons (Clear All Data, View Logs, Settings, Help)

3. **Component Layout**
   - Use grid layout for precise positioning
   - Proper spacing and padding between components
   - Scrollbars attached to appropriate textboxes
   - Components properly aligned
   - Responsive layout (adapts to window resizing)

4. **Event Binding**
   - Button click handlers
   - Dropdown selection handlers
   - Keyboard event handlers
   - Window close handler

5. **Button State Management**
   - Buttons enable/disable based on application state
   - Update button states when text is entered
   - Update button states when strategy is loaded

### Non-Functional Requirements

1. **Cross-Platform Compatibility**
   - Works on Windows
   - Works on Linux
   - Works on macOS
   - Consistent appearance across platforms

2. **Performance**
   - Window launches in <2 seconds
   - UI remains responsive during operations
   - No lag when typing in textboxes

3. **Code Quality**
   - Clean, well-organized code structure
   - Proper separation of concerns
   - Documented methods
   - Follows Python best practices

4. **Maintainability**
   - Easy to add new components
   - Easy to modify layout
   - Clear method naming
   - Modular design

---

## Acceptance Criteria

### AC1: Main Window
- [ ] Window created with correct title
- [ ] Default dimensions set to 900x800px
- [ ] Minimum dimensions set to 800x600px
- [ ] Window is resizable
- [ ] Window closes cleanly without errors

### AC2: UI Components Created
- [ ] Strategy dropdown implemented
- [ ] Reload Configs button implemented
- [ ] Configuration preview textbox implemented with scrollbar
- [ ] Text indexing textbox implemented with scrollbar
- [ ] Index Text button implemented
- [ ] File path entry field implemented
- [ ] Browse button implemented
- [ ] Index File button implemented
- [ ] Query entry field implemented
- [ ] Top K dropdown implemented
- [ ] Retrieve button implemented
- [ ] Results textbox implemented with scrollbar
- [ ] Status bar implemented
- [ ] All utility buttons implemented

### AC3: Component Layout
- [ ] All components properly positioned using grid layout
- [ ] Proper spacing between components
- [ ] Scrollbars attached and functional
- [ ] Components aligned correctly
- [ ] Layout adapts to window resizing
- [ ] Row and column weights configured for resizing

### AC4: Event Binding
- [ ] All button click handlers bound
- [ ] Strategy dropdown selection handler bound
- [ ] Text input handlers bound for button state updates
- [ ] Window close handler bound

### AC5: Button State Management
- [ ] Index Text button disabled when no text or no strategy
- [ ] Index File button disabled when no file path or no strategy
- [ ] Retrieve button disabled when no query or no strategy
- [ ] Button states update correctly on user input
- [ ] Button states update correctly when strategy loaded

### AC6: Cross-Platform Testing
- [ ] Tested on Windows (if available)
- [ ] Tested on Linux
- [ ] Tested on macOS (if available)
- [ ] Appearance consistent across platforms
- [ ] No platform-specific bugs

### AC7: Application Lifecycle
- [ ] Application launches without errors
- [ ] Application can be launched from command line
- [ ] Application exits cleanly
- [ ] No resource leaks on exit

---

## Technical Specifications

### File Structure

```
rag_factory/
├── gui/
│   ├── __init__.py
│   └── main_window.py          # Main GUI implementation
│
scripts/
└── launch_gui.py               # Entry point script
```

### Main Window Implementation

```python
# rag_factory/gui/main_window.py
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class RAGFactoryGUI:
    """Main GUI window for RAG Factory Strategy Pair Tester."""
    
    def __init__(self):
        """Initialize the GUI."""
        # Initialize tkinter
        self.root = tk.Tk()
        self.root.title("RAG Factory - Strategy Pair Tester")
        self.root.geometry("900x800")
        self.root.minsize(800, 600)
        
        # Initialize state
        self.current_strategy: Optional[str] = None
        self.strategy_manager = None
        self.indexing_pipeline = None
        self.retrieval_pipeline = None
        self.service_registry = None
        
        # Build UI
        self._create_widgets()
        self._layout_widgets()
        self._bind_events()
        
    def _create_widgets(self):
        """Create all UI widgets."""
        # [1] Strategy Selection
        self.strategy_label = ttk.Label(self.root, text="Strategy Pair:")
        self.strategy_dropdown = ttk.Combobox(self.root, state="readonly", width=30)
        self.reload_button = ttk.Button(self.root, text="Reload Configs")
        
        # [2] Configuration Preview
        self.config_label = ttk.Label(self.root, text="Configuration Preview (Read-Only)")
        self.config_textbox = tk.Text(
            self.root, 
            state="disabled", 
            height=8, 
            font=("Courier", 10),
            wrap="none"
        )
        self.config_scrollbar_y = ttk.Scrollbar(
            self.root, 
            orient="vertical", 
            command=self.config_textbox.yview
        )
        self.config_scrollbar_x = ttk.Scrollbar(
            self.root, 
            orient="horizontal", 
            command=self.config_textbox.xview
        )
        self.config_textbox.config(
            yscrollcommand=self.config_scrollbar_y.set,
            xscrollcommand=self.config_scrollbar_x.set
        )
        
        # [3] Text Indexing
        self.text_index_label = ttk.Label(self.root, text="Text to Index:")
        self.text_to_index = tk.Text(self.root, height=4, wrap="word")
        self.text_scrollbar = ttk.Scrollbar(
            self.root, 
            orient="vertical", 
            command=self.text_to_index.yview
        )
        self.text_to_index.config(yscrollcommand=self.text_scrollbar.set)
        self.index_text_button = ttk.Button(self.root, text="Index Text", state="disabled")
        
        # [4] File Indexing
        self.file_label = ttk.Label(self.root, text="File Path:")
        self.file_path_entry = ttk.Entry(self.root, width=50)
        self.browse_button = ttk.Button(self.root, text="Browse")
        self.index_file_button = ttk.Button(self.root, text="Index File", state="disabled")
        
        # [5] Query & Retrieval
        self.query_label = ttk.Label(self.root, text="Query:")
        self.query_entry = ttk.Entry(self.root, width=50)
        self.topk_label = ttk.Label(self.root, text="Top K:")
        self.topk_dropdown = ttk.Combobox(
            self.root, 
            values=[1, 3, 5, 10, 20], 
            state="readonly",
            width=5
        )
        self.topk_dropdown.set(5)
        self.retrieve_button = ttk.Button(self.root, text="Retrieve", state="disabled")
        
        self.results_label = ttk.Label(self.root, text="Results:")
        self.results_textbox = tk.Text(
            self.root, 
            state="disabled", 
            height=10, 
            font=("Courier", 10),
            wrap="word"
        )
        self.results_scrollbar = ttk.Scrollbar(
            self.root, 
            orient="vertical", 
            command=self.results_textbox.yview
        )
        self.results_textbox.config(yscrollcommand=self.results_scrollbar.set)
        
        # [6] Status Bar
        self.status_bar = ttk.Label(
            self.root, 
            text="⚫ Ready", 
            relief="sunken", 
            anchor="w"
        )
        
        # Bottom buttons
        self.clear_button = ttk.Button(self.root, text="Clear All Data")
        self.logs_button = ttk.Button(self.root, text="View Logs")
        self.settings_button = ttk.Button(self.root, text="Settings")
        self.help_button = ttk.Button(self.root, text="Help")
        
    def _layout_widgets(self):
        """Layout all widgets using grid."""
        # Configure grid weights for resizing
        self.root.grid_rowconfigure(2, weight=1)  # Config preview
        self.root.grid_rowconfigure(4, weight=1)  # Text indexing
        self.root.grid_rowconfigure(11, weight=2)  # Results
        self.root.grid_columnconfigure(1, weight=1)
        
        # Section 1: Strategy Selection (row 0)
        self.strategy_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.strategy_dropdown.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.reload_button.grid(row=0, column=2, sticky="e", padx=10, pady=5)
        
        # Section 2: Config Preview (row 1-2)
        self.config_label.grid(row=1, column=0, columnspan=3, sticky="w", padx=10, pady=5)
        self.config_textbox.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
        self.config_scrollbar_y.grid(row=2, column=2, sticky="ns", pady=5)
        self.config_scrollbar_x.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10)
        
        # Section 3: Text Indexing (row 4-6)
        self.text_index_label.grid(row=4, column=0, columnspan=3, sticky="w", padx=10, pady=5)
        self.text_to_index.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
        self.text_scrollbar.grid(row=5, column=2, sticky="ns", pady=5)
        self.index_text_button.grid(row=6, column=1, sticky="e", padx=5, pady=5)
        
        # Section 4: File Indexing (row 7-8)
        self.file_label.grid(row=7, column=0, sticky="w", padx=10, pady=5)
        self.file_path_entry.grid(row=7, column=1, sticky="ew", padx=5, pady=5)
        self.browse_button.grid(row=7, column=2, sticky="e", padx=10, pady=5)
        self.index_file_button.grid(row=8, column=1, sticky="e", padx=5, pady=5)
        
        # Section 5: Query & Retrieval (row 9-11)
        self.query_label.grid(row=9, column=0, sticky="w", padx=10, pady=5)
        self.query_entry.grid(row=9, column=1, sticky="ew", padx=5, pady=5)
        
        self.topk_label.grid(row=10, column=0, sticky="w", padx=10, pady=5)
        self.topk_dropdown.grid(row=10, column=1, sticky="w", padx=5, pady=5)
        self.retrieve_button.grid(row=10, column=2, sticky="e", padx=10, pady=5)
        
        self.results_label.grid(row=11, column=0, columnspan=3, sticky="w", padx=10, pady=5)
        self.results_textbox.grid(row=12, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
        self.results_scrollbar.grid(row=12, column=2, sticky="ns", pady=5)
        
        # Section 6: Status Bar (row 13)
        self.status_bar.grid(row=13, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        
        # Bottom buttons (row 14)
        self.clear_button.grid(row=14, column=0, sticky="w", padx=10, pady=10)
        self.logs_button.grid(row=14, column=1, sticky="w", padx=5, pady=10)
        self.settings_button.grid(row=14, column=1, sticky="e", padx=5, pady=10)
        self.help_button.grid(row=14, column=2, sticky="e", padx=10, pady=10)
        
    def _bind_events(self):
        """Bind event handlers."""
        # Button commands
        self.reload_button.config(command=self.on_reload_configs)
        self.browse_button.config(command=self.on_browse_file)
        self.index_text_button.config(command=self.on_index_text)
        self.index_file_button.config(command=self.on_index_file)
        self.retrieve_button.config(command=self.on_retrieve)
        self.clear_button.config(command=self.on_clear_data)
        self.logs_button.config(command=self.on_view_logs)
        self.settings_button.config(command=self.on_settings)
        self.help_button.config(command=self.on_show_help)
        
        # Dropdown selection
        self.strategy_dropdown.bind("<<ComboboxSelected>>", self.on_strategy_selected)
        
        # Text input handlers for button state updates
        self.text_to_index.bind("<KeyRelease>", lambda e: self.update_button_states())
        self.query_entry.bind("<KeyRelease>", lambda e: self.update_button_states())
        self.file_path_entry.bind("<KeyRelease>", lambda e: self.update_button_states())
        
        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def update_button_states(self):
        """Enable/disable buttons based on current state."""
        has_strategy = self.current_strategy is not None
        has_text = len(self.text_to_index.get("1.0", "end").strip()) > 0
        has_file = len(self.file_path_entry.get().strip()) > 0
        has_query = len(self.query_entry.get().strip()) > 0
        
        # Update button states
        self.index_text_button.config(
            state="normal" if (has_strategy and has_text) else "disabled"
        )
        self.index_file_button.config(
            state="normal" if (has_strategy and has_file) else "disabled"
        )
        self.retrieve_button.config(
            state="normal" if (has_strategy and has_query) else "disabled"
        )
        
    # Placeholder event handlers (to be implemented in later stories)
    def on_strategy_selected(self, event=None):
        """Handler for strategy dropdown selection."""
        logger.info("Strategy selected (placeholder)")
        
    def on_reload_configs(self):
        """Handler for Reload Configs button."""
        logger.info("Reload configs (placeholder)")
        
    def on_browse_file(self):
        """Handler for Browse button."""
        logger.info("Browse file (placeholder)")
        
    def on_index_text(self):
        """Handler for Index Text button."""
        logger.info("Index text (placeholder)")
        
    def on_index_file(self):
        """Handler for Index File button."""
        logger.info("Index file (placeholder)")
        
    def on_retrieve(self):
        """Handler for Retrieve button."""
        logger.info("Retrieve (placeholder)")
        
    def on_clear_data(self):
        """Handler for Clear All Data button."""
        logger.info("Clear data (placeholder)")
        
    def on_view_logs(self):
        """Handler for View Logs button."""
        logger.info("View logs (placeholder)")
        
    def on_settings(self):
        """Handler for Settings button."""
        messagebox.showinfo("Settings", "Settings dialog (future enhancement)")
        
    def on_show_help(self):
        """Handler for Help button."""
        logger.info("Show help (placeholder)")
        
    def on_close(self):
        """Handler for window close."""
        self.root.destroy()
        
    def run(self):
        """Start the GUI event loop."""
        self.root.mainloop()


# Entry point
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = RAGFactoryGUI()
    app.run()
```

### Launch Script

```python
# scripts/launch_gui.py
#!/usr/bin/env python3
"""Launch script for RAG Factory GUI."""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_factory.gui.main_window import RAGFactoryGUI

def main():
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run GUI
    app = RAGFactoryGUI()
    app.run()

if __name__ == "__main__":
    main()
```

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/gui/test_main_window.py
import pytest
import tkinter as tk
from rag_factory.gui.main_window import RAGFactoryGUI

class TestMainWindow:
    """Unit tests for main window."""
    
    def test_window_creation(self):
        """Test window is created with correct properties."""
        app = RAGFactoryGUI()
        
        assert app.root.title() == "RAG Factory - Strategy Pair Tester"
        assert app.root.winfo_width() >= 800
        assert app.root.winfo_height() >= 600
        
        app.root.destroy()
        
    def test_all_widgets_created(self):
        """Test all widgets are created."""
        app = RAGFactoryGUI()
        
        # Check key widgets exist
        assert app.strategy_dropdown is not None
        assert app.config_textbox is not None
        assert app.text_to_index is not None
        assert app.query_entry is not None
        assert app.results_textbox is not None
        assert app.status_bar is not None
        
        app.root.destroy()
        
    def test_button_states_initial(self):
        """Test initial button states."""
        app = RAGFactoryGUI()
        
        # Initially, action buttons should be disabled
        assert str(app.index_text_button['state']) == 'disabled'
        assert str(app.index_file_button['state']) == 'disabled'
        assert str(app.retrieve_button['state']) == 'disabled'
        
        app.root.destroy()
        
    def test_button_state_updates(self):
        """Test button states update correctly."""
        app = RAGFactoryGUI()
        
        # Simulate strategy loaded
        app.current_strategy = "test-strategy"
        
        # Add text
        app.text_to_index.insert("1.0", "Test text")
        app.update_button_states()
        
        # Index Text button should be enabled
        assert str(app.index_text_button['state']) == 'normal'
        
        app.root.destroy()
```

### Integration Tests

```python
# tests/integration/gui/test_gui_launch.py
import pytest
from rag_factory.gui.main_window import RAGFactoryGUI

def test_gui_launches_without_error():
    """Test GUI can be launched without errors."""
    app = RAGFactoryGUI()
    
    # Verify window is created
    assert app.root is not None
    
    # Clean up
    app.root.destroy()
```

---

## Story Points Breakdown

- **Widget Creation:** 4 points
- **Layout Implementation:** 3 points
- **Event Binding:** 2 points
- **Button State Management:** 2 points
- **Testing:** 2 points

**Total:** 13 points

---

## Dependencies

- Story 18.1 (Design GUI Layout) - MUST BE COMPLETED
- Python 3.10+
- tkinter (built into Python)

---

## Notes

- This story implements the GUI shell without backend integration
- All event handlers are placeholders (will be implemented in later stories)
- Focus is on layout and component creation
- Backend integration happens in Story 18.3
- Keep code clean and well-documented for future enhancements
