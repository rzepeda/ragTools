Epic 18: Minimal GUI for RAG Strategy Testing
Epic Goal: Create a lightweight, single-window GUI application for testing RAG strategy pairs with minimal dependencies, enabling visual validation of indexing and retrieval workflows without requiring CLI knowledge.
Epic Story Points Total: 34
Dependencies:

Epic 17 (Strategy Pair Configuration - COMPLETED ✅)
Epic 14 (CLI Enhancements - for configuration validation)

Status: Ready for implementation

Background
The RAG Factory currently has:

✅ Complete strategy pair system (Epic 17)
✅ CLI tools for testing (Epic 8.5, Epic 14)
✅ Service registry and dependency injection (Epic 11)
✅ Pipeline validation (Epic 12)

The Problem:

CLI requires command-line comfort
Non-technical users need visual feedback
Quick demos need a simple interface
Development testing benefits from visual validation

The Solution:
A minimal, single-window GUI that wraps the existing StrategyPairManager from Epic 17, providing:

Strategy pair selection
Live configuration preview
Text and file indexing
Query interface with results display
Status feedback and error messages

Design Philosophy:

Lightweight: Single file, minimal dependencies (tkinter built into Python)
Development Tool: Not production-ready, focused on testing
Delegates to Library: Uses StrategyPairManager, not reimplementing logic
Read-Only Configuration: Shows config but doesn't edit it (use text editor for that)


Story 18.1: Design GUI Layout and Component Specification
As a developer
I want a detailed GUI layout specification
So that implementation is straightforward and consistent
Acceptance Criteria:

Define complete window layout with dimensions
Specify all UI components (dropdowns, textboxes, buttons)
Define component behavior (enable/disable states)
Define status feedback mechanism
Define error display strategy
Wireframe diagram
Component interaction flow diagram
Documentation for extending the GUI

Window Layout Specification:
┌─────────────────────────────────────────────────────────────────────┐
│ RAG Factory - Strategy Pair Tester                          [_][□][X]│
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ [1] Strategy Selection                                                │
│ ┌───────────────────────────────────────────────────────────────┐   │
│ │ Strategy Pair: [semantic-local-pair ▼]  [Reload Configs]     │   │
│ └───────────────────────────────────────────────────────────────┘   │
│                                                                       │
│ [2] Configuration Preview (Read-Only)                                 │
│ ┌───────────────────────────────────────────────────────────────┐   │
│ │ strategy_name: "semantic-local-pair"                          │   │
│ │ version: "1.0.0"                                              │   │
│ │ indexer:                                                      │   │
│ │   strategy: "VectorEmbeddingIndexer"                          │   │
│ │   services: {embedding: "$embedding_local", ...}              │   │
│ │ retriever:                                                    │   │
│ │   strategy: "SemanticRetriever"                               │   │
│ │   ...                                                         │   │
│ │                                    [scrollbar]                │   │
│ └───────────────────────────────────────────────────────────────┘   │
│                                                                       │
│ [3] Text Indexing                                                     │
│ ┌───────────────────────────────────────────────────────────────┐   │
│ │ Text to Index:                                                │   │
│ │ ┌─────────────────────────────────────────────────────────┐   │   │
│ │ │ Type or paste text here...                              │   │   │
│ │ │                                            [scrollbar]   │   │   │
│ │ │                                                          │   │   │
│ │ └─────────────────────────────────────────────────────────┘   │   │
│ │                                      [Index Text]             │   │
│ └───────────────────────────────────────────────────────────────┘   │
│                                                                       │
│ [4] File Indexing                                                     │
│ ┌───────────────────────────────────────────────────────────────┐   │
│ │ File Path: [/path/to/file.txt                    ] [Browse]   │   │
│ │                                      [Index File]             │   │
│ └───────────────────────────────────────────────────────────────┘   │
│                                                                       │
│ [5] Query & Retrieval                                                 │
│ ┌───────────────────────────────────────────────────────────────┐   │
│ │ Query: [What is machine learning?                          ]   │   │
│ │                           [Retrieve] Top K: [5 ▼]             │   │
│ │                                                               │   │
│ │ Results:                                                      │   │
│ │ ┌─────────────────────────────────────────────────────────┐   │   │
│ │ │ 1. Score: 0.89                                          │   │   │
│ │ │    Machine learning is a subset of artificial           │   │   │
│ │ │    intelligence that enables systems to learn...        │   │   │
│ │ │    Source: machine_learning.txt                         │   │   │
│ │ │                                                          │   │   │
│ │ │ 2. Score: 0.76                                          │   │   │
│ │ │    Types of Machine Learning: 1. Supervised...          │   │   │
│ │ │    Source: machine_learning.txt                         │   │   │
│ │ │                                            [scrollbar]   │   │   │
│ │ └─────────────────────────────────────────────────────────┘   │   │
│ └───────────────────────────────────────────────────────────────┘   │
│                                                                       │
│ [6] Status Bar                                                        │
│ ┌───────────────────────────────────────────────────────────────┐   │
│ │ ⚫ Ready | Documents: 3 | Chunks: 7 | Last action: 0.3s ago   │   │
│ └───────────────────────────────────────────────────────────────┘   │
│                                                                       │
│ [Clear All Data] [View Logs]                    [Settings] [Help]    │
└─────────────────────────────────────────────────────────────────────┘

Window Size: 900px (width) x 800px (height)
Font: System default, monospace for config/results
Colors: Light theme (white background, dark text)
Component Specifications:
pseudocodeCOMPONENT_SPECS:
  StrategyDropdown:
    type: Combobox (read-only)
    values: List of .yaml files from strategies/ directory
    default: "semantic-local-pair"
    on_change: Load selected strategy configuration
    
  ReloadConfigsButton:
    type: Button
    action: Rescan strategies/ directory and refresh dropdown
    
  ConfigPreviewTextbox:
    type: Text (multiline, read-only, scrollable)
    content: YAML content of selected strategy file
    font: Monospace (for alignment)
    height: 8 lines
    
  TextToIndexTextbox:
    type: Text (multiline, editable, scrollable)
    placeholder: "Type or paste text here..."
    height: 4 lines
    
  IndexTextButton:
    type: Button
    enabled_when: TextToIndexTextbox is not empty AND strategy loaded
    action: Call indexing_pipeline.index(text)
    
  FilePathTextbox:
    type: Entry (single-line)
    placeholder: "/path/to/file.txt"
    
  BrowseButton:
    type: Button
    action: Open file dialog, populate FilePathTextbox
    
  IndexFileButton:
    type: Button
    enabled_when: FilePathTextbox contains valid path AND strategy loaded
    action: Call indexing_pipeline.index(file_content)
    
  QueryTextbox:
    type: Entry (single-line)
    placeholder: "What is machine learning?"
    
  TopKDropdown:
    type: Combobox
    values: [1, 3, 5, 10, 20]
    default: 5
    
  RetrieveButton:
    type: Button
    enabled_when: QueryTextbox is not empty AND strategy loaded
    action: Call retrieval_pipeline.retrieve(query, top_k)
    
  ResultsTextbox:
    type: Text (multiline, read-only, scrollable)
    content: Formatted retrieval results
    font: Monospace for alignment
    height: 10 lines
    
  StatusBar:
    type: Label (bottom of window)
    sections:
      - Status indicator (⚫ Ready / 🟢 Success / 🔴 Error)
      - Document count
      - Chunk count  
      - Last action timestamp
    updates: After every operation
    
  ClearAllDataButton:
    type: Button (warning style)
    action: Confirm dialog → Clear database tables for current strategy
    
  ViewLogsButton:
    type: Button
    action: Open popup window with application logs
    
  SettingsButton:
    type: Button  
    action: Open settings dialog (future enhancement)
    
  HelpButton:
    type: Button
    action: Open help dialog with keyboard shortcuts
```

**Component Interaction Flow:**
```
USER WORKFLOW 1: Load Strategy and Index Text
  1. User selects strategy from dropdown
     → GUI calls: load_strategy(strategy_name)
     → ConfigPreview updates with YAML content
     → StatusBar: "⚫ Ready | Strategy: {name} loaded"
     
  2. User types text in TextToIndexTextbox
     → IndexTextButton becomes enabled
     
  3. User clicks IndexTextButton
     → GUI validates text is not empty
     → GUI calls: indexing_pipeline.index([{id, content}])
     → Progress indicator shows (optional)
     → StatusBar updates: "🟢 Indexed 1 document in 0.5s"
     → Document/chunk count updates

USER WORKFLOW 2: Index File
  1. User clicks BrowseButton
     → File dialog opens
     → User selects file
     → FilePathTextbox populates
     → IndexFileButton becomes enabled
     
  2. User clicks IndexFileButton
     → GUI reads file content
     → GUI calls: indexing_pipeline.index([{id, content}])
     → StatusBar updates with result

USER WORKFLOW 3: Query
  1. User types query in QueryTextbox
     → RetrieveButton becomes enabled
     
  2. User optionally changes TopK value
     
  3. User clicks RetrieveButton
     → GUI calls: retrieval_pipeline.retrieve(query, top_k)
     → ResultsTextbox populates with formatted results
     → StatusBar updates: "🟢 Retrieved 5 results in 0.3s"

ERROR HANDLING:
  - Invalid strategy YAML → Show error in StatusBar + popup
  - Missing services → Show error with missing service names
  - Missing migrations → Show error with upgrade command
  - File not found → Show error in StatusBar
  - Empty text/query → Disable buttons
  - Database connection error → Show error + retry option
Story Points: 5

Story 18.2: Implement Core GUI Framework (tkinter)
As a developer
I want the basic GUI window and component layout
So that I have a foundation for adding functionality
Acceptance Criteria:

Create main window with title and dimensions
Implement all UI components from Story 18.1 layout
Components are properly positioned and sized
Window is resizable (with minimum size)
All textboxes have scrollbars where needed
Status bar is functional (can update text)
Application can be launched from command line
Application exits cleanly
Cross-platform tested (Windows, Linux, macOS)

Pseudocode Structure:
pseudocodeCLASS RAGFactoryGUI:
  CONSTRUCTOR():
    # Initialize tkinter
    self.root = Tk()
    self.root.title("RAG Factory - Strategy Pair Tester")
    self.root.geometry("900x800")
    self.root.minsize(800, 600)
    
    # Initialize state
    self.current_strategy = None
    self.strategy_manager = None
    self.indexing_pipeline = None
    self.retrieval_pipeline = None
    
    # Build UI
    self._create_widgets()
    self._layout_widgets()
    self._bind_events()
    
  METHOD _create_widgets():
    # [1] Strategy Selection
    self.strategy_label = Label("Strategy Pair:")
    self.strategy_dropdown = Combobox(state="readonly")
    self.reload_button = Button("Reload Configs")
    
    # [2] Configuration Preview
    self.config_label = Label("Configuration Preview (Read-Only)")
    self.config_textbox = Text(state="disabled", height=8, font="Courier")
    self.config_scrollbar = Scrollbar()
    
    # [3] Text Indexing
    self.text_index_label = Label("Text to Index:")
    self.text_to_index = Text(height=4)
    self.text_scrollbar = Scrollbar()
    self.index_text_button = Button("Index Text", state="disabled")
    
    # [4] File Indexing
    self.file_label = Label("File Path:")
    self.file_path_entry = Entry()
    self.browse_button = Button("Browse")
    self.index_file_button = Button("Index File", state="disabled")
    
    # [5] Query & Retrieval
    self.query_label = Label("Query:")
    self.query_entry = Entry()
    self.topk_label = Label("Top K:")
    self.topk_dropdown = Combobox(values=[1,3,5,10,20], state="readonly")
    self.topk_dropdown.set(5)
    self.retrieve_button = Button("Retrieve", state="disabled")
    
    self.results_label = Label("Results:")
    self.results_textbox = Text(state="disabled", height=10, font="Courier")
    self.results_scrollbar = Scrollbar()
    
    # [6] Status Bar
    self.status_bar = Label("⚫ Ready", anchor="w")
    
    # Bottom buttons
    self.clear_button = Button("Clear All Data")
    self.logs_button = Button("View Logs")
    self.settings_button = Button("Settings")
    self.help_button = Button("Help")
    
  METHOD _layout_widgets():
    # Use grid layout for precise positioning
    # Section 1: Strategy Selection (row 0)
    self.strategy_label.grid(row=0, column=0)
    self.strategy_dropdown.grid(row=0, column=1)
    self.reload_button.grid(row=0, column=2)
    
    # Section 2: Config Preview (row 1-2)
    self.config_label.grid(row=1, column=0, columnspan=3)
    self.config_textbox.grid(row=2, column=0, columnspan=2)
    self.config_scrollbar.grid(row=2, column=2)
    
    # Section 3: Text Indexing (row 3-5)
    self.text_index_label.grid(row=3, column=0, columnspan=3)
    self.text_to_index.grid(row=4, column=0, columnspan=2)
    self.text_scrollbar.grid(row=4, column=2)
    self.index_text_button.grid(row=5, column=1)
    
    # Section 4: File Indexing (row 6-7)
    self.file_label.grid(row=6, column=0)
    self.file_path_entry.grid(row=6, column=1)
    self.browse_button.grid(row=6, column=2)
    self.index_file_button.grid(row=7, column=1)
    
    # Section 5: Query & Retrieval (row 8-11)
    self.query_label.grid(row=8, column=0)
    self.query_entry.grid(row=8, column=1)
    self.topk_label.grid(row=9, column=0)
    self.topk_dropdown.grid(row=9, column=1)
    self.retrieve_button.grid(row=9, column=2)
    
    self.results_label.grid(row=10, column=0, columnspan=3)
    self.results_textbox.grid(row=11, column=0, columnspan=2)
    self.results_scrollbar.grid(row=11, column=2)
    
    # Section 6: Status Bar (row 12)
    self.status_bar.grid(row=12, column=0, columnspan=3)
    
    # Bottom buttons (row 13)
    self.clear_button.grid(row=13, column=0)
    self.logs_button.grid(row=13, column=1)
    self.settings_button.grid(row=13, column=2)
    self.help_button.grid(row=13, column=3)
    
    # Configure grid weights for resizing
    FOR each_row IN [0..13]:
      self.root.grid_rowconfigure(each_row, weight=1)
    FOR each_col IN [0..3]:
      self.root.grid_columnconfigure(each_col, weight=1)
    
  METHOD _bind_events():
    # Connect UI events to handler methods
    self.strategy_dropdown.bind("<<ComboboxSelected>>", self.on_strategy_selected)
    self.reload_button.config(command=self.on_reload_configs)
    self.browse_button.config(command=self.on_browse_file)
    self.index_text_button.config(command=self.on_index_text)
    self.index_file_button.config(command=self.on_index_file)
    self.retrieve_button.config(command=self.on_retrieve)
    self.clear_button.config(command=self.on_clear_data)
    self.logs_button.config(command=self.on_view_logs)
    self.help_button.config(command=self.on_show_help)
    
    # Enable/disable buttons based on input
    self.text_to_index.bind("<KeyRelease>", self.update_button_states)
    self.query_entry.bind("<KeyRelease>", self.update_button_states)
    self.file_path_entry.bind("<KeyRelease>", self.update_button_states)
    
  METHOD update_button_states():
    # Enable/disable buttons based on current state
    has_strategy = self.current_strategy IS NOT None
    has_text = len(self.text_to_index.get()) > 0
    has_file = len(self.file_path_entry.get()) > 0 AND file_exists(self.file_path_entry.get())
    has_query = len(self.query_entry.get()) > 0
    
    self.index_text_button.config(state="normal" IF (has_strategy AND has_text) ELSE "disabled")
    self.index_file_button.config(state="normal" IF (has_strategy AND has_file) ELSE "disabled")
    self.retrieve_button.config(state="normal" IF (has_strategy AND has_query) ELSE "disabled")
    
  METHOD run():
    # Start the GUI event loop
    self.root.mainloop()
    
# Entry point
IF __name__ == "__main__":
  app = RAGFactoryGUI()
  app.run()
Technical Notes:

Use tkinter (comes with Python, no extra dependencies)
Use ttk for modern-looking widgets (Combobox, etc.)
Use grid() layout manager for precise control
Use pack() for scrollbars attached to textboxes
Configure row/column weights for proper resizing
Use monospace font (Courier or Consolas) for config/results

Story Points: 13

Story 18.3: Integrate StrategyPairManager (Backend Connection)
As a developer
I want the GUI to load and use strategy pairs from Epic 17
So that the GUI leverages existing functionality
Acceptance Criteria:

Initialize ServiceRegistry on GUI startup
Initialize StrategyPairManager with service registry
Load available strategy pairs from strategies/ directory
Populate strategy dropdown with available pairs
Load selected strategy configuration (YAML)
Display configuration in read-only textbox
Instantiate indexing and retrieval pipelines for selected strategy
Validate migrations before allowing operations
Handle missing services gracefully with user-friendly errors
Handle missing migrations with upgrade suggestions
All errors displayed in status bar and/or popup dialogs

Pseudocode:
pseudocodeCLASS RAGFactoryGUI:
  # ... existing code from Story 18.2 ...
  
  CONSTRUCTOR():
    # ... existing initialization ...
    
    # Initialize backend (Epic 17 components)
    TRY:
      self._initialize_backend()
      self._load_available_strategies()
    CATCH Exception AS e:
      self.show_error("Initialization failed: {e}")
      self.status_bar.config(text="🔴 Initialization failed")
    
  METHOD _initialize_backend():
    """Initialize service registry and strategy pair manager"""
    # Load service registry from config/services.yaml
    TRY:
      self.service_registry = ServiceRegistry("config/services.yaml")
      self.status_bar.config(text="⚫ Loading services...")
      
      # Verify required services are available
      required_services = ["db_main"]  # At minimum need database
      FOR service IN required_services:
        IF NOT service IN self.service_registry.list_services():
          THROW Exception("Missing required service: {service}")
      
    CATCH FileNotFoundError:
      THROW Exception("config/services.yaml not found. Please create service configuration.")
    CATCH Exception AS e:
      THROW Exception("Service registry error: {e}")
    
    # Initialize strategy pair manager
    TRY:
      self.strategy_manager = StrategyPairManager(
        service_registry=self.service_registry,
        config_dir="strategies/",
        alembic_config="alembic.ini"
      )
      self.status_bar.config(text="⚫ Ready")
      
    CATCH Exception AS e:
      THROW Exception("Strategy manager error: {e}")
  
  METHOD _load_available_strategies():
    """Scan strategies/ directory and populate dropdown"""
    strategy_files = []
    
    # Find all .yaml files in strategies/
    FOR file IN os.listdir("strategies/"):
      IF file.endswith(".yaml") AND file != "README.md":
        strategy_name = file.replace(".yaml", "")
        strategy_files.append(strategy_name)
    
    IF len(strategy_files) == 0:
      self.show_warning("No strategy pairs found in strategies/ directory")
      RETURN
    
    # Sort alphabetically
    strategy_files.sort()
    
    # Populate dropdown
    self.strategy_dropdown.config(values=strategy_files)
    
    # Select first strategy by default
    self.strategy_dropdown.set(strategy_files[0])
    self.on_strategy_selected()  # Trigger load
  
  METHOD on_strategy_selected(event=None):
    """Handler for strategy dropdown selection"""
    selected_strategy = self.strategy_dropdown.get()
    
    IF selected_strategy == "":
      RETURN
    
    self.status_bar.config(text="⚫ Loading strategy: {selected_strategy}...")
    
    TRY:
      # Load strategy configuration (YAML content)
      config_path = f"strategies/{selected_strategy}.yaml"
      config_yaml = read_file(config_path)
      
      # Display in config preview textbox
      self.config_textbox.config(state="normal")
      self.config_textbox.delete("1.0", "end")
      self.config_textbox.insert("1.0", config_yaml)
      self.config_textbox.config(state="disabled")
      
      # Instantiate strategy pair using StrategyPairManager
      self.indexing_pipeline, self.retrieval_pipeline = self.strategy_manager.load_pair(
        selected_strategy
      )
      
      self.current_strategy = selected_strategy
      self.status_bar.config(text=f"🟢 Strategy loaded: {selected_strategy}")
      
      # Enable operation buttons
      self.update_button_states()
      
    CATCH MigrationError AS e:
      # Missing migrations
      error_msg = f"Missing migrations for {selected_strategy}:\n{e.missing_revisions}"
      suggestion = "Run: alembic upgrade head"
      
      self.show_error(f"{error_msg}\n\nSuggestion: {suggestion}")
      self.status_bar.config(text=f"🔴 Migration error: {selected_strategy}")
      
      # Keep config visible but disable operations
      self.current_strategy = None
      self.update_button_states()
      
    CATCH ServiceMissingError AS e:
      # Missing services
      error_msg = f"Missing services for {selected_strategy}:\n{e.missing_services}"
      suggestion = "Check config/services.yaml"
      
      self.show_error(f"{error_msg}\n\nSuggestion: {suggestion}")
      self.status_bar.config(text=f"🔴 Service error: {selected_strategy}")
      
      self.current_strategy = None
      self.update_button_states()
      
    CATCH Exception AS e:
      self.show_error(f"Failed to load strategy: {e}")
      self.status_bar.config(text=f"🔴 Error loading: {selected_strategy}")
      
      self.current_strategy = None
      self.update_button_states()
  
  METHOD on_reload_configs():
    """Reload strategy configurations from disk"""
    self.status_bar.config(text="⚫ Reloading configurations...")
    
    TRY:
      # Reload service registry (in case services.yaml changed)
      self.service_registry.reload_all()
      
      # Rescan strategies directory
      self._load_available_strategies()
      
      self.status_bar.config(text="🟢 Configurations reloaded")
      
    CATCH Exception AS e:
      self.show_error(f"Reload failed: {e}")
      self.status_bar.config(text="🔴 Reload failed")
  
  METHOD show_error(message: str):
    """Display error in popup dialog"""
    messagebox.showerror("Error", message)
  
  METHOD show_warning(message: str):
    """Display warning in popup dialog"""
    messagebox.showwarning("Warning", message)
  
  METHOD show_info(message: str):
    """Display info in popup dialog"""
    messagebox.showinfo("Info", message)
Error Handling Strategy:
pseudocodeERROR_TYPES:
  MigrationError:
    - Caught from StrategyPairManager.load_pair()
    - Contains: list of missing revision IDs
    - Display: Error dialog + status bar
    - Action: Suggest "alembic upgrade head"
  
  ServiceMissingError:
    - Caught from StrategyPairManager.load_pair()
    - Contains: list of missing service names
    - Display: Error dialog + status bar
    - Action: Suggest checking config/services.yaml
  
  DatabaseConnectionError:
    - Caught from service registry initialization
    - Display: Error dialog with connection details
    - Action: Suggest checking DATABASE_URL in .env
  
  FileNotFoundError:
    - Caught from file indexing
    - Display: Status bar only (less critical)
    - Action: User can try different file
  
  ValidationError:
    - Caught from pipeline validation
    - Display: Error dialog with validation details
    - Action: Explain capability mismatch
Story Points: 8

Story 18.4: Implement Indexing Operations
As a user
I want to index text and files through the GUI
So that I can populate the database for testing retrieval
Acceptance Criteria:

Text indexing creates document from textbox content
File indexing reads file and creates document
Both operations call indexing_pipeline.index(documents)
Progress indication during indexing (optional spinner)
Success message with document/chunk count
Error handling for indexing failures
Status bar updates with operation result and timing
Document count updates after successful indexing
Chunk count updates after successful indexing
Thread safety (don't freeze GUI during indexing)

Pseudocode:
pseudocodeCLASS RAGFactoryGUI:
  # ... existing code ...
  
  CONSTRUCTOR():
    # ... existing initialization ...
    
    # Add tracking for indexed documents
    self.document_count = 0
    self.chunk_count = 0
  
  METHOD on_index_text():
    """Handler for Index Text button"""
    # Get text from textbox
    text_content = self.text_to_index.get("1.0", "end").strip()
    
    IF text_content == "":
      self.show_warning("Please enter some text to index")
      RETURN
    
    # Disable button during operation
    self.index_text_button.config(state="disabled")
    self.status_bar.config(text="⚫ Indexing text...")
    
    # Run indexing in background thread (to avoid freezing GUI)
    thread = Thread(target=self._index_text_worker, args=(text_content,))
    thread.start()
  
  METHOD _index_text_worker(text_content: str):
    """Background worker for text indexing"""
    start_time = time.time()
    
    TRY:
      # Create document from text
      document = {
        "id": generate_document_id(),  # UUID or timestamp-based
        "content": text_content,
        "source": "gui_text_input",
        "metadata": {
          "indexed_at": datetime.now().isoformat(),
          "source_type": "text"
        }
      }
      
      # Create temporary indexing context
      context = IndexingContext(
        database_service=self.service_registry.get("db_main"),
        config={}
      )
      
      # Call indexing pipeline (from Epic 17)
      result = self.indexing_pipeline.process([document], context)
      
      # Update tracking
      self.document_count += result.document_count
      self.chunk_count += result.chunk_count
      
      elapsed = time.time() - start_time
      
      # Update GUI on main thread
      self.root.after(0, self._on_index_text_success, result, elapsed)
      
    CATCH Exception AS e:
      # Update GUI on main thread
      self.root.after(0, self._on_index_text_error, str(e))
  
  METHOD _on_index_text_success(result: IndexingResult, elapsed: float):
    """Called on main thread after successful indexing"""
    # Re-enable button
    self.index_text_button.config(state="normal")
    
    # Update status bar
    status_text = (
      f"🟢 Indexed: {result.document_count} doc, "
      f"{result.chunk_count} chunks in {elapsed:.2f}s | "
      f"Total: {self.document_count} docs, {self.chunk_count} chunks"
    )
    self.status_bar.config(text=status_text)
    
    # Optional: Clear text input
    self.text_to_index.delete("1.0", "end")
    
    # Update button states
    self.update_button_states()
  
  METHOD _on_index_text_error(error_message: str):
    """Called on main thread after indexing error"""
    # Re-enable button
    self.index_text_button.config(state="normal")
    
    # Show error
    self.show_error(f"Indexing failed: {error_message}")
    self.status_bar.config(text="🔴 Indexing failed")
    
    # Update button states
    self.update_button_states()
  
  METHOD on_browse_file():
    """Handler for Browse button (file dialog)"""
    file_path = filedialog.askopenfilename(
      title="Select file to index",
      filetypes=[
        ("Text files", "*.txt"),
        ("Markdown files", "*.md"),
        ("All files", "*.*")
      ]
    )
    
    IF file_path != "":
      self.file_path_entry.delete(0, "end")
      self.file_path_entry.insert(0, file_path)
      
      # Update button states
      self.update_button_states()
  
  METHOD on_index_file():
    """Handler for Index File button"""
    file_path = self.file_path_entry.get().strip()
    
    IF file_path == "":
      self.show_warning("Please select a file to index")
      RETURN
    
    IF NOT os.path.exists(file_path):
      self.show_error(f"File not found: {file_path}")
      RETURN
    
    # Disable button during operation
    self.index_file_button.config(state="disabled")
    self.status_bar.config(text=f"⚫ Indexing file: {os.path.basename(file_path)}...")
    
    # Run indexing in background thread
    thread = Thread(target=self._index_file_worker, args=(file_path,))
    thread.start()
  
  METHOD _index_file_worker(file_path: str):
    """Background worker for file indexing"""
    start_time = time.time()
    
    TRY:
      # Read file content
      WITH open(file_path, 'r', encoding='utf-8') AS f:
        file_content = f.read()
      
      # Create document from file
      document = {
        "id": generate_document_id(),
        "content": file_content,
        "source": file_path,
        "metadata": {
          "indexed_at": datetime.now().isoformat(),
          "source_type": "file",
          "filename": os.path.basename(file_path),
          "file_size": os.path.getsize(file_path)
        }
      }
      
      # Create indexing context
      context = IndexingContext(
        database_service=self.service_registry.get("db_main"),
        config={}
      )
      
      # Call indexing pipeline
      result = self.indexing_pipeline.process([document], context)
      
      # Update tracking
      self.document_count += result.document_count
      self.chunk_count += result.chunk_count
      
      elapsed = time.time() - start_time
      
      # Update GUI on main thread
      self.root.after(0, self._on_index_file_success, result, elapsed, file_path)
      
    CATCH UnicodeDecodeError:
      error_msg = f"File encoding error. Please ensure file is UTF-8 text."
      self.root.after(0, self._on_index_file_error, error_msg)
      
    CATCH Exception AS e:
      self.root.after(0, self._on_index_file_error, str(e))
  
  METHOD _on_index_file_success(result: IndexingResult, elapsed: float, file_path: str):
    """Called on main thread after successful file indexing"""
    # Re-enable button
    self.index_file_button.config(state="normal")
    
    # Update status bar
    filename = os.path.basename(file_path)
    status_text = (
      f"🟢 Indexed {filename}: {result.document_count} doc, "
      f"{result.chunk_count} chunks in {elapsed:.2f}s | "
      f"Total: {self.document_count} docs, {self.chunk_count} chunks"
    )
    self.status_bar.config(text=status_text)
    
    # Optional: Clear file path
    self.file_path_entry.delete(0, "end")
    
    # Update button states
    self.update_button_states()
  
  METHOD _on_index_file_error(error_message: str):
    """Called on main thread after file indexing error"""
    # Re-enable button
    self.index_file_button.config(state="normal")
    
    # Show error
    self.show_error(f"File indexing failed: {error_message}")
    self.status_bar.config(text="🔴 File indexing failed")
    
    # Update button states
    self.update_button_states()
  
  METHOD generate_document_id() -> str:
    """Generate unique document ID"""
    RETURN f"doc_{uuid.uuid4().hex[:12]}"
Threading Considerations:
pseudocodeTHREADING_RULES:
  1. Never call indexing_pipeline from main GUI thread
     → Use Thread(target=worker_method)
     
  2. Never update GUI from background thread
     → Use root.after(0, update_method) to schedule on main thread
     
  3. Disable operation buttons during background work
     → Prevents duplicate operations
     
  4. Use try-catch in all worker methods
     → Ensure errors are caught and reported to GUI
     
  5. Optional: Add progress callback for long operations
     → Update status bar with progress percentage
Story Points: 5

Story 18.5: Implement Retrieval Operations
As a user
I want to query indexed documents and see results
So that I can validate retrieval is working correctly
Acceptance Criteria:

Query text captured from textbox
Top-K value read from dropdown
Retrieval calls retrieval_pipeline.retrieve(query, top_k)
Results formatted and displayed in results textbox
Each result shows: rank, score, content preview, source
Empty results handled gracefully
Error handling for retrieval failures
Status bar updates with result count and timing
Results are scrollable
Thread safety (background retrieval)

Pseudocode:
pseudocodeCLASS RAGFactoryGUI:
  # ... existing code ...
  
  METHOD on_retrieve():
    """Handler for Retrieve button"""
    query_text = self.query_entry.get().strip()
    
    IF query_text == "":
      self.show_warning("Please enter a query")
      RETURN
    
    top_k = int(self.topk_dropdown.get())
    
    # Disable button during operation
    self.retrieve_button.config(state="disabled")
    self.status_bar.config(text=f"⚫ Retrieving top {top_k} results...")
    
    # Clear previous results
    self.results_textbox.config(state="normal")
    self.results_textbox.delete("1.0", "end")
    self.results_textbox.insert("1.0", "Searching...\n")
    self.results_textbox.config(state="disabled")
    
    # Run retrieval in background thread
    thread = Thread(target=self._retrieve_worker, args=(query_text, top_k))
    thread.start()
  
  METHOD _retrieve_worker(query: str, top_k: int):
    """Background worker for retrieval"""
    start_time = time.time()
    
    TRY:
      # Create retrieval context
      context = RetrievalContext(
        database_service=self.service_registry.get("db_main"),
        config={}
      )
      
      # Call retrieval pipeline (from Epic 17)
      results = self.retrieval_pipeline.retrieve(
        query=query,
        context=context,
        top_k=top_k
      )
      
      elapsed = time.time() - start_time
      
      # Update GUI on main thread
      self.root.after(0, self._on_retrieve_success, results, query, elapsed)
      
    CATCH Exception AS e:
      self.root.after(0, self._on_retrieve_error, str(e))
  
  METHOD _on_retrieve_success(results: list, query: str, elapsed: float):
    """Called on main thread after successful retrieval"""
    # Re-enable button
    self.retrieve_button.config(state="normal")
    
    # Format results for display
    formatted_results = self._format_results(results, query)
    
    # Update results textbox
    self.results_textbox.config(state="normal")
    self.results_textbox.delete("1.0", "end")
    self.results_textbox.insert("1.0", formatted_results)
    self.results_textbox.config(state="disabled")
    
    # Update status bar
    result_count = len(results)
    status_text = f"🟢 Retrieved {result_count} results in {elapsed:.3f}s"
    self.status_bar.config(text=status_text)
    
    # Update button states
    self.update_button_states()
  
  METHOD _on_retrieve_error(error_message: str):
    """Called on main thread after retrieval error"""
    # Re-enable button
    self.retrieve_button.config(state="normal")
    
    # Show error in results textbox
    self.results_textbox.config(state="normal")
    self.results_textbox.delete("1.0", "end")
    self.results_textbox.insert("1.0", f"Error: {error_message}")
    self.results_textbox.config(state="disabled")
    
    # Show error dialog
    self.show_error(f"Retrieval failed: {error_message}")
    self.status_bar.config(text="🔴 Retrieval failed")
    
    # Update button states
    self.update_button_states()
  
  METHOD _format_results(results: list, query: str) -> str:
    """Format retrieval results for display"""
    IF len(results) == 0:
      RETURN f"No results found for query: \"{query}\"\n\nTry:\n- Indexing some documents first\n- Using different keywords\n- Checking your query spelling"
    
    output = f"Query: \"{query}\"\n"
    output += f"Found {len(results)} results:\n"
    output += "=" * 60 + "\n\n"
    
    FOR rank, result IN enumerate(results, start=1):
      # Extract result fields
      score = result.score IF hasattr(result, 'score') ELSE "N/A"
      content = result.content IF hasattr(result, 'content') ELSE result.text
      source = result.metadata.get('source', 'Unknown') IF hasattr(result, 'metadata') ELSE 'Unknown'
      
      # Truncate content for preview (first 200 chars)
      content_preview = content[:200]
      IF len(content) > 200:
        content_preview += "..."
      
      # Format result
      output += f"[{rank}] Score: {score:.4f}\n"
      output += f"    {content_preview}\n"
      output += f"    Source: {source}\n"
      output += "-" * 60 + "\n\n"
    
    RETURN output
```

**Result Formatting Examples:**
```
Query: "What is machine learning?"
Found 3 results:
============================================================

[1] Score: 0.8923
    Machine learning is a subset of artificial intelligence 
    that enables systems to learn and improve from experience 
    without being explicitly programmed. It focuses on the 
    development of computer programs...
    Source: machine_learning.txt
------------------------------------------------------------

[2] Score: 0.7645
    Types of Machine Learning: 1. Supervised Learning - 
    Training with labeled data 2. Unsupervised Learning - 
    Finding patterns in unlabeled data 3. Reinforcement 
    Learning - Learning through trial...
    Source: machine_learning.txt
------------------------------------------------------------

[3] Score: 0.6812
    Applications: - Image recognition - Natural language 
    processing - Recommendation systems - Autonomous vehicles
    Source: machine_learning.txt
------------------------------------------------------------
```

**No Results Example:**
```
Query: "quantum computing"
Found 0 results:
============================================================

No results found for query: "quantum computing"

Try:
- Indexing some documents first
- Using different keywords
- Checking your query spelling
Story Points: 5

Story 18.6: Implement Utility Operations (Clear, Logs, Help)
As a user
I want utility operations for managing data and getting help
So that I can reset state and troubleshoot issues
Acceptance Criteria:

Clear All Data button warns before deleting
Clear All Data removes indexed documents for current strategy only
View Logs button opens popup window with application logs
Logs window is scrollable and read-only
Help button shows keyboard shortcuts and usage tips
Settings button placeholder (for future enhancement)
All utility operations are non-blocking
Proper confirmation dialogs for destructive operations

Pseudocode:
pseudocodeCLASS RAGFactoryGUI:
  # ... existing code ...
  
  CONSTRUCTOR():
    # ... existing initialization ...
    
    # Initialize logging
    self.log_buffer = []  # Store log messages
    self._setup_logging()
  
  METHOD _setup_logging():
    """Configure logging to capture messages"""
    import logging
    
    # Create custom handler that writes to buffer
    class GUILogHandler(logging.Handler):
      def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer
      
      def emit(self, record):
        log_entry = self.format(record)
        self.buffer.append(log_entry)
        
        # Keep only last 1000 messages
        IF len(self.buffer) > 1000:
          self.buffer.pop(0)
    
    # Add handler to root logger
    handler = GUILogHandler(self.log_buffer)
    handler.setFormatter(
      logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    
    logger = logging.getLogger('rag_factory')
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
  
  METHOD on_clear_data():
    """Handler for Clear All Data button"""
    # Confirmation dialog
    strategy_name = self.current_strategy
    
    IF strategy_name IS None:
      self.show_warning("No strategy loaded. Nothing to clear.")
      RETURN
    
    message = (
      f"Are you sure you want to delete ALL indexed data "
      f"for strategy '{strategy_name}'?\n\n"
      f"This will delete:\n"
      f"- {self.document_count} documents\n"
      f"- {self.chunk_count} chunks\n"
      f"- All associated vectors and metadata\n\n"
      f"This action cannot be undone!"
    )
    
    confirmed = messagebox.askyesno(
      "Confirm Clear Data",
      message,
      icon='warning'
    )
    
    IF NOT confirmed:
      RETURN
    
    # Disable button during operation
    self.clear_button.config(state="disabled")
    self.status_bar.config(text="⚫ Clearing data...")
    
    # Run clear operation in background
    thread = Thread(target=self._clear_data_worker)
    thread.start()
  
  METHOD _clear_data_worker():
    """Background worker for clearing data"""
    TRY:
      # Get database context for current strategy
      db_service = self.service_registry.get("db_main")
      
      # Get table names from strategy configuration
      strategy_config = self.strategy_manager.get_config(self.current_strategy)
      tables = strategy_config['indexer']['db_config']['tables']
      
      # Delete from each table
      FOR logical_name, physical_name IN tables.items():
        db_service.execute(f"DELETE FROM {physical_name}")
      
      # Reset counters
      self.document_count = 0
      self.chunk_count = 0
      
      # Update GUI on main thread
      self.root.after(0, self._on_clear_data_success)
      
    CATCH Exception AS e:
      self.root.after(0, self._on_clear_data_error, str(e))
  
  METHOD _on_clear_data_success():
    """Called on main thread after successful clear"""
    # Re-enable button
    self.clear_button.config(state="normal")
    
    # Update status bar
    self.status_bar.config(text="🟢 All data cleared")
    
    # Show confirmation
    self.show_info("All data has been cleared successfully.")
    
    # Update button states
    self.update_button_states()
  
  METHOD _on_clear_data_error(error_message: str):
    """Called on main thread after clear error"""
    # Re-enable button
    self.clear_button.config(state="normal")
    
    # Show error
    self.show_error(f"Failed to clear data: {error_message}")
    self.status_bar.config(text="🔴 Clear data failed")
  
  METHOD on_view_logs():
    """Handler for View Logs button"""
    # Create popup window
    log_window = Toplevel(self.root)
    log_window.title("Application Logs")
    log_window.geometry("800x600")
    
    # Create text widget with scrollbar
    log_text = Text(log_window, font="Courier", wrap="word")
    log_scrollbar = Scrollbar(log_window, command=log_text.yview)
    log_text.config(yscrollcommand=log_scrollbar.set)
    
    # Layout
    log_text.pack(side="left", fill="both", expand=True)
    log_scrollbar.pack(side="right", fill="y")
    
    # Insert log messages
    log_text.insert("1.0", "\n".join(self.log_buffer))
    log_text.config(state="disabled")
    
    # Auto-scroll to bottom
    log_text.see("end")
    
    # Add refresh button
    def refresh_logs():
      log_text.config(state="normal")
      log_text.delete("1.0", "end")
      log_text.insert("1.0", "\n".join(self.log_buffer))
      log_text.config(state="disabled")
      log_text.see("end")
    
    refresh_button = Button(log_window, text="Refresh", command=refresh_logs)
    refresh_button.pack(side="bottom", pady=5)
  
  METHOD on_show_help():
    """Handler for Help button"""
    help_text = """
RAG Factory - Strategy Pair Tester
===================================

QUICK START:
1. Select a strategy pair from the dropdown
2. Index some text or files
3. Enter a query and click Retrieve

KEYBOARD SHORTCUTS:
Ctrl+L    - Load strategy
Ctrl+I    - Focus text to index
Ctrl+F    - Browse file
Ctrl+Q    - Focus query
Ctrl+R    - Retrieve (when query entered)
Ctrl+K    - Clear all data
Ctrl+H    - Show this help

TIPS:
- Index multiple documents before querying
- Use Top K slider to control result count
- Check logs if something goes wrong
- Clear data to start fresh with new strategy

REQUIREMENTS:
- PostgreSQL with pgvector extension running
- Service registry configured (config/services.yaml)
- At least one strategy pair in strategies/ directory

TROUBLESHOOTING:
- "Missing migrations" → Run: alembic upgrade head
- "Missing services" → Check config/services.yaml
- "No results" → Index documents first
- "Connection error" → Check DATABASE_URL in .env

For more help, see documentation at:
https://github.com/your-repo/rag-factory/docs
"""
    
    # Create popup window
    help_window = Toplevel(self.root)
    help_window.title("Help")
    help_window.geometry("700x600")
    
    # Create text widget with scrollbar
    help_text_widget = Text(help_window, font="Courier", wrap="word")
    help_scrollbar = Scrollbar(help_window, command=help_text_widget.yview)
    help_text_widget.config(yscrollcommand=help_scrollbar.set)
    
    # Layout
    help_text_widget.pack(side="left", fill="both", expand=True)
    help_scrollbar.pack(side="right", fill="y")
    
    # Insert help text
    help_text_widget.insert("1.0", help_text)
    help_text_widget.config(state="disabled")
    
    # Add close button
    close_button = Button(help_window, text="Close", command=help_window.destroy)
    close_button.pack(side="bottom", pady=5)
  
  METHOD _bind_keyboard_shortcuts():
    """Bind keyboard shortcuts"""
    self.root.bind('<Control-l>', lambda e: self.strategy_dropdown.focus())
    self.root.bind('<Control-i>', lambda e: self.text_to_index.focus())
    self.root.bind('<Control-f>', lambda e: self.on_browse_file())
    self.root.bind('<Control-q>', lambda e: self.query_entry.focus())
    self.root.bind('<Control-r>', lambda e: self.on_retrieve() IF self.retrieve_button['state'] == 'normal' ELSE None)
    self.root.bind('<Control-k>', lambda e: self.on_clear_data())
    self.root.bind('<Control-h>', lambda e: self.on_show_help())
Story Points: 3

Story 18.7: Add Polish and User Experience Enhancements
As a user
I want a polished, professional-looking GUI
So that the application is pleasant to use
Acceptance Criteria:

Consistent color scheme throughout application
Proper spacing and padding between components
Icons for buttons (optional, but nice)
Loading spinners during operations (optional)
Tooltips on hover for buttons and controls
Window icon/logo (optional)
Proper window centering on startup
Remember window size/position between sessions
Graceful shutdown (cleanup resources)
About dialog with version information

Pseudocode:
pseudocodeCLASS RAGFactoryGUI:
  # ... existing code ...
  
  CONSTRUCTOR():
    # ... existing initialization ...
    
    # Apply styling
    self._apply_styling()
    
    # Add tooltips
    self._add_tooltips()
    
    # Center window
    self._center_window()
    
    # Bind cleanup on close
    self.root.protocol("WM_DELETE_WINDOW", self.on_close)
  
  METHOD _apply_styling():
    """Apply consistent styling throughout GUI"""
    # Configure ttk styles
    style = ttk.Style()
    style.theme_use('clam')  # Modern theme
    
    # Define color palette
    COLORS = {
      'primary': '#2E7D32',     # Green (success)
      'danger': '#D32F2F',      # Red (error)
      'warning': '#F57C00',     # Orange (warning)
      'info': '#1976D2',        # Blue (info)
      'bg': '#FFFFFF',          # White background
      'fg': '#212121',          # Dark text
      'border': '#BDBDBD'       # Gray border
    }
    
    # Button styles
    style.configure('TButton',
      padding=6,
      relief='flat',
      background=COLORS['primary'],
      foreground='white'
    )
    
    style.configure('Danger.TButton',
      background=COLORS['danger']
    )
    
    # Configure padding for all sections
    PADDING = {
      'section': 10,   # Between sections
      'widget': 5,     # Between widgets
      'internal': 3    # Inside widgets
    }
    
    # Apply padding to grid
    FOR widget IN self.root.winfo_children():
      IF widget.winfo_class() IN ['Frame', 'LabelFrame']:
        widget.grid_configure(padx=PADDING['section'], pady=PADDING['section'])
  
  METHOD _add_tooltips():
    """Add helpful tooltips to UI elements"""
    TOOLTIPS = {
      self.strategy_dropdown: "Select a pre-configured strategy pair",
      self.reload_button: "Rescan strategies/ directory for new configurations",
      self.index_text_button: "Index the text from the textbox above",
      self.browse_button: "Open file dialog to select a file",
      self.index_file_button: "Index the selected file",
      self.topk_dropdown: "Number of results to retrieve",
      self.retrieve_button: "Search for documents matching the query",
      self.clear_button: "Delete all indexed data (cannot be undone)",
      self.logs_button: "View application logs for debugging",
      self.help_button: "Show keyboard shortcuts and usage tips"
    }
    
    FOR widget, tooltip_text IN TOOLTIPS.items():
      self._create_tooltip(widget, tooltip_text)
  
  METHOD _create_tooltip(widget, text: str):
    """Create tooltip that appears on hover"""
    tooltip = None
    
    def show_tooltip(event):
      # Create tooltip window
      tooltip = Toplevel(widget)
      tooltip.wm_overrideredirect(True)  # Remove window decorations
      tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
      
      label = Label(
        tooltip,
        text=text,
        background='#FFFFCC',  # Light yellow
        relief='solid',
        borderwidth=1,
        font=('TkDefaultFont', 9)
      )
      label.pack()
      
      widget.tooltip_window = tooltip
    
    def hide_tooltip(event):
      IF hasattr(widget, 'tooltip_window'):
        widget.tooltip_window.destroy()
        delattr(widget, 'tooltip_window')
    
    widget.bind('<Enter>', show_tooltip)
    widget.bind('<Leave>', hide_tooltip)
  
  METHOD _center_window():
    """Center window on screen"""
    self.root.update_idletasks()
    
    # Get screen dimensions
    screen_width = self.root.winfo_screenwidth()
    screen_height = self.root.winfo_screenheight()
    
    # Get window dimensions
    window_width = self.root.winfo_width()
    window_height = self.root.winfo_height()
    
    # Calculate position
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    
    # Set position
    self.root.geometry(f"+{x}+{y}")
  
  METHOD on_close():
    """Cleanup and close application"""
    # Confirm if data exists
    IF self.document_count > 0:
      confirmed = messagebox.askyesno(
        "Confirm Exit",
        f"You have {self.document_count} indexed documents. "
        f"Exit anyway? (Data will remain in database)"
      )
      
      IF NOT confirmed:
        RETURN
    
    # Cleanup resources
    TRY:
      IF self.service_registry:
        self.service_registry.shutdown()
      
      IF self.strategy_manager:
        # Close any open connections
        pass
    
    CATCH Exception AS e:
      # Log but don't prevent exit
      logging.error(f"Error during cleanup: {e}")
    
    # Destroy window
    self.root.destroy()
  
  METHOD add_about_dialog():
    """Add About dialog to help menu"""
    about_text = """
RAG Factory - Strategy Pair Tester
Version 1.0.0

A lightweight GUI for testing RAG strategy pairs
with indexing and retrieval capabilities.

Built with:
- Python 3.10+
- tkinter (GUI)
- Epic 17 (Strategy Pair System)

© 2024 RAG Factory Project
Licensed under MIT License

For more information:
https://github.com/your-repo/rag-factory
"""
    
    def show_about():
      messagebox.showinfo("About RAG Factory", about_text)
    
    # Add to help menu (if menu bar exists)
    # OR add as button in bottom bar
    self.about_button = Button(self.root, text="About", command=show_about)
    self.about_button.grid(row=13, column=3)
Story Points: 3

Story 18.8: Testing and Documentation
As a developer
I want comprehensive testing and documentation
So that the GUI is reliable and maintainable
Acceptance Criteria:

Unit tests for core GUI methods (mocked backend)
Integration tests with real StrategyPairManager
Test all error paths (missing services, migrations, etc.)
Test threading safety
Test memory leaks (long-running sessions)
User guide with screenshots
Developer guide for extending the GUI
Troubleshooting guide
Installation instructions
Requirements.txt updated

Pseudocode for Tests:
pseudocode# tests/gui/test_gui_core.py

CLASS TestGUICore:
  """Unit tests for GUI components"""
  
  METHOD setup():
    # Create mock backend
    self.mock_registry = Mock(ServiceRegistry)
    self.mock_manager = Mock(StrategyPairManager)
    
    # Create GUI with mocked backend
    self.gui = RAGFactoryGUI()
    self.gui.service_registry = self.mock_registry
    self.gui.strategy_manager = self.mock_manager
  
  METHOD test_strategy_loading():
    """Test strategy selection and loading"""
    # Arrange
    self.mock_manager.load_pair.return_value = (Mock(), Mock())
    
    # Act
    self.gui.strategy_dropdown.set("semantic-local-pair")
    self.gui.on_strategy_selected()
    
    # Assert
    assert self.gui.current_strategy == "semantic-local-pair"
    assert self.mock_manager.load_pair.called_once_with("semantic-local-pair")
  
  METHOD test_missing_migrations_error():
    """Test error handling for missing migrations"""
    # Arrange
    self.mock_manager.load_pair.side_effect = MigrationError(
      missing_revisions=["abc123", "def456"]
    )
    
    # Act
    self.gui.strategy_dropdown.set("test-strategy")
    self.gui.on_strategy_selected()
    
    # Assert
    assert self.gui.current_strategy IS None
    assert "Missing migrations" IN self.gui.status_bar.cget("text")
  
  METHOD test_text_indexing():
    """Test text indexing operation"""
    # Arrange
    self.gui.current_strategy = "test-strategy"
    self.gui.indexing_pipeline = Mock()
    self.gui.text_to_index.insert("1.0", "Test document content")
    
    # Act
    self.gui.on_index_text()
    
    # Wait for background thread
    time.sleep(0.5)
    
    # Assert
    assert self.gui.indexing_pipeline.process.called
  
  METHOD test_retrieval():
    """Test retrieval operation"""
    # Arrange
    self.gui.current_strategy = "test-strategy"
    self.gui.retrieval_pipeline = Mock()
    self.gui.retrieval_pipeline.retrieve.return_value = [
      Mock(score=0.9, content="Result 1", metadata={'source': 'doc1'}),
      Mock(score=0.8, content="Result 2", metadata={'source': 'doc2'})
    ]
    
    self.gui.query_entry.insert(0, "test query")
    
    # Act
    self.gui.on_retrieve()
    
    # Wait for background thread
    time.sleep(0.5)
    
    # Assert
    results_text = self.gui.results_textbox.get("1.0", "end")
    assert "Result 1" IN results_text
    assert "Result 2" IN results_text

# tests/gui/test_gui_integration.py

CLASS TestGUIIntegration:
  """Integration tests with real backend"""
  
  METHOD setup():
    # Use test database
    os.environ['DATABASE_URL'] = 'postgresql://test:test@localhost/rag_test'
    
    # Create real GUI
    self.gui = RAGFactoryGUI()
  
  METHOD test_end_to_end_workflow():
    """Test complete workflow: load → index → query"""
    # Step 1: Load strategy
    self.gui.strategy_dropdown.set("semantic-local-pair")
    self.gui.on_strategy_selected()
    
    assert self.gui.current_strategy IS NOT None
    
    # Step 2: Index text
    self.gui.text_to_index.insert("1.0", "Machine learning is amazing")
    self.gui.on_index_text()
    
    # Wait for indexing
    time.sleep(2)
    
    assert self.gui.document_count > 0
    
    # Step 3: Query
    self.gui.query_entry.insert(0, "What is machine learning?")
    self.gui.on_retrieve()
    
    # Wait for retrieval
    time.sleep(1)
    
    results_text = self.gui.results_textbox.get("1.0", "end")
    assert "Machine learning" IN results_text
```

**Documentation Structure:**
```
docs/gui/
├── user-guide.md              # End-user documentation
│   ├── Installation
│   ├── Quick Start
│   ├── Strategy Selection
│   ├── Indexing Documents
│   ├── Querying
│   ├── Troubleshooting
│   └── FAQ
│
├── developer-guide.md         # Developer documentation
│   ├── Architecture Overview
│   ├── Component Structure
│   ├── Threading Model
│   ├── Adding Features
│   ├── Testing Strategy
│   └── Debugging Tips
│
└── screenshots/               # Screenshots for documentation
    ├── main-window.png
    ├── strategy-loaded.png
    ├── indexing.png
    ├── query-results.png
    └── error-dialog.png
```

**Story Points:** 5

---

## Sprint Planning

**Sprint 19 (Epic 18 - GUI Development):**
- Story 18.1: Design Layout (5 points) - Week 1
- Story 18.2: Core Framework (13 points) - Week 1-2
- Story 18.3: Backend Integration (8 points) - Week 2
- Story 18.4: Indexing Operations (5 points) - Week 3
- Story 18.5: Retrieval Operations (5 points) - Week 3
- Story 18.6: Utility Operations (3 points) - Week 3
- Story 18.7: Polish & UX (3 points) - Week 4
- Story 18.8: Testing & Docs (5 points) - Week 4

**Total:** 47 points (~1 month)

---

## Technical Stack

**GUI Framework:**
- tkinter (built into Python, no extra dependencies)
- ttk for modern widgets

**Threading:**
- Python threading module
- queue module for thread-safe communication

**File Dialogs:**
- tkinter.filedialog

**Message Boxes:**
- tkinter.messagebox

**Logging:**
- Python logging module

**Dependencies (requirements.txt addition):**
```
# GUI (tkinter is built-in, no need to list)
# No additional dependencies needed!

Success Criteria

 GUI launches without errors
 All UI components render correctly
 Strategy pairs can be loaded and displayed
 Text and file indexing works
 Query retrieval works and displays results
 Error handling shows user-friendly messages
 Threading doesn't freeze GUI
 Clear data works without errors
 Logs can be viewed
 Help dialog shows useful information
 All tests pass
 Documentation complete with screenshots
 Cross-platform tested (Windows, Linux, macOS)


Benefits Achieved
User Experience:

✅ Visual interface for non-CLI users
✅ Immediate feedback on operations
✅ No command-line knowledge required
✅ Easy demo tool for presentations

Developer Experience:

✅ Quick way to test strategy pairs
✅ Visual validation of configurations
✅ Easy debugging with logs
✅ Fast iteration during development

Quality:

✅ Delegates to Epic 17 implementation (no logic duplication)
✅ Proper error handling and user feedback
✅ Thread-safe operations
✅ Clean separation of GUI and business logic

Deployment:

✅ Single Python file (easy to distribute)
✅ No extra dependencies (tkinter built-in)
✅ Works on all platforms
✅ Can be packaged as executable (PyInstaller)


Future Enhancements (Post-Epic 18)
Possible additions in future epics:

Settings dialog for configuring service registry
Visual strategy pair editor (YAML editor)
Performance graphs and metrics visualization
Multi-strategy comparison view
Batch document indexing with progress bar
Export results to CSV/JSON
Dark theme support
Internationalization (i18n)
Plugin system for custom visualizations


Notes

This is a development tool, not a production interface
Focus is on simplicity and usefulness, not feature completeness
GUI delegates to Epic 17 - no business logic duplication
tkinter chosen for zero extra dependencies
Single file implementation for easy distribution
Can be extended later with more features if needed
Thread safety is critical - all backend calls in background threads
Read-only config - use text editor for YAML editing (keeps GUI simple)