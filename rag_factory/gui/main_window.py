"""Main GUI window for RAG Factory Strategy Pair Testing.

This module implements the main application window with all UI components
as specified in Story 18.1.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
import asyncio
import yaml
import time
import threading
import traceback
import sys

from rag_factory.config.strategy_pair_manager import StrategyPairManager, ConfigurationError, CompatibilityError
from rag_factory.registry.service_registry import ServiceRegistry
from rag_factory.core.indexing_interface import IIndexingStrategy, IndexingContext
from rag_factory.core.retrieval_interface import IRetrievalStrategy, RetrievalContext
from rag_factory.gui.components import StatusBar, ScrolledText
from rag_factory.gui.utils import run_async_in_thread, safe_gui_update, format_yaml, format_results
import logging

logger = logging.getLogger(__name__)


class RAGFactoryGUI:
    """Main GUI application for RAG Factory Strategy Pair Testing.
    
    This class implements the complete GUI layout from Story 18.1, including:
    - Strategy selection and configuration preview
    - Text and file indexing
    - Query and retrieval
    - Status monitoring
    - Utility functions
    
    Example:
        >>> from rag_factory.registry.service_registry import ServiceRegistry
        >>> from rag_factory.config.strategy_pair_manager import StrategyPairManager
        >>> 
        >>> registry = ServiceRegistry("config/services.yaml")
        >>> manager = StrategyPairManager(registry)
        >>> 
        >>> app = RAGFactoryGUI(manager)
        >>> app.run()
    """
    
    def __init__(
        self,
        config_path: str = "config/services.yaml",
        strategies_dir: str = "strategies",
        alembic_config: str = "alembic.ini"
    ):
        """Initialize the RAG Factory GUI.
        
        Args:
            config_path: Path to services configuration file
            strategies_dir: Directory containing strategy YAML files
            alembic_config: Path to Alembic configuration file
        """
        self.root = tk.Tk()
        self.root.title("RAG Factory - Strategy Pair Tester")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # Center window on screen
        self._center_window()
        
        # Set theme
        try:
            style = ttk.Style()
            style.theme_use('clam')  # Modern theme
        except:
            pass  # Use default if clam not available
        
        # Configuration paths
        self.config_path = Path(config_path)
        self.strategies_dir = Path(strategies_dir)
        self.alembic_config = alembic_config
        
        # Backend components (initialized later)
        self.service_registry: Optional[ServiceRegistry] = None
        self.strategy_manager: Optional[StrategyPairManager] = None
        
        # Strategy state
        self.current_strategy_name: Optional[str] = None
        self.indexing_strategy: Optional[IIndexingStrategy] = None
        self.retrieval_strategy: Optional[IRetrievalStrategy] = None
        
        # Data tracking
        self.indexed_documents: List[Dict[str, Any]] = []
        
        # Log capture buffer
        self.log_buffer: List[str] = []
        self._setup_log_capture()
        
        # Create UI
        self._create_menu()
        self._create_ui()
        
        # Initialize backend
        self._initialize_backend()
        
        # Load strategies if backend initialized
        if self.strategy_manager is not None:
            self._load_strategy_list()
    
    def _center_window(self) -> None:
        """Center the window on the screen."""
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
    
    def _create_tooltip(self, widget, text: str) -> None:
        """Create tooltip that appears on hover.
        
        Args:
            widget: Widget to add tooltip to
            text: Tooltip text
        """
        def show_tooltip(event):
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = tk.Label(
                tooltip,
                text=text,
                background='#FFFFCC',
                foreground='#000000',
                relief='solid',
                borderwidth=1,
                font=('Arial', 9),
                padx=5,
                pady=3
            )
            label.pack()
            widget.tooltip_window = tooltip
        
        def hide_tooltip(event):
            if hasattr(widget, 'tooltip_window'):
                try:
                    widget.tooltip_window.destroy()
                except:
                    pass
        
        widget.bind('<Enter>', show_tooltip)
        widget.bind('<Leave>', hide_tooltip)
    
    def _create_menu(self) -> None:
        """Create application menu bar."""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Reload Configs", command=self._reload_configs)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Clear All Data", command=self._clear_all_data)
        tools_menu.add_command(label="View Logs", command=self._view_logs)
        tools_menu.add_separator()
        tools_menu.add_command(label="Settings", command=self._show_settings)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Help", command=self._show_help)
        help_menu.add_command(label="About", command=self._show_about)
        
        self.root.config(menu=menubar)
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    
    def _show_error_dialog(self, title: str, message: str, details: Optional[str] = None) -> None:
        """Show custom error dialog with selectable text.
        
        Args:
            title: Dialog title
            message: Main error message
            details: Optional detailed error info (traceback, etc.)
        """
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("700x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (350)
        y = (dialog.winfo_screenheight() // 2) - (250)
        dialog.geometry(f"700x500+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget for selectable error message
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        error_text = tk.Text(text_frame, wrap=tk.WORD, font=('TkDefaultFont', 10), height=20)
        scrollbar = ttk.Scrollbar(text_frame, command=error_text.yview)
        error_text.configure(yscrollcommand=scrollbar.set)
        
        error_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Insert error content
        full_text = f"{message}\n"
        if details:
            full_text += f"\n{'='*60}\nDetails:\n{'='*60}\n{details}"
        
        error_text.insert('1.0', full_text)
        error_text.configure(state='normal')  # Keep it editable so text is selectable
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        def copy_all():
            """Copy all error text to clipboard."""
            self.root.clipboard_clear()
            self.root.clipboard_append(full_text)
            self.root.update()
            copy_btn.config(text="✓ Copied!")
            dialog.after(1500, lambda: copy_btn.config(text="Copy to Clipboard"))
        
        # Copy button
        copy_btn = ttk.Button(button_frame, text="Copy to Clipboard", command=copy_all)
        copy_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Close button
        close_btn = ttk.Button(button_frame, text="Close", command=dialog.destroy)
        close_btn.pack(side=tk.RIGHT)
        
        # Select all text by default for easy copying
        error_text.tag_add(tk.SEL, "1.0", tk.END)
        error_text.mark_set(tk.INSERT, "1.0")
        error_text.focus_set()
    
    def _create_ui(self) -> None:
        """Create all UI components."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure root grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Configure main frame grid
        main_frame.grid_rowconfigure(7, weight=1)  # Query section expands
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Create sections
        row = 0
        row = self._create_strategy_section(main_frame, row)
        row = self._create_config_preview_section(main_frame, row)
        row = self._create_text_indexing_section(main_frame, row)
        row = self._create_file_indexing_section(main_frame, row)
        row = self._create_query_section(main_frame, row)
        row = self._create_utility_buttons(main_frame, row)
        
        # Status bar at bottom
        self.status_bar = StatusBar(self.root)
        self.status_bar.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Start periodic status updates
        self.status_bar.start_periodic_update()
    
    def _setup_log_capture(self) -> None:
        """Setup log capture to memory buffer."""
        class GUILogHandler(logging.Handler):
            """Custom log handler that captures logs to buffer."""
            def __init__(self, buffer: List[str]):
                super().__init__()
                self.buffer = buffer
                self.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S'
                ))
            
            def emit(self, record):
                try:
                    log_entry = self.format(record)
                    self.buffer.append(log_entry)
                    
                    # Keep only last 1000 messages
                    if len(self.buffer) > 1000:
                        self.buffer.pop(0)
                except Exception:
                    self.handleError(record)
        
        # Get the GUI logger
        gui_logger = logging.getLogger('rag_factory.gui.main_window')
        gui_logger.setLevel(logging.INFO)  # Set level BEFORE adding handler
        gui_logger.disabled = False  # Ensure logger is not disabled by other tests
        gui_logger.propagate = True  # Ensure propagation is enabled
        
        # Remove any existing GUILogHandler instances to prevent accumulation
        # This is important when multiple GUI instances are created (e.g., in tests)
        handlers_to_remove = []
        for handler in gui_logger.handlers[:]:  # Copy list to avoid modification during iteration
            if type(handler).__name__ == 'GUILogHandler':
                handlers_to_remove.append(handler)
        
        for handler in handlers_to_remove:
            gui_logger.removeHandler(handler)
        
        # Create and add new handler
        handler = GUILogHandler(self.log_buffer)
        handler.setLevel(logging.INFO)
        gui_logger.addHandler(handler)
        
        logger.info("GUI log capture initialized")
    
    def _initialize_backend(self) -> None:
        """Initialize ServiceRegistry and StrategyPairManager.
        
        This method initializes the backend components required for
        strategy loading and execution. Errors are displayed to the user
        but do not prevent the GUI from launching.
        """
        try:
            # Check if config file exists
            if not self.config_path.exists():
                error_msg = (
                    f"Service configuration not found: {self.config_path}\n\n"
                    f"Please create the configuration file or specify a custom path with:\n"
                    f"rag-factory gui --config /path/to/services.yaml"
                )
                self.status_bar.set_status("error", "Service config not found")
                self._show_error_dialog("Configuration Error", error_msg)
                logger.error(f"Service configuration not found: {self.config_path}")
                return
            
            # Initialize ServiceRegistry
            self.status_bar.set_status("working", "Loading services...")
            logger.info(f"Initializing ServiceRegistry from {self.config_path}")
            
            try:
                self.service_registry = ServiceRegistry(str(self.config_path))
            except Exception as e:
                error_msg = (
                    f"Failed to initialize ServiceRegistry:\n{str(e)}\n\n"
                    f"Please check {self.config_path} for errors."
                )
                self.status_bar.set_status("error", "Service initialization failed")
                self._show_error_dialog("Service Registry Error", error_msg)
                logger.error(f"ServiceRegistry initialization failed: {e}", exc_info=True)
                return
            
            # Verify required services
            if not self._verify_required_services():
                return
            
            # Initialize StrategyPairManager
            self.status_bar.set_status("working", "Initializing strategy manager...")
            logger.info("Initializing StrategyPairManager")
            
            try:
                self.strategy_manager = StrategyPairManager(
                    service_registry=self.service_registry,
                    config_dir=str(self.strategies_dir),
                    alembic_config=self.alembic_config
                )
            except Exception as e:
                error_msg = (
                    f"Failed to initialize StrategyPairManager:\n{str(e)}\n\n"
                    f"Please check your configuration."
                )
                self.status_bar.set_status("error", "Strategy manager initialization failed")
                self._show_error_dialog("Strategy Manager Error", error_msg)
                logger.error(f"StrategyPairManager initialization failed: {e}", exc_info=True)
                return
            
            self.status_bar.set_status("success", "Backend initialized")
            logger.info("Backend initialization complete")
            
        except Exception as e:
            error_msg = f"Unexpected error during backend initialization:\n{str(e)}"
            self.status_bar.set_status("error", "Initialization failed")
            self._show_error_dialog("Initialization Error", error_msg)
            logger.error(f"Backend initialization failed: {e}", exc_info=True)
    
    def _verify_required_services(self) -> bool:
        """Verify that required services are available.
        
        Returns:
            True if all required services are available, False otherwise
        """
        if self.service_registry is None:
            return False
        
        # Get list of available services
        available_services = self.service_registry.list_services()
        
        # Define required services (at minimum, we need a database)
        required_services = ["db_main"]
        
        # Check for missing services
        missing_services = [s for s in required_services if s not in available_services]
        
        if missing_services:
            error_msg = (
                f"Missing required services:\n"
                f"{''.join(f'  - {s}' + chr(10) for s in missing_services)}\n"
                f"Please check {self.config_path} and ensure all required services are configured."
            )
            self.status_bar.set_status("error", "Missing required services")
            self._show_error_dialog("Missing Services", error_msg)
            logger.error(f"Missing required services: {missing_services}")
            return False
        
        logger.info(f"All required services available: {required_services}")
        return True
    
    def _create_strategy_section(self, parent: ttk.Frame, row: int) -> int:
        """Create strategy selection section.
        
        Args:
            parent: Parent frame
            row: Starting row number
            
        Returns:
            Next available row number
        """
        # Section label
        ttk.Label(
            parent,
            text="[1] Strategy Selection",
            font=("TkDefaultFont", 10, "bold")
        ).grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1
        
        # Strategy frame
        strategy_frame = ttk.LabelFrame(parent, text="", padding="5")
        strategy_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        strategy_frame.grid_columnconfigure(1, weight=1)
        row += 1
        
        # Strategy dropdown
        ttk.Label(strategy_frame, text="Strategy Pair:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        
        self.strategy_var = tk.StringVar()
        self.strategy_dropdown = ttk.Combobox(
            strategy_frame,
            textvariable=self.strategy_var,
            state="readonly",
            width=30
        )
        self.strategy_dropdown.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        self.strategy_dropdown.bind("<<ComboboxSelected>>", self._on_strategy_selected)
        
        # Reload button
        self.reload_btn = ttk.Button(
            strategy_frame,
            text="Reload Configs",
            command=self._reload_strategies
        )
        self.reload_btn.grid(row=0, column=2, sticky="e")
        
        return row
    
    def _create_config_preview_section(self, parent: ttk.Frame, row: int) -> int:
        """Create configuration preview section.
        
        Args:
            parent: Parent frame
            row: Starting row number
            
        Returns:
            Next available row number
        """
        # Section label
        ttk.Label(
            parent,
            text="[2] Configuration Preview (Read-Only)",
            font=("TkDefaultFont", 10, "bold")
        ).grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1
        
        # Config preview text
        self.config_preview = ScrolledText(
            parent,
            read_only=True,
            monospace=True,
            height=8,
            wrap=tk.NONE
        )
        self.config_preview.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        row += 1
        
        return row
    
    def _create_text_indexing_section(self, parent: ttk.Frame, row: int) -> int:
        """Create text indexing section.
        
        Args:
            parent: Parent frame
            row: Starting row number
            
        Returns:
            Next available row number
        """
        # Section label
        ttk.Label(
            parent,
            text="[3] Text Indexing",
            font=("TkDefaultFont", 10, "bold")
        ).grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1
        
        # Text indexing frame
        text_frame = ttk.LabelFrame(parent, text="", padding="5")
        text_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        text_frame.grid_rowconfigure(1, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        row += 1
        
        # Label
        ttk.Label(text_frame, text="Text to Index:").grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        # Text input
        self.text_input = ScrolledText(
            text_frame,
            read_only=False,
            height=4
        )
        self.text_input.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        self.text_input.set_placeholder("Type or paste text here...")
        
        # Index button
        self.index_text_btn = ttk.Button(
            text_frame,
            text="Index Text",
            command=self._index_text,
            state=tk.DISABLED
        )
        self.index_text_btn.grid(row=2, column=0, sticky="e")
        
        # Bind text change to update button state
        self.text_input.text.bind("<KeyRelease>", lambda e: self._update_button_states())
        
        return row
    
    def _create_file_indexing_section(self, parent: ttk.Frame, row: int) -> int:
        """Create file indexing section.
        
        Args:
            parent: Parent frame
            row: Starting row number
            
        Returns:
            Next available row number
        """
        # Section label
        ttk.Label(
            parent,
            text="[4] File Indexing",
            font=("TkDefaultFont", 10, "bold")
        ).grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1
        
        # File indexing frame
        file_frame = ttk.LabelFrame(parent, text="", padding="5")
        file_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        file_frame.grid_columnconfigure(1, weight=1)
        row += 1
        
        # File path
        ttk.Label(file_frame, text="File Path:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        
        self.file_path_var = tk.StringVar()
        self.file_path_entry = ttk.Entry(file_frame, textvariable=self.file_path_var)
        self.file_path_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        self.file_path_var.trace_add("write", lambda *args: self._update_button_states())
        
        # Browse button
        self.browse_btn = ttk.Button(
            file_frame,
            text="Browse",
            command=self._browse_file
        )
        self.browse_btn.grid(row=0, column=2)
        
        # Index file button
        self.index_file_btn = ttk.Button(
            file_frame,
            text="Index File",
            command=self._index_file,
            state=tk.DISABLED
        )
        self.index_file_btn.grid(row=1, column=0, columnspan=3, sticky="e", pady=(5, 0))
        
        return row
    
    def _create_query_section(self, parent: ttk.Frame, row: int) -> int:
        """Create query and retrieval section.
        
        Args:
            parent: Parent frame
            row: Starting row number
            
        Returns:
            Next available row number
        """
        # Section label
        ttk.Label(
            parent,
            text="[5] Query & Retrieval",
            font=("TkDefaultFont", 10, "bold")
        ).grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1
        
        # Query frame
        query_frame = ttk.LabelFrame(parent, text="", padding="5")
        query_frame.grid(row=row, column=0, sticky="nsew", pady=(0, 10))
        query_frame.grid_rowconfigure(3, weight=1)
        query_frame.grid_columnconfigure(1, weight=1)
        row += 1
        
        # Query input
        ttk.Label(query_frame, text="Query:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        
        self.query_var = tk.StringVar()
        self.query_entry = ttk.Entry(query_frame, textvariable=self.query_var)
        self.query_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        self.query_var.trace_add("write", lambda *args: self._update_button_states())
        
        # Retrieve button and Top K
        button_frame = ttk.Frame(query_frame)
        button_frame.grid(row=1, column=0, columnspan=2, sticky="e", pady=(5, 0))
        
        self.retrieve_btn = ttk.Button(
            button_frame,
            text="Retrieve",
            command=self._retrieve,
            state=tk.DISABLED
        )
        self.retrieve_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(button_frame, text="Top K:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.top_k_var = tk.StringVar(value="5")
        self.top_k_dropdown = ttk.Combobox(
            button_frame,
            textvariable=self.top_k_var,
            values=["1", "3", "5", "10", "20"],
            state="readonly",
            width=5
        )
        self.top_k_dropdown.pack(side=tk.LEFT)
        
        # Results label
        ttk.Label(query_frame, text="Results:").grid(row=2, column=0, sticky="nw", pady=(10, 5))
        
        # Results display
        self.results_display = ScrolledText(
            query_frame,
            read_only=True,
            monospace=True,
            height=10
        )
        self.results_display.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(0, 0))
        
        return row
    
    def _create_utility_buttons(self, parent: ttk.Frame, row: int) -> int:
        """Create utility buttons section.
        
        Args:
            parent: Parent frame
            row: Starting row number
            
        Returns:
            Next available row number
        """
        # Utility buttons frame
        util_frame = ttk.Frame(parent)
        util_frame.grid(row=row, column=0, sticky="ew", pady=(10, 0))
        row += 1
        
        return row
    
    def _load_strategy_list(self) -> None:
        """Load list of available strategy YAML files."""
        if self.strategy_manager is None:
            logger.warning("Cannot load strategies: StrategyPairManager not initialized")
            return
        
        try:
            if not self.strategies_dir.exists():
                warning_msg = (
                    f"Strategies directory not found: {self.strategies_dir}\n\n"
                    f"Please create the directory and add strategy configurations."
                )
                self.status_bar.set_status("error", "Strategies directory not found")
                messagebox.showwarning("Strategies Not Found", warning_msg)
                logger.warning(f"Strategies directory not found: {self.strategies_dir}")
                return
            
            # Find all .yaml files
            yaml_files = sorted(self.strategies_dir.glob("*.yaml"))
            strategy_names = [f.stem for f in yaml_files]
            
            if not strategy_names:
                warning_msg = (
                    f"No strategy pairs found in {self.strategies_dir}\n\n"
                    f"Please add at least one strategy configuration (.yaml file)."
                )
                self.status_bar.set_status("error", "No strategy files found")
                messagebox.showwarning("No Strategies", warning_msg)
                logger.warning(f"No strategy files found in {self.strategies_dir}")
                return
            
            # Update dropdown
            self.strategy_dropdown["values"] = strategy_names
            
            # Select first strategy by default
            if strategy_names:
                self.strategy_var.set(strategy_names[0])
                self._on_strategy_selected(None)
            
        except Exception as e:
            self.status_bar.set_status("error", f"Failed to load strategies: {e}")
    
    def _reload_strategies(self) -> None:
        """Reload strategy list from directory."""
        self.status_bar.set_status("working", "Reloading strategies...")
        self._load_strategy_list()
        self.status_bar.set_status("success", "Strategies reloaded")
    
    def _on_strategy_selected(self, event) -> None:
        """Handle strategy selection from dropdown.
        
        Args:
            event: Tkinter event (unused)
        """
        strategy_name = self.strategy_var.get()
        if not strategy_name:
            return
        
        self.status_bar.set_status("working", f"Loading strategy: {strategy_name}")
        
        def load_strategy():
            """Load strategy in background thread."""
            try:
                # Load strategy pair
                indexing, retrieval = self.strategy_manager.load_pair(strategy_name)
                
                # Load YAML for preview
                yaml_path = self.strategies_dir / f"{strategy_name}.yaml"
                with open(yaml_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Update GUI in main thread
                safe_gui_update(
                    self.root,
                    self._on_strategy_loaded,
                    strategy_name,
                    indexing,
                    retrieval,
                    config
                )
                
            except (ConfigurationError, CompatibilityError) as e:
                safe_gui_update(
                    self.root,
                    self._on_strategy_load_error,
                    str(e)
                )
            except Exception as e:
                safe_gui_update(
                    self.root,
                    self._on_strategy_load_error,
                    f"Unexpected error: {e}"
                )
        
        # Run in background thread
        import threading
        thread = threading.Thread(target=load_strategy, daemon=True)
        thread.start()
    
    def _on_strategy_loaded(
        self,
        strategy_name: str,
        indexing: IIndexingStrategy,
        retrieval: IRetrievalStrategy,
        config: Dict[str, Any]
    ) -> None:
        """Handle successful strategy loading.
        
        Args:
            strategy_name: Name of loaded strategy
            indexing: Indexing strategy instance
            retrieval: Retrieval strategy instance
            config: Strategy configuration dict
        """
        self.current_strategy_name = strategy_name
        self.indexing_strategy = indexing
        self.retrieval_strategy = retrieval
        
        # Update config preview
        self.config_preview.set_text(format_yaml(config))
        
        # Update status
        self.status_bar.set_status("success", f"Strategy '{strategy_name}' loaded")
        
        # Update button states
        self._update_button_states()
    
    def _on_strategy_load_error(self, error_msg: str) -> None:
        """Handle strategy loading error.
        
        Args:
            error_msg: Error message
        """
        # Parse error message to categorize error type
        if "migration" in error_msg.lower():
            # Migration error - provide specific guidance
            enhanced_msg = (
                f"{error_msg}\n\n"
                f"To fix this issue, run:\n"
                f"  alembic upgrade heads\n\n"
                f"This will apply all pending database migrations."
            )
            details = f"Full error:\n{error_msg}"
            self.status_bar.set_status("error", "Missing migrations")
            self._show_error_dialog("Migration Error", enhanced_msg, details)
            logger.error(f"Migration error: {error_msg}")
        elif "service" in error_msg.lower():
            # Service error - suggest checking config
            enhanced_msg = (
                f"{error_msg}\n\n"
                f"Please check {self.config_path} and ensure all required services are configured."
            )
            self.status_bar.set_status("error", "Missing services")
            self._show_error_dialog("Service Error", enhanced_msg)
            logger.error(f"Service error: {error_msg}")
        else:
            # Generic error
            self.status_bar.set_status("error", "Strategy load failed")
            self._show_error_dialog("Strategy Load Error", error_msg)
            logger.error(f"Strategy load error: {error_msg}")
        
        # Clear strategy
        self.current_strategy_name = None
        self.indexing_strategy = None
        self.retrieval_strategy = None
        self.config_preview.clear()
        self._update_button_states()
    
    def _update_button_states(self) -> None:
        """Update enable/disable state of all buttons based on current state."""
        strategy_loaded = self.indexing_strategy is not None
        
        # Text indexing button
        text_not_empty = not self.text_input.is_empty()
        self.index_text_btn.config(
            state=tk.NORMAL if (strategy_loaded and text_not_empty) else tk.DISABLED
        )
        
        # File indexing button
        file_path = self.file_path_var.get().strip()
        file_valid = file_path and Path(file_path).exists()
        self.index_file_btn.config(
            state=tk.NORMAL if (strategy_loaded and file_valid) else tk.DISABLED
        )
        
        # Retrieve button
        query_not_empty = bool(self.query_var.get().strip())
        self.retrieve_btn.config(
            state=tk.NORMAL if (strategy_loaded and query_not_empty) else tk.DISABLED
        )
    
    def _browse_file(self) -> None:
        """Open file browser dialog."""
        filename = filedialog.askopenfilename(
            title="Select File to Index",
            filetypes=[
                ("Text files", "*.txt"),
                ("Markdown files", "*.md"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.file_path_var.set(filename)
    
    def _index_text(self) -> None:
        """Index text from text input."""
        text = self.text_input.get_text()
        if not text or not self.indexing_strategy:
            return
        
        self.status_bar.set_status("working", "Indexing text...")
        self.index_text_btn.config(state=tk.DISABLED)
        
        def index_operation():
            """Perform indexing in background."""
            try:
                start_time = time.time()
                
                # Create document with metadata
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                doc_id = f"text_doc_{timestamp}_{len(self.indexed_documents) + 1}"
                document = {
                    "id": doc_id,
                    "content": text,
                    "metadata": {
                        "source": "gui_text_input",
                        "indexed_at": datetime.now().isoformat()
                    }
                }
                
                # Create indexing context
                if self.service_registry is None:
                    raise Exception("Service registry not initialized")
                
                context = IndexingContext(
                    database_service=self.service_registry.get("db_main")
                )
                
                # Call indexing pipeline (async)
                logger.info(f"Indexing document: {doc_id}")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self.indexing_strategy.process([document], context)
                    )
                finally:
                    loop.close()
                
                # Extract results
                chunk_count = result.total_chunks if hasattr(result, 'total_chunks') else 0
                logger.info(f"Indexed {doc_id}: {chunk_count} chunks")
                
                # Track indexed document
                self.indexed_documents.append(document)
                
                elapsed = time.time() - start_time
                
                # Update GUI
                safe_gui_update(
                    self.root,
                    self._on_index_complete,
                    f"Indexed 1 document in {elapsed:.2f}s",
                    documents=1,
                    chunks=chunk_count
                )
                
            except Exception as e:
                logger.error(f"Text indexing failed: {e}", exc_info=True)
                safe_gui_update(
                    self.root,
                    self._on_index_error,
                    str(e)
                )
        
        # Run in background
        thread = threading.Thread(target=index_operation, daemon=True)
        thread.start()
    
    def _index_file(self) -> None:
        """Index file from file path."""
        file_path = Path(self.file_path_var.get())
        if not file_path.exists() or not self.indexing_strategy:
            return
        
        self.status_bar.set_status("working", f"Indexing file: {file_path.name}")
        self.index_file_btn.config(state=tk.DISABLED)
        
        def index_operation():
            """Perform file indexing in background."""
            try:
                start_time = time.time()
                
                # Read file with encoding handling
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # Try with different encoding
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            content = f.read()
                        logger.warning(f"File {file_path} read with latin-1 encoding")
                    except Exception as e:
                        raise Exception(
                            f"Unable to read file: {file_path.name}\n\n"
                            f"The file appears to be binary or uses an unsupported encoding.\n"
                            f"Please use UTF-8 encoded text files (.txt, .md, etc.)."
                        )
                
                # Create document with file metadata
                doc_id = file_path.stem
                document = {
                    "id": doc_id,
                    "content": content,
                    "metadata": {
                        "source": str(file_path),
                        "filename": file_path.name,
                        "indexed_at": datetime.now().isoformat()
                    }
                }
                
                # Create indexing context
                if self.service_registry is None:
                    raise Exception("Service registry not initialized")
                
                context = IndexingContext(
                    database_service=self.service_registry.get("db_main")
                )
                
                # Call indexing pipeline (async)
                logger.info(f"Indexing file: {file_path.name}")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self.indexing_strategy.process([document], context)
                    )
                finally:
                    loop.close()
                
                # Extract results
                chunk_count = result.total_chunks if hasattr(result, 'total_chunks') else 0
                logger.info(f"Indexed {file_path.name}: {chunk_count} chunks")
                
                # Track indexed document
                self.indexed_documents.append(document)
                
                elapsed = time.time() - start_time
                
                # Update GUI
                safe_gui_update(
                    self.root,
                    self._on_index_complete,
                    f"Indexed {file_path.name} in {elapsed:.2f}s",
                    documents=1,
                    chunks=chunk_count
                )
                
            except Exception as e:
                logger.error(f"File indexing failed: {e}", exc_info=True)
                safe_gui_update(
                    self.root,
                    self._on_index_error,
                    str(e)
                )
        
        # Run in background
        thread = threading.Thread(target=index_operation, daemon=True)
        thread.start()
    
    def _on_index_complete(self, message: str, documents: int, chunks: int) -> None:
        """Handle successful indexing completion.
        
        Args:
            message: Success message
            documents: Number of documents indexed
            chunks: Number of chunks created
        """
        self.status_bar.set_status("success", message)
        self.status_bar.increment_counts(documents=documents, chunks=chunks)
        self._update_button_states()
    
    def _on_index_error(self, error_msg: str) -> None:
        """Handle indexing error.
        
        Args:
            error_msg: Error message
        """
        self.status_bar.set_status("error", f"Indexing failed: {error_msg}")
        self._show_error_dialog("Indexing Error", error_msg)
        self._update_button_states()
    
    def _retrieve(self) -> None:
        """Perform retrieval query."""
        query = self.query_var.get().strip()
        if not query or not self.retrieval_strategy:
            return
        
        # Get top_k value
        try:
            top_k = int(self.top_k_var.get())
        except ValueError:
            top_k = 5
        
        self.status_bar.set_status("working", "Retrieving...")
        self.retrieve_btn.config(state=tk.DISABLED)
        
        # Clear previous results and show searching message
        self.results_display.set_text("Searching...")
        
        def retrieve_operation():
            """Perform retrieval in background."""
            try:
                start_time = time.time()
                
                # Create retrieval context
                if self.service_registry is None:
                    raise Exception("Service registry not initialized")
                
                context = RetrievalContext(
                    database_service=self.service_registry.get("db_main"),
                    config={"top_k": top_k}
                )
                
                # Call retrieval pipeline (async)
                logger.info(f"Retrieving for query: {query} (top_k={top_k})")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(
                        self.retrieval_strategy.retrieve(query, context)
                    )
                finally:
                    loop.close()
                
                # Extract results
                # results could be a list of Chunk objects or RetrievalResult
                if hasattr(results, 'chunks'):
                    chunks = results.chunks
                elif isinstance(results, list):
                    chunks = results
                else:
                    chunks = []
                
                logger.info(f"Retrieved {len(chunks)} results for query: {query}")
                
                elapsed = time.time() - start_time
                
                # Format results
                formatted_results = self._format_retrieval_results(query, chunks, top_k)
                
                # Update GUI
                safe_gui_update(
                    self.root,
                    self._on_retrieve_complete,
                    formatted_results,
                    len(chunks),
                    elapsed
                )
                
            except Exception as e:
                logger.error(f"Retrieval failed: {e}", exc_info=True)
                safe_gui_update(
                    self.root,
                    self._on_retrieve_error,
                    str(e)
                )
        
        # Run in background
        thread = threading.Thread(target=retrieve_operation, daemon=True)
        thread.start()
    
    def _format_retrieval_results(self, query: str, results: List, top_k: int) -> str:
        """Format retrieval results for display.
        
        Args:
            query: The query string
            results: List of retrieved chunks/results
            top_k: Maximum number of results to display
            
        Returns:
            Formatted results string
        """
        if not results:
            return f'''No results found for query: "{query}"

Suggestions:
- Make sure documents are indexed first (use Index Text or Index File)
- Try different keywords
- Check spelling
- Try broader search terms
- Index more documents to improve coverage'''
        
        output = []
        output.append(f'Query: "{query}"')
        output.append(f'Found {len(results)} result{"s" if len(results) != 1 else ""}:')
        output.append('=' * 60)
        output.append('')
        
        for i, result in enumerate(results[:top_k], 1):
            # Extract score
            if hasattr(result, 'score'):
                score = result.score
            elif hasattr(result, 'similarity'):
                score = result.similarity
            elif isinstance(result, dict) and 'score' in result:
                score = result['score']
            else:
                score = 0.0
            
            # Extract content
            if hasattr(result, 'content'):
                content = result.content
            elif hasattr(result, 'text'):
                content = result.text
            elif isinstance(result, dict) and 'content' in result:
                content = result['content']
            else:
                content = str(result)
            
            # Extract source
            source = "Unknown"
            if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
                source = result.metadata.get('source', result.metadata.get('filename', 'Unknown'))
            elif isinstance(result, dict) and 'metadata' in result:
                metadata = result['metadata']
                if isinstance(metadata, dict):
                    source = metadata.get('source', metadata.get('filename', 'Unknown'))
            
            # Truncate content to 200 chars
            if len(content) > 200:
                content = content[:197] + '...'
            
            # Format result
            output.append(f'[{i}] Score: {score:.4f}')
            output.append(f'    {content}')
            output.append(f'    Source: {source}')
            output.append('-' * 60)
            output.append('')
        
        return '\n'.join(output)
    
    def _on_retrieve_complete(self, results: str, count: int, elapsed: float) -> None:
        """Handle successful retrieval completion.
        
        Args:
            results: Formatted results string
            count: Number of results found
            elapsed: Time taken in seconds
        """
        self.results_display.set_text(results)
        self.status_bar.set_status("success", f"Found {count} result{('s' if count != 1 else '')} in {elapsed:.2f}s")
        self._update_button_states()
    
    def _on_retrieve_error(self, error_msg: str) -> None:
        """Handle retrieval error.
        
        Args:
            error_msg: Error message
        """
        error_display = f"Retrieval Error:\n\n{error_msg}\n\nPlease check:\n- Database is running\n- Documents are indexed\n- Service configuration is correct"
        self.results_display.set_text(error_display)
        self.status_bar.set_status("error", "Retrieval failed")
        self._show_error_dialog("Retrieval Error", error_msg)
        self._update_button_states()
    
    def _clear_all_data(self) -> None:
        """Clear all indexed data after confirmation."""
        if not self.indexed_documents and self.status_bar.document_count == 0:
            messagebox.showinfo("Clear Data", "No data to clear.")
            return
        
        # Build confirmation message with details
        strategy_name = self.current_strategy_name or "Unknown"
        doc_count = self.status_bar.document_count
        chunk_count = self.status_bar.chunk_count
        
        confirmation_msg = f"""Are you sure you want to clear all indexed data?

Strategy: {strategy_name}
Documents: {doc_count}
Chunks: {chunk_count}

This will:
- Remove all documents and chunks from the database
- Reset all counters to zero
- Clear the results display

This action cannot be undone!"""
        
        response = messagebox.askyesno(
            "Clear All Data - Confirmation Required",
            confirmation_msg,
            icon='warning'
        )
        
        if response:
            try:
                # Clear database if we have a strategy and service registry
                if self.service_registry and self.current_strategy_name:
                    db_service = self.service_registry.get("db_main")
                    if db_service and hasattr(db_service, 'clear_all'):
                        # Call database clear method
                        db_service.clear_all()
                        logger.info("Database cleared successfully")
                
                # Clear indexed documents
                self.indexed_documents.clear()
                
                # Reset counts
                self.status_bar.reset_counts()
                
                # Clear results
                self.results_display.clear()
                
                # Clear text inputs
                self.text_input.clear()
                
                self.status_bar.set_status("success", "All data cleared successfully")
                logger.info("All data cleared")
                
            except Exception as e:
                error_msg = f"Error clearing data: {e}"
                logger.error(error_msg, exc_info=True)
                self._show_error_dialog("Clear Data Error", error_msg)
                self.status_bar.set_status("error", "Failed to clear data")
    
    def _view_logs(self) -> None:
        """Open logs viewer window."""
        # Create log viewer window
        log_window = tk.Toplevel(self.root)
        log_window.title("Application Logs")
        log_window.geometry("800x600")
        
        # Create frame for controls
        control_frame = ttk.Frame(log_window)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add refresh button
        def refresh_logs():
            log_text.delete('1.0', tk.END)
            log_text.insert('1.0', '\n'.join(self.log_buffer))
            log_text.see(tk.END)  # Auto-scroll to bottom
        
        refresh_btn = ttk.Button(control_frame, text="Refresh", command=refresh_logs)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # Add clear button
        def clear_logs():
            self.log_buffer.clear()
            log_text.delete('1.0', tk.END)
            logger.info("Log buffer cleared")
        
        clear_btn = ttk.Button(control_frame, text="Clear Buffer", command=clear_logs)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Add label showing log count
        count_label = ttk.Label(control_frame, text=f"Logs: {len(self.log_buffer)}")
        count_label.pack(side=tk.RIGHT, padx=5)
        
        # Create scrolled text for logs
        log_frame = ttk.Frame(log_window)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        log_text = tk.Text(log_frame, wrap=tk.WORD, font=('Courier', 9))
        log_scrollbar = ttk.Scrollbar(log_frame, command=log_text.yview)
        log_text.configure(yscrollcommand=log_scrollbar.set)
        
        log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Make read-only
        log_text.configure(state='normal')
        
        # Load logs
        refresh_logs()
        
        logger.info("Log viewer opened")
    
    def _reload_configs(self) -> None:
        """Reload configuration files and strategies."""
        try:
            self.status_bar.set_status("working", "Reloading configurations...")
            
            # Reinitialize backend
            self._initialize_backend()
            
            # Reload strategy list
            if self.strategy_manager is not None:
                self._load_strategy_list()
                self.status_bar.set_status("success", "Configurations reloaded")
                messagebox.showinfo("Reload Complete", "Configurations reloaded successfully")
            else:
                self.status_bar.set_status("error", "Failed to reload")
                self._show_error_dialog("Reload Failed", "Failed to reload configurations")
                
        except Exception as e:
            error_msg = f"Error reloading configurations: {e}"
            logger.error(error_msg, exc_info=True)
            self.status_bar.set_status("error", "Reload failed")
            self._show_error_dialog("Reload Error", error_msg)
    
    def _show_settings(self) -> None:
        """Show settings dialog."""
        messagebox.showinfo(
            "Settings",
            "Settings dialog not yet implemented.\n\n"
            "This is a placeholder for future configuration options.\n\n"
            "Future settings may include:\n"
            "- Database connection parameters\n"
            "- Default strategy selection\n"
            "- UI preferences\n"
            "- Logging levels"
        )
    
    def _show_help(self) -> None:
        """Show help dialog."""
        help_text = """RAG Factory - Strategy Pair Tester
===================================

QUICK START:
1. Select a strategy pair from the dropdown
2. Index some text or files using the indexing sections
3. Enter a query and click Retrieve to search

WORKFLOW:
• Load Strategy: Select from dropdown (Ctrl+L)
• Index Text: Enter text and click "Index Text" (Ctrl+I)
• Index File: Browse and select file (Ctrl+F)
• Query: Enter query and click "Retrieve" (Ctrl+Q, then Ctrl+R)
• Clear Data: Use "Clear All Data" button (Ctrl+K)

KEYBOARD SHORTCUTS:
Ctrl+L    - Focus strategy dropdown
Ctrl+I    - Focus text to index
Ctrl+F    - Open file browser
Ctrl+Q    - Focus query entry
Ctrl+R    - Retrieve (when query entered)
Ctrl+K    - Clear all data
Ctrl+H    - Show this help
F1        - Show this help
Ctrl+W    - Close window

TROUBLESHOOTING:
• "Missing migrations" error:
  → Run: alembic upgrade head
  
• "Missing services" error:
  → Check config/services.yaml has db_main configured
  
• "No results found":
  → Make sure you've indexed documents first
  → Try different keywords or broader search terms
  
• "Database connection failed":
  → Ensure PostgreSQL is running
  → Check connection string in config/services.yaml
  
• "Strategy load failed":
  → Check strategy YAML file syntax
  → Ensure all required services are configured

FEATURES:
• Real-time indexing with chunk counting
• Semantic search with relevance scoring
• Multiple strategy support
• Database persistence
• Comprehensive error handling

For more information, see the documentation."""
        
        # Create help window
        help_window = tk.Toplevel(self.root)
        help_window.title("Help - RAG Factory")
        help_window.geometry("700x600")
        
        # Create scrolled text
        help_frame = ttk.Frame(help_window)
        help_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        help_text_widget = tk.Text(help_frame, wrap=tk.WORD, font=('Arial', 10))
        help_scrollbar = ttk.Scrollbar(help_frame, command=help_text_widget.yview)
        help_text_widget.configure(yscrollcommand=help_scrollbar.set)
        
        help_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        help_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Insert help text
        help_text_widget.insert('1.0', help_text)
        help_text_widget.configure(state='disabled')  # Make read-only
        
        # Add close button
        close_btn = ttk.Button(help_window, text="Close", command=help_window.destroy)
        close_btn.pack(pady=5)
    
    def _show_about(self) -> None:
        """Show About dialog with version and credits."""
        about_text = """RAG Factory - Strategy Pair Tester
Version 1.0.0

A graphical interface for testing RAG (Retrieval-Augmented Generation) 
strategy pairs with real-time indexing and retrieval.

FEATURES:
• Real-time document indexing
• Semantic search with relevance scoring
• Multiple strategy support
• Database persistence
• Comprehensive error handling
• Log capture and viewing

BUILT WITH:
• Python 3.8+
• Tkinter (GUI framework)
• PostgreSQL (Database)
• SQLAlchemy (ORM)
• Alembic (Migrations)

LICENSE:
MIT License

For documentation and support, visit:
https://github.com/yourusername/rag-factory

© 2024 RAG Factory Team"""
        
        # Create about window
        about_window = tk.Toplevel(self.root)
        about_window.title("About RAG Factory")
        about_window.geometry("500x550")
        about_window.resizable(False, False)
        
        # Center the about window
        about_window.update_idletasks()
        x = (about_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (about_window.winfo_screenheight() // 2) - (550 // 2)
        about_window.geometry(f"+{x}+{y}")
        
        # Create scrolled text
        about_frame = ttk.Frame(about_window)
        about_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        about_text_widget = tk.Text(about_frame, wrap=tk.WORD, font=('Arial', 10), height=20)
        about_scrollbar = ttk.Scrollbar(about_frame, command=about_text_widget.yview)
        about_text_widget.configure(yscrollcommand=about_scrollbar.set)
        
        about_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        about_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Insert about text
        about_text_widget.insert('1.0', about_text)
        about_text_widget.configure(state='disabled')  # Make read-only
        
        # Add close button
        close_btn = ttk.Button(about_window, text="Close", command=about_window.destroy)
        close_btn.pack(pady=(0, 10))
    
    def _on_closing(self) -> None:
        """Handle window close event with confirmation if data exists."""
        # Check if there's data
        has_data = len(self.indexed_documents) > 0 or self.status_bar.document_count > 0
        
        if has_data:
            response = messagebox.askyesnocancel(
                "Confirm Exit",
                "You have indexed data in the database.\n\n"
                "Note: Data will remain in the database and can be accessed later.\n\n"
                "Do you want to exit?",
                icon='warning'
            )
            
            if response is None:  # Cancel
                return
            elif response:  # Yes
                self._cleanup_and_exit()
        else:
            # No data, just exit
            self._cleanup_and_exit()
    
    def _cleanup_and_exit(self) -> None:
        """Cleanup resources and exit application."""
        try:
            logger.info("Application closing")
            # Cleanup can be added here if needed
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            self.root.quit()
    
    def run(self) -> None:
        """Start the GUI application main loop."""
        # Bind keyboard shortcuts
        self.root.bind("<Control-l>", lambda e: self.strategy_dropdown.focus())
        self.root.bind("<Control-i>", lambda e: self.text_input.text.focus())
        self.root.bind("<Control-f>", lambda e: self._browse_file())
        self.root.bind("<Control-q>", lambda e: self.query_entry.focus())
        self.root.bind("<Control-r>", lambda e: self._retrieve() if self.query_var.get().strip() else None)
        self.root.bind("<Control-k>", lambda e: self._clear_all_data())
        self.root.bind("<Control-h>", lambda e: self._show_help())
        self.root.bind("<F1>", lambda e: self._show_help())
        self.root.bind("<Control-w>", lambda e: self.root.quit())
        
        logger.info("GUI application started")
        
        # Start main loop
        self.root.mainloop()


def main():
    """Entry point for running GUI standalone."""
    from rag_factory.registry.service_registry import ServiceRegistry
    from rag_factory.config.strategy_pair_manager import StrategyPairManager
    
    # Initialize services
    registry = ServiceRegistry("config/services.yaml")
    manager = StrategyPairManager(registry)
    
    # Create and run GUI
    app = RAGFactoryGUI(manager)
    app.run()


if __name__ == "__main__":
    main()
