"""Status bar component for displaying operation status and statistics.

This module provides a reusable status bar widget that displays:
- Status indicator (Ready/Success/Error)
- Document count
- Chunk count
- Last action timestamp
"""

import tkinter as tk
from tkinter import ttk
from datetime import datetime
from typing import Optional


class StatusBar(ttk.Frame):
    """Status bar widget for displaying operation status and statistics.
    
    The status bar displays:
    - Status indicator with color coding (⚫ Ready / 🟢 Success / 🔴 Error)
    - Document count
    - Chunk count
    - Last action timestamp
    
    Example:
        >>> root = tk.Tk()
        >>> status_bar = StatusBar(root)
        >>> status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        >>> status_bar.set_status("success", "Indexed 5 documents")
        >>> status_bar.update_counts(documents=5, chunks=25)
    """
    
    def __init__(self, parent, **kwargs):
        """Initialize the status bar.
        
        Args:
            parent: Parent widget
            **kwargs: Additional arguments for ttk.Frame
        """
        super().__init__(parent, **kwargs)
        
        # Status indicators
        self.status_indicators = {
            "ready": "⚫ Ready",
            "success": "🟢 Success",
            "error": "🔴 Error",
            "working": "🟡 Working"
        }
        
        # Create widgets
        self._create_widgets()
        
        # Initialize state
        self.document_count = 0
        self.chunk_count = 0
        self.last_action_time: Optional[datetime] = None
        
        # Set initial status
        self.set_status("ready")
    
    def _create_widgets(self) -> None:
        """Create status bar widgets."""
        # Status label
        self.status_label = ttk.Label(
            self,
            text="⚫ Ready",
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 2)
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Separator
        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=2)
        
        # Document count label
        self.doc_count_label = ttk.Label(
            self,
            text="Documents: 0",
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 2),
            width=15
        )
        self.doc_count_label.pack(side=tk.LEFT)
        
        # Separator
        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=2)
        
        # Chunk count label
        self.chunk_count_label = ttk.Label(
            self,
            text="Chunks: 0",
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 2),
            width=12
        )
        self.chunk_count_label.pack(side=tk.LEFT)
        
        # Separator
        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=2)
        
        # Last action label
        self.last_action_label = ttk.Label(
            self,
            text="Last action: Never",
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 2),
            width=25
        )
        self.last_action_label.pack(side=tk.LEFT)
    
    def set_status(self, status: str, message: Optional[str] = None) -> None:
        """Set the status indicator.
        
        Args:
            status: Status type ("ready", "success", "error", "working")
            message: Optional custom message to display
        """
        if status in self.status_indicators:
            base_text = self.status_indicators[status]
        else:
            base_text = f"⚫ {status}"
        
        if message:
            display_text = f"{base_text} | {message}"
        else:
            display_text = base_text
        
        self.status_label.config(text=display_text)
        
        # Update last action time for non-ready statuses
        if status != "ready":
            self.last_action_time = datetime.now()
            self._update_last_action()
    
    def update_counts(self, documents: Optional[int] = None, chunks: Optional[int] = None) -> None:
        """Update document and chunk counts.
        
        Args:
            documents: New document count (None to keep current)
            chunks: New chunk count (None to keep current)
        """
        if documents is not None:
            self.document_count = documents
            self.doc_count_label.config(text=f"Documents: {self.document_count}")
        
        if chunks is not None:
            self.chunk_count = chunks
            self.chunk_count_label.config(text=f"Chunks: {self.chunk_count}")
    
    def increment_counts(self, documents: int = 0, chunks: int = 0) -> None:
        """Increment document and chunk counts.
        
        Args:
            documents: Number of documents to add
            chunks: Number of chunks to add
        """
        self.update_counts(
            documents=self.document_count + documents,
            chunks=self.chunk_count + chunks
        )
    
    def reset_counts(self) -> None:
        """Reset all counts to zero."""
        self.update_counts(documents=0, chunks=0)
    
    def _update_last_action(self) -> None:
        """Update the last action timestamp display."""
        if self.last_action_time:
            elapsed = datetime.now() - self.last_action_time
            
            if elapsed.total_seconds() < 60:
                time_str = f"{elapsed.total_seconds():.1f}s ago"
            elif elapsed.total_seconds() < 3600:
                time_str = f"{elapsed.total_seconds() / 60:.1f}m ago"
            else:
                time_str = f"{elapsed.total_seconds() / 3600:.1f}h ago"
            
            self.last_action_label.config(text=f"Last action: {time_str}")
        else:
            self.last_action_label.config(text="Last action: Never")
    
    def start_periodic_update(self, interval_ms: int = 1000) -> None:
        """Start periodic updates of the last action timestamp.
        
        Args:
            interval_ms: Update interval in milliseconds
        """
        def update():
            self._update_last_action()
            self.after(interval_ms, update)
        
        update()
