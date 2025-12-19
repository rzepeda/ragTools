"""Scrolled text widget for displaying text with scrollbars.

This module provides a reusable text widget with automatic scrollbars
for use in configuration preview, text input, and results display.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional


class ScrolledText(ttk.Frame):
    """Text widget with automatic scrollbars.
    
    This widget combines a Text widget with vertical and horizontal scrollbars
    that appear automatically when needed. It can be configured as read-only
    or editable, and supports monospace fonts for code/data display.
    
    Example:
        >>> root = tk.Tk()
        >>> # Read-only config preview
        >>> config_text = ScrolledText(root, read_only=True, monospace=True)
        >>> config_text.pack(fill=tk.BOTH, expand=True)
        >>> config_text.set_text("name: test\\nvalue: 123")
        >>> 
        >>> # Editable text input
        >>> input_text = ScrolledText(root, read_only=False, height=10)
        >>> input_text.pack(fill=tk.BOTH, expand=True)
        >>> text = input_text.get_text()
    """
    
    def __init__(
        self,
        parent,
        read_only: bool = False,
        monospace: bool = False,
        height: Optional[int] = None,
        width: Optional[int] = None,
        wrap: str = tk.WORD,
        **kwargs
    ):
        """Initialize the scrolled text widget.
        
        Args:
            parent: Parent widget
            read_only: If True, text cannot be edited
            monospace: If True, use monospace font
            height: Height in lines (None for default)
            width: Width in characters (None for default)
            wrap: Text wrapping mode (tk.WORD, tk.CHAR, or tk.NONE)
            **kwargs: Additional arguments for ttk.Frame
        """
        super().__init__(parent, **kwargs)
        
        self.read_only = read_only
        
        # Create widgets
        self._create_widgets(monospace, height, width, wrap)
        
        # Configure read-only if needed
        if read_only:
            self.text.config(state=tk.DISABLED)
    
    def _create_widgets(
        self,
        monospace: bool,
        height: Optional[int],
        width: Optional[int],
        wrap: str
    ) -> None:
        """Create text and scrollbar widgets.
        
        Args:
            monospace: Whether to use monospace font
            height: Height in lines
            width: Width in characters
            wrap: Text wrapping mode
        """
        # Create text widget
        text_kwargs = {
            "wrap": wrap,
            "undo": True,
            "maxundo": -1
        }
        
        if height is not None:
            text_kwargs["height"] = height
        if width is not None:
            text_kwargs["width"] = width
        
        # Set font
        if monospace:
            text_kwargs["font"] = ("Courier", 10)
        
        self.text = tk.Text(self, **text_kwargs)
        
        # Create scrollbars
        self.v_scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.text.yview)
        self.h_scrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.text.xview)
        
        # Configure text widget to use scrollbars
        self.text.config(
            yscrollcommand=self.v_scrollbar.set,
            xscrollcommand=self.h_scrollbar.set
        )
        
        # Grid layout
        self.text.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
    
    def get_text(self) -> str:
        """Get the current text content.
        
        Returns:
            Text content as string
        """
        return self.text.get("1.0", tk.END).rstrip()
    
    def set_text(self, content: str) -> None:
        """Set the text content.
        
        Args:
            content: Text content to set
        """
        # Temporarily enable if read-only
        was_disabled = False
        if self.read_only:
            self.text.config(state=tk.NORMAL)
            was_disabled = True
        
        # Clear and set new content
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", content)
        
        # Restore read-only state
        if was_disabled:
            self.text.config(state=tk.DISABLED)
    
    def append_text(self, content: str) -> None:
        """Append text to the current content.
        
        Args:
            content: Text to append
        """
        # Temporarily enable if read-only
        was_disabled = False
        if self.read_only:
            self.text.config(state=tk.NORMAL)
            was_disabled = True
        
        # Append content
        self.text.insert(tk.END, content)
        
        # Auto-scroll to end
        self.text.see(tk.END)
        
        # Restore read-only state
        if was_disabled:
            self.text.config(state=tk.DISABLED)
    
    def clear(self) -> None:
        """Clear all text content."""
        self.set_text("")
    
    def is_empty(self) -> bool:
        """Check if the text widget is empty.
        
        Returns:
            True if empty, False otherwise
        """
        return len(self.get_text().strip()) == 0
    
    def set_placeholder(self, placeholder: str) -> None:
        """Set placeholder text (shown when empty).
        
        Args:
            placeholder: Placeholder text
        """
        if not self.read_only and self.is_empty():
            self.text.insert("1.0", placeholder)
            self.text.config(foreground="gray")
            
            # Bind focus events to clear/restore placeholder
            def on_focus_in(event):
                if self.get_text() == placeholder:
                    self.clear()
                    self.text.config(foreground="black")
            
            def on_focus_out(event):
                if self.is_empty():
                    self.set_placeholder(placeholder)
            
            self.text.bind("<FocusIn>", on_focus_in)
            self.text.bind("<FocusOut>", on_focus_out)
