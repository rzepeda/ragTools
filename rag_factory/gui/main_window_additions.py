    
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
