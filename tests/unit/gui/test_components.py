"""Unit tests for GUI components.

These tests verify the reusable GUI components (StatusBar, ScrolledText).
"""

import pytest
import tkinter as tk

# Skip all tests if tkinter is not available
pytest.importorskip("tkinter")

from rag_factory.gui.components import StatusBar, ScrolledText


@pytest.fixture
def root():
    """Create a tkinter root window for testing.
    
    Yields:
        tk.Tk instance
    """
    root = tk.Tk()
    yield root
    try:
        root.destroy()
    except:
        pass


class TestStatusBar:
    """Tests for StatusBar component."""
    
    def test_status_bar_creation(self, root):
        """Test StatusBar can be created."""
        status_bar = StatusBar(root)
        assert status_bar is not None
        assert isinstance(status_bar, StatusBar)
    
    def test_initial_status(self, root):
        """Test StatusBar has correct initial status."""
        status_bar = StatusBar(root)
        
        # Initial status should be "Ready"
        status_text = status_bar.status_label.cget('text')
        assert "Ready" in status_text
    
    def test_set_status_ready(self, root):
        """Test setting status to ready."""
        status_bar = StatusBar(root)
        status_bar.set_status("ready")
        
        status_text = status_bar.status_label.cget('text')
        assert "⚫ Ready" in status_text
    
    def test_set_status_success(self, root):
        """Test setting status to success."""
        status_bar = StatusBar(root)
        status_bar.set_status("success", "Operation completed")
        
        status_text = status_bar.status_label.cget('text')
        assert "🟢 Success" in status_text
        assert "Operation completed" in status_text
    
    def test_set_status_error(self, root):
        """Test setting status to error."""
        status_bar = StatusBar(root)
        status_bar.set_status("error", "Something went wrong")
        
        status_text = status_bar.status_label.cget('text')
        assert "🔴 Error" in status_text
        assert "Something went wrong" in status_text
    
    def test_set_status_working(self, root):
        """Test setting status to working."""
        status_bar = StatusBar(root)
        status_bar.set_status("working", "Processing...")
        
        status_text = status_bar.status_label.cget('text')
        assert "🟡 Working" in status_text
        assert "Processing..." in status_text
    
    def test_initial_counts(self, root):
        """Test initial document and chunk counts are zero."""
        status_bar = StatusBar(root)
        
        assert status_bar.document_count == 0
        assert status_bar.chunk_count == 0
    
    def test_update_document_count(self, root):
        """Test updating document count."""
        status_bar = StatusBar(root)
        status_bar.update_counts(documents=5)
        
        assert status_bar.document_count == 5
        doc_text = status_bar.doc_count_label.cget('text')
        assert "5" in doc_text
    
    def test_update_chunk_count(self, root):
        """Test updating chunk count."""
        status_bar = StatusBar(root)
        status_bar.update_counts(chunks=25)
        
        assert status_bar.chunk_count == 25
        chunk_text = status_bar.chunk_count_label.cget('text')
        assert "25" in chunk_text
    
    def test_increment_counts(self, root):
        """Test incrementing counts."""
        status_bar = StatusBar(root)
        status_bar.update_counts(documents=5, chunks=20)
        status_bar.increment_counts(documents=3, chunks=10)
        
        assert status_bar.document_count == 8
        assert status_bar.chunk_count == 30
    
    def test_reset_counts(self, root):
        """Test resetting counts to zero."""
        status_bar = StatusBar(root)
        status_bar.update_counts(documents=10, chunks=50)
        status_bar.reset_counts()
        
        assert status_bar.document_count == 0
        assert status_bar.chunk_count == 0


class TestScrolledText:
    """Tests for ScrolledText component."""
    
    def test_scrolled_text_creation(self, root):
        """Test ScrolledText can be created."""
        scrolled_text = ScrolledText(root)
        assert scrolled_text is not None
        assert isinstance(scrolled_text, ScrolledText)
    
    def test_scrolled_text_has_text_widget(self, root):
        """Test ScrolledText has internal text widget."""
        scrolled_text = ScrolledText(root)
        assert hasattr(scrolled_text, 'text')
        assert isinstance(scrolled_text.text, tk.Text)
    
    def test_scrolled_text_has_scrollbars(self, root):
        """Test ScrolledText has scrollbars."""
        scrolled_text = ScrolledText(root)
        assert hasattr(scrolled_text, 'v_scrollbar')
        assert hasattr(scrolled_text, 'h_scrollbar')
    
    def test_read_only_mode(self, root):
        """Test ScrolledText in read-only mode."""
        scrolled_text = ScrolledText(root, read_only=True)
        
        # Text widget should be disabled
        state = str(scrolled_text.text.cget('state'))
        assert state == 'disabled'
    
    def test_editable_mode(self, root):
        """Test ScrolledText in editable mode."""
        scrolled_text = ScrolledText(root, read_only=False)
        
        # Text widget should be normal (editable)
        state = str(scrolled_text.text.cget('state'))
        assert state == 'normal'
    
    def test_monospace_font(self, root):
        """Test ScrolledText with monospace font."""
        scrolled_text = ScrolledText(root, monospace=True)
        
        # Font should be Courier
        font = scrolled_text.text.cget('font')
        assert 'Courier' in str(font)
    
    def test_set_text(self, root):
        """Test setting text content."""
        scrolled_text = ScrolledText(root)
        scrolled_text.set_text("Hello, World!")
        
        content = scrolled_text.get_text()
        assert content == "Hello, World!"
    
    def test_get_text(self, root):
        """Test getting text content."""
        scrolled_text = ScrolledText(root)
        scrolled_text.text.insert("1.0", "Test content")
        
        content = scrolled_text.get_text()
        assert "Test content" in content
    
    def test_append_text(self, root):
        """Test appending text."""
        scrolled_text = ScrolledText(root)
        scrolled_text.set_text("Line 1\n")
        scrolled_text.append_text("Line 2\n")
        
        content = scrolled_text.get_text()
        assert "Line 1" in content
        assert "Line 2" in content
    
    def test_clear_text(self, root):
        """Test clearing text."""
        scrolled_text = ScrolledText(root)
        scrolled_text.set_text("Some content")
        scrolled_text.clear()
        
        content = scrolled_text.get_text()
        assert content == ""
    
    def test_is_empty(self, root):
        """Test checking if text is empty."""
        scrolled_text = ScrolledText(root)
        
        # Initially empty
        assert scrolled_text.is_empty()
        
        # After adding text
        scrolled_text.set_text("Not empty")
        assert not scrolled_text.is_empty()
        
        # After clearing
        scrolled_text.clear()
        assert scrolled_text.is_empty()
    
    def test_set_text_in_read_only_mode(self, root):
        """Test setting text in read-only mode."""
        scrolled_text = ScrolledText(root, read_only=True)
        scrolled_text.set_text("Read-only content")
        
        content = scrolled_text.get_text()
        assert content == "Read-only content"
        
        # Text widget should still be disabled
        state = str(scrolled_text.text.cget('state'))
        assert state == 'disabled'
    
    def test_height_parameter(self, root):
        """Test setting height parameter."""
        scrolled_text = ScrolledText(root, height=15)
        
        height = scrolled_text.text.cget('height')
        assert height == 15
    
    def test_width_parameter(self, root):
        """Test setting width parameter."""
        scrolled_text = ScrolledText(root, width=80)
        
        width = scrolled_text.text.cget('width')
        assert width == 80
