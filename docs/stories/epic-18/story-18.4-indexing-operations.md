# Story 18.4: Implement Indexing Operations

**Story ID:** 18.4  
**Epic:** Epic 18 - Minimal GUI for RAG Strategy Testing  
**Story Points:** 5  
**Priority:** High  
**Dependencies:** Story 18.3 (Backend Integration)

---

## User Story

**As a** user  
**I want** to index text and files through the GUI  
**So that** I can populate the database for testing retrieval

---

## Detailed Requirements

### Functional Requirements

1. **Text Indexing**
   - Capture text from textbox
   - Create document from text content
   - Call indexing pipeline with document
   - Display success message with document/chunk count
   - Update document and chunk counters
   - Clear textbox after successful indexing (optional)

2. **File Indexing**
   - File browser dialog for selecting files
   - Read file content (UTF-8 encoding)
   - Create document from file content
   - Call indexing pipeline with document
   - Display success message with file name and counts
   - Handle file encoding errors

3. **Progress Indication**
   - Disable buttons during indexing
   - Update status bar with progress message
   - Optional: Show spinner or progress indicator

4. **Error Handling**
   - Handle empty text input
   - Handle missing files
   - Handle file encoding errors
   - Handle indexing pipeline errors
   - Display user-friendly error messages

5. **Thread Safety**
   - Run indexing in background threads
   - Don't freeze GUI during indexing
   - Update GUI from main thread only

### Non-Functional Requirements

1. **Performance**
   - Indexing doesn't block GUI
   - Status updates appear immediately
   - Background thread completes within reasonable time

2. **Usability**
   - Clear feedback during operations
   - Success/error messages are informative
   - Counters update accurately

---

## Acceptance Criteria

### AC1: Text Indexing
- [ ] Text captured from textbox correctly
- [ ] Document created with unique ID
- [ ] Indexing pipeline called with document
- [ ] Success message shows document and chunk count
- [ ] Document counter incremented
- [ ] Chunk counter incremented
- [ ] Status bar updated with timing information
- [ ] Textbox optionally cleared after success

### AC2: File Indexing
- [ ] Browse button opens file dialog
- [ ] File path populated in entry field
- [ ] File content read correctly (UTF-8)
- [ ] Document created with file metadata
- [ ] Indexing pipeline called with document
- [ ] Success message shows filename and counts
- [ ] Counters updated correctly
- [ ] File encoding errors handled gracefully

### AC3: Progress Indication
- [ ] Index Text button disabled during operation
- [ ] Index File button disabled during operation
- [ ] Status bar shows "Indexing..." message
- [ ] Status bar shows success/error after completion
- [ ] Buttons re-enabled after operation

### AC4: Error Handling
- [ ] Empty text input shows warning
- [ ] Missing file shows error
- [ ] File encoding error shows helpful message
- [ ] Indexing errors displayed with details
- [ ] All errors logged appropriately

### AC5: Thread Safety
- [ ] Indexing runs in background thread
- [ ] GUI remains responsive during indexing
- [ ] GUI updates only from main thread
- [ ] No race conditions or crashes

---

## Technical Specifications

See Epic 18 document lines 666-898 for complete pseudocode.

### Key Implementation Points

```python
# Threading pattern for indexing
def on_index_text(self):
    # Disable button
    self.index_text_button.config(state="disabled")
    
    # Run in background thread
    thread = Thread(target=self._index_text_worker, args=(text_content,))
    thread.start()

def _index_text_worker(self, text_content: str):
    try:
        # Do indexing work
        result = self.indexing_pipeline.process([document], context)
        
        # Update GUI on main thread
        self.root.after(0, self._on_index_text_success, result, elapsed)
    except Exception as e:
        self.root.after(0, self._on_index_text_error, str(e))
```

---

## Testing Strategy

### Unit Tests
- [ ] Test text indexing with valid input
- [ ] Test file indexing with valid file
- [ ] Test empty text input handling
- [ ] Test missing file handling
- [ ] Test file encoding error handling
- [ ] Test counter updates
- [ ] Test button state changes

### Integration Tests
- [ ] Test end-to-end text indexing
- [ ] Test end-to-end file indexing
- [ ] Test multiple indexing operations
- [ ] Test concurrent indexing (if supported)

---

## Story Points Breakdown

- **Text Indexing Implementation:** 2 points
- **File Indexing Implementation:** 2 points
- **Testing:** 1 point

**Total:** 5 points

---

## Dependencies

- Story 18.3 (Backend Integration) - MUST BE COMPLETED
- Indexing pipeline from Epic 17

---

## Notes

- Focus on thread safety to prevent GUI freezing
- Use `root.after(0, callback)` for all GUI updates from background threads
- Keep error messages user-friendly and actionable
- Document and chunk counters should persist across operations
