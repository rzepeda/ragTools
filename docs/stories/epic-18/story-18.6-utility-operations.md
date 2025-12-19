# Story 18.6: Implement Utility Operations (Clear, Logs, Help)

**Story ID:** 18.6  
**Epic:** Epic 18 - Minimal GUI for RAG Strategy Testing  
**Story Points:** 3  
**Priority:** Medium  
**Dependencies:** Story 18.5 (Retrieval Operations)

---

## User Story

**As a** user  
**I want** utility operations for managing data and getting help  
**So that** I can reset state and troubleshoot issues

---

## Detailed Requirements

### Functional Requirements

1. **Clear All Data**
   - Confirmation dialog before deleting
   - Delete indexed documents for current strategy only
   - Clear database tables associated with strategy
   - Reset document and chunk counters
   - Update status bar with result

2. **View Logs**
   - Capture application logs in memory buffer
   - Display logs in popup window
   - Logs are scrollable and read-only
   - Refresh button to update log display
   - Auto-scroll to bottom on open

3. **Help Dialog**
   - Show keyboard shortcuts
   - Show usage tips
   - Show troubleshooting guide
   - Show quick start instructions
   - Scrollable help text

4. **Settings Placeholder**
   - Show "future enhancement" message
   - Placeholder for future settings dialog

### Non-Functional Requirements

1. **Safety**
   - Clear data requires explicit confirmation
   - Warning shows what will be deleted
   - Cannot be undone message displayed

2. **Usability**
   - Help is easily accessible
   - Logs are useful for debugging
   - Clear feedback on all operations

---

## Acceptance Criteria

### AC1: Clear All Data
- [ ] Confirmation dialog shown before clearing
- [ ] Dialog shows current strategy name
- [ ] Dialog shows document and chunk counts
- [ ] Dialog warns action cannot be undone
- [ ] Data cleared only if user confirms
- [ ] Database tables cleared correctly
- [ ] Counters reset to zero
- [ ] Success message displayed
- [ ] Status bar updated

### AC2: View Logs
- [ ] Logs window opens on button click
- [ ] Logs displayed in scrollable textbox
- [ ] Logs are read-only
- [ ] Refresh button updates log display
- [ ] Auto-scrolls to bottom on open
- [ ] Window can be closed
- [ ] Logs captured from application start

### AC3: Help Dialog
- [ ] Help window opens on button click
- [ ] Keyboard shortcuts listed
- [ ] Usage tips provided
- [ ] Troubleshooting guide included
- [ ] Quick start instructions shown
- [ ] Help text is scrollable
- [ ] Window can be closed

### AC4: Settings Placeholder
- [ ] Settings button shows placeholder message
- [ ] Message indicates future enhancement

### AC5: Keyboard Shortcuts
- [ ] Ctrl+L focuses strategy dropdown
- [ ] Ctrl+I focuses text to index
- [ ] Ctrl+F opens file browser
- [ ] Ctrl+Q focuses query entry
- [ ] Ctrl+R triggers retrieve (if enabled)
- [ ] Ctrl+K triggers clear data
- [ ] Ctrl+H shows help

---

## Technical Specifications

See Epic 18 document lines 1083-1326 for complete pseudocode.

### Logging Setup

```python
# Capture logs in memory buffer
self.log_buffer = []

class GUILogHandler(logging.Handler):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer
    
    def emit(self, record):
        log_entry = self.format(record)
        self.buffer.append(log_entry)
        
        # Keep only last 1000 messages
        if len(self.buffer) > 1000:
            self.buffer.pop(0)
```

### Help Text Example

```
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

TROUBLESHOOTING:
- "Missing migrations" → Run: alembic upgrade head
- "Missing services" → Check config/services.yaml
- "No results" → Index documents first
```

---

## Testing Strategy

### Unit Tests
- [ ] Test clear data confirmation dialog
- [ ] Test clear data execution
- [ ] Test counter reset
- [ ] Test log capture
- [ ] Test log display
- [ ] Test help dialog display
- [ ] Test keyboard shortcuts

### Integration Tests
- [ ] Test clear data with real database
- [ ] Test log capture during operations
- [ ] Test keyboard shortcuts in full workflow

---

## Story Points Breakdown

- **Clear Data Implementation:** 1 point
- **Logs Implementation:** 1 point
- **Help and Shortcuts:** 1 point

**Total:** 3 points

---

## Dependencies

- Story 18.5 (Retrieval Operations) - MUST BE COMPLETED

---

## Notes

- Clear data is destructive - ensure proper confirmation
- Logs should be helpful for debugging issues
- Help should be comprehensive but concise
- Keyboard shortcuts improve power user experience
