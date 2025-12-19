# Epic 18: Minimal GUI for RAG Strategy Testing - Stories

This directory contains the user stories for Epic 18, which implements a lightweight, single-window GUI application for testing RAG strategy pairs.

## Epic Overview

**Epic Goal:** Create a lightweight, single-window GUI application for testing RAG strategy pairs with minimal dependencies, enabling visual validation of indexing and retrieval workflows without requiring CLI knowledge.

**Total Story Points:** 47 points (~1 month)

**Status:** Ready for implementation

## Stories

### Story 18.1: Design GUI Layout and Component Specification
- **Points:** 5
- **Priority:** Critical
- **File:** [story-18.1-gui-design-layout.md](./story-18.1-gui-design-layout.md)
- **Description:** Design complete GUI layout with wireframes, component specifications, and interaction flows.

### Story 18.2: Implement Core GUI Framework (tkinter)
- **Points:** 13
- **Priority:** Critical
- **File:** [story-18.2-core-gui-framework.md](./story-18.2-core-gui-framework.md)
- **Description:** Build the basic GUI window and component layout using tkinter.
- **Dependencies:** Story 18.1

### Story 18.3: Integrate StrategyPairManager (Backend Connection)
- **Points:** 8
- **Priority:** Critical
- **File:** [story-18.3-backend-integration.md](./story-18.3-backend-integration.md)
- **Description:** Connect GUI to Epic 17 StrategyPairManager for loading and using strategy pairs.
- **Dependencies:** Story 18.2, Epic 17

### Story 18.4: Implement Indexing Operations
- **Points:** 5
- **Priority:** High
- **File:** [story-18.4-indexing-operations.md](./story-18.4-indexing-operations.md)
- **Description:** Implement text and file indexing through the GUI.
- **Dependencies:** Story 18.3

### Story 18.5: Implement Retrieval Operations
- **Points:** 5
- **Priority:** High
- **File:** [story-18.5-retrieval-operations.md](./story-18.5-retrieval-operations.md)
- **Description:** Implement query execution and results display.
- **Dependencies:** Story 18.4

### Story 18.6: Implement Utility Operations (Clear, Logs, Help)
- **Points:** 3
- **Priority:** Medium
- **File:** [story-18.6-utility-operations.md](./story-18.6-utility-operations.md)
- **Description:** Add utility features for data management and user help.
- **Dependencies:** Story 18.5

### Story 18.7: Add Polish and User Experience Enhancements
- **Points:** 3
- **Priority:** Medium
- **File:** [story-18.7-polish-and-ux.md](./story-18.7-polish-and-ux.md)
- **Description:** Apply visual styling, tooltips, and UX improvements.
- **Dependencies:** Story 18.6

### Story 18.8: Testing and Documentation
- **Points:** 5
- **Priority:** High
- **File:** [story-18.8-testing-and-documentation.md](./story-18.8-testing-and-documentation.md)
- **Description:** Comprehensive testing and documentation for the GUI.
- **Dependencies:** Story 18.7

## Sprint Planning

**Sprint 19 (Epic 18 - GUI Development):**
- **Week 1:** Stories 18.1, 18.2 (18 points)
- **Week 2:** Stories 18.2 (continued), 18.3 (8 points)
- **Week 3:** Stories 18.4, 18.5, 18.6 (13 points)
- **Week 4:** Stories 18.7, 18.8 (8 points)

**Total:** 47 points (~1 month)

## Technical Stack

- **GUI Framework:** tkinter (built into Python, no extra dependencies)
- **Threading:** Python threading module
- **File Dialogs:** tkinter.filedialog
- **Message Boxes:** tkinter.messagebox
- **Logging:** Python logging module

## Key Features

1. **Strategy Selection:** Load and preview strategy pair configurations
2. **Text Indexing:** Index text directly from GUI textbox
3. **File Indexing:** Browse and index files
4. **Query & Retrieval:** Execute queries and view formatted results
5. **Status Feedback:** Real-time status updates and error messages
6. **Utility Operations:** Clear data, view logs, access help
7. **Polish & UX:** Professional styling, tooltips, keyboard shortcuts

## Design Philosophy

- **Lightweight:** Single file, minimal dependencies (tkinter built into Python)
- **Development Tool:** Not production-ready, focused on testing
- **Delegates to Library:** Uses StrategyPairManager, not reimplementing logic
- **Read-Only Configuration:** Shows config but doesn't edit it (use text editor for that)

## Dependencies

- Epic 17 (Strategy Pair Configuration) - COMPLETED ✅
- Epic 14 (CLI Enhancements - for configuration validation)
- Python 3.10+
- PostgreSQL with pgvector extension
- Service registry configured (`config/services.yaml`)
- At least one strategy pair in `strategies/` directory

## Success Criteria

- ✅ GUI launches without errors
- ✅ All UI components render correctly
- ✅ Strategy pairs can be loaded and displayed
- ✅ Text and file indexing works
- ✅ Query retrieval works and displays results
- ✅ Error handling shows user-friendly messages
- ✅ Threading doesn't freeze GUI
- ✅ Clear data works without errors
- ✅ Logs can be viewed
- ✅ Help dialog shows useful information
- ✅ All tests pass
- ✅ Documentation complete with screenshots
- ✅ Cross-platform tested (Windows, Linux, macOS)

## Benefits

### User Experience
- Visual interface for non-CLI users
- Immediate feedback on operations
- No command-line knowledge required
- Easy demo tool for presentations

### Developer Experience
- Quick way to test strategy pairs
- Visual validation of configurations
- Easy debugging with logs
- Fast iteration during development

### Quality
- Delegates to Epic 17 implementation (no logic duplication)
- Proper error handling and user feedback
- Thread-safe operations
- Clean separation of GUI and business logic

## Future Enhancements (Post-Epic 18)

Possible additions in future epics:
- Settings dialog for configuring service registry
- Visual strategy pair editor (YAML editor)
- Performance graphs and metrics visualization
- Multi-strategy comparison view
- Batch document indexing with progress bar
- Export results to CSV/JSON
- Dark theme support
- Internationalization (i18n)
- Plugin system for custom visualizations

## Notes

- This is a development tool, not a production interface
- Focus is on simplicity and usefulness, not feature completeness
- GUI delegates to Epic 17 - no business logic duplication
- tkinter chosen for zero extra dependencies
- Single file implementation for easy distribution
- Can be extended later with more features if needed
- Thread safety is critical - all backend calls in background threads
- Read-only config - use text editor for YAML editing (keeps GUI simple)

## Related Documentation

- [Epic 18 Full Specification](../../epics/epic-18-gui.md)
- [Epic 17 Strategy Pair Configuration](../../epics/epic-17-strategy-pairs.md)
- [Epic 04 Stories (Template Reference)](../Completed/epic-04/)
