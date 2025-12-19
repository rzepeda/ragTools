# Story 18.1: Design GUI Layout and Component Specification

**Story ID:** 18.1  
**Epic:** Epic 18 - Minimal GUI for RAG Strategy Testing  
**Story Points:** 5  
**Priority:** Critical  
**Dependencies:** Epic 17 (Strategy Pair Configuration)

---

## User Story

**As a** developer  
**I want** a detailed GUI layout specification  
**So that** implementation is straightforward and consistent

---

## Detailed Requirements

### Functional Requirements

1. **Window Layout Definition**
   - Define complete window layout with dimensions (900x800px)
   - Specify all UI components (dropdowns, textboxes, buttons)
   - Define component behavior (enable/disable states)
   - Define status feedback mechanism
   - Define error display strategy

2. **Component Specifications**
   - Strategy dropdown for selecting strategy pairs
   - Configuration preview textbox (read-only, scrollable)
   - Text indexing section with multiline textbox
   - File indexing section with file browser
   - Query and retrieval section with results display
   - Status bar with operation feedback
   - Utility buttons (Clear Data, View Logs, Help)

3. **Component Interaction Flow**
   - User workflow for loading strategy and indexing text
   - User workflow for indexing files
   - User workflow for querying
   - Error handling workflows

4. **Visual Design**
   - Wireframe diagram showing component layout
   - Component interaction flow diagram
   - Color scheme and typography specifications
   - Spacing and padding guidelines

### Non-Functional Requirements

1. **Usability**
   - Intuitive layout following standard GUI patterns
   - Clear visual hierarchy
   - Consistent spacing and alignment
   - Accessible color contrast

2. **Responsiveness**
   - Window is resizable with minimum size constraints
   - Components adapt to window size changes
   - Scrollbars appear when needed

3. **Documentation**
   - Complete component specifications
   - Interaction flow documentation
   - Extension guidelines for future enhancements

---

## Acceptance Criteria

### AC1: Window Layout Specification
- [ ] Complete window layout defined with dimensions
- [ ] All UI sections clearly specified
- [ ] Component positioning documented
- [ ] Wireframe diagram created
- [ ] Minimum window size defined (800x600px)

### AC2: Component Specifications
- [ ] Strategy dropdown specification complete
- [ ] Configuration preview textbox specification complete
- [ ] Text indexing components specified
- [ ] File indexing components specified
- [ ] Query and retrieval components specified
- [ ] Status bar specification complete
- [ ] Utility buttons specified

### AC3: Component Behavior
- [ ] Enable/disable states documented for all buttons
- [ ] Input validation rules specified
- [ ] Error display strategy defined
- [ ] Status feedback mechanism defined
- [ ] Threading considerations documented

### AC4: Interaction Flows
- [ ] User workflow 1 (Load Strategy and Index Text) documented
- [ ] User workflow 2 (Index File) documented
- [ ] User workflow 3 (Query) documented
- [ ] Error handling workflows documented
- [ ] Component interaction flow diagram created

### AC5: Visual Design
- [ ] Color scheme defined
- [ ] Typography specifications complete
- [ ] Spacing and padding guidelines documented
- [ ] Scrollbar specifications complete
- [ ] Icon specifications (if applicable)

### AC6: Extension Guidelines
- [ ] Documentation for adding new components
- [ ] Guidelines for modifying layout
- [ ] Best practices for maintaining consistency

---

## Technical Specifications

### Window Layout

```
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
```

### Component Specifications

```python
COMPONENT_SPECS:
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

### Component Interaction Flow

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
```

### Visual Design Specifications

**Window Size:**
- Default: 900px (width) x 800px (height)
- Minimum: 800px (width) x 600px (height)
- Resizable: Yes

**Font:**
- System default for labels and buttons
- Monospace (Courier or Consolas) for config/results

**Colors:**
- Light theme (white background, dark text)
- Status indicators:
  - ⚫ Ready (gray)
  - 🟢 Success (green)
  - 🔴 Error (red)

**Spacing:**
- Section padding: 10px
- Widget padding: 5px
- Internal padding: 3px

---

## Implementation Notes

1. **Layout Manager:** Use `grid()` for precise component positioning
2. **Scrollbars:** Use `pack()` for scrollbars attached to textboxes
3. **Resizing:** Configure row/column weights for proper resizing behavior
4. **Threading:** All backend operations must run in background threads
5. **GUI Updates:** Use `root.after(0, callback)` to update GUI from background threads

---

## Testing Strategy

### Unit Tests
- [ ] Test component specifications are complete
- [ ] Test interaction flows are documented
- [ ] Test all acceptance criteria are met

### Documentation Review
- [ ] Wireframe diagram reviewed
- [ ] Component specifications reviewed
- [ ] Interaction flows reviewed
- [ ] Extension guidelines reviewed

---

## Documentation Deliverables

1. **Wireframe Diagram:** ASCII art or image showing layout
2. **Component Specifications:** Detailed specs for each component
3. **Interaction Flow Diagram:** Visual representation of user workflows
4. **Extension Guidelines:** Documentation for future enhancements

---

## Story Points Breakdown

- **Research and Design:** 2 points
- **Wireframe Creation:** 1 point
- **Component Specification:** 1 point
- **Documentation:** 1 point

**Total:** 5 points

---

## Dependencies

- Epic 17 (Strategy Pair Configuration) - COMPLETED ✅
- Understanding of tkinter capabilities
- Understanding of StrategyPairManager API

---

## Notes

- This story is design-only, no code implementation
- Focus on clarity and completeness of specifications
- Specifications should enable straightforward implementation in Story 18.2
- Keep design simple and focused on core functionality
- Avoid feature creep - this is a development tool, not a production UI
