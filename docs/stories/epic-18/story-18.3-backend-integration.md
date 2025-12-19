# Story 18.3: Integrate StrategyPairManager (Backend Connection)

**Story ID:** 18.3  
**Epic:** Epic 18 - Minimal GUI for RAG Strategy Testing  
**Story Points:** 8  
**Priority:** Critical  
**Dependencies:** Story 18.2 (Core GUI Framework), Epic 17 (Strategy Pair Configuration)

---

## User Story

**As a** developer  
**I want** the GUI to load and use strategy pairs from Epic 17  
**So that** the GUI leverages existing functionality

---

## Detailed Requirements

### Functional Requirements

1. **Service Registry Initialization**
   - Initialize ServiceRegistry on GUI startup
   - Load service configuration from `config/services.yaml`
   - Verify required services are available
   - Handle missing service configuration gracefully

2. **StrategyPairManager Integration**
   - Initialize StrategyPairManager with service registry
   - Load available strategy pairs from `strategies/` directory
   - Populate strategy dropdown with available pairs
   - Handle empty strategies directory

3. **Strategy Loading**
   - Load selected strategy configuration (YAML)
   - Display configuration in read-only textbox
   - Instantiate indexing and retrieval pipelines
   - Store pipeline references for later use

4. **Migration Validation**
   - Validate migrations before allowing operations
   - Detect missing migrations
   - Display user-friendly error messages
   - Suggest upgrade commands

5. **Error Handling**
   - Handle missing services gracefully
   - Handle missing migrations with upgrade suggestions
   - Handle invalid YAML configurations
   - Handle database connection errors
   - Display errors in status bar and/or popup dialogs

### Non-Functional Requirements

1. **Reliability**
   - Graceful degradation on errors
   - Clear error messages for troubleshooting
   - No crashes on configuration errors

2. **Performance**
   - Strategy loading completes in <2 seconds
   - Service registry initialization in <1 second

3. **Usability**
   - User-friendly error messages
   - Actionable suggestions for fixing errors
   - Clear status feedback during operations

---

## Acceptance Criteria

### AC1: Service Registry Initialization
- [ ] ServiceRegistry initialized on GUI startup
- [ ] Service configuration loaded from `config/services.yaml`
- [ ] Required services verified (at minimum `db_main`)
- [ ] Missing service configuration handled with error message
- [ ] Database connection errors handled gracefully

### AC2: StrategyPairManager Integration
- [ ] StrategyPairManager initialized with service registry
- [ ] Available strategy pairs loaded from `strategies/` directory
- [ ] Strategy dropdown populated with available pairs
- [ ] Empty strategies directory handled gracefully
- [ ] First strategy selected by default (if available)

### AC3: Strategy Loading
- [ ] Selected strategy configuration loaded from YAML file
- [ ] Configuration displayed in read-only textbox
- [ ] Indexing pipeline instantiated successfully
- [ ] Retrieval pipeline instantiated successfully
- [ ] Pipeline references stored for later use
- [ ] Status bar updated with success message

### AC4: Migration Validation
- [ ] Migrations validated before allowing operations
- [ ] Missing migrations detected
- [ ] User-friendly error message displayed
- [ ] Upgrade command suggested (`alembic upgrade head`)
- [ ] Operations disabled when migrations missing

### AC5: Error Handling
- [ ] Missing services error displayed with service names
- [ ] Missing migrations error displayed with revision IDs
- [ ] Invalid YAML error displayed with details
- [ ] Database connection error displayed with connection details
- [ ] All errors displayed in status bar
- [ ] Critical errors also shown in popup dialogs

### AC6: Configuration Reload
- [ ] Reload Configs button rescans strategies directory
- [ ] Service registry reloaded on config reload
- [ ] Strategy dropdown updated with new strategies
- [ ] Current strategy reloaded if still available
- [ ] Status bar updated with reload result

---

## Technical Specifications

### Backend Initialization

```python
# rag_factory/gui/main_window.py (additions to Story 18.2)

from rag_factory.core.service_registry import ServiceRegistry
from rag_factory.strategies.strategy_pair_manager import StrategyPairManager
from rag_factory.core.exceptions import MigrationError, ServiceMissingError
import os
import yaml

class RAGFactoryGUI:
    def __init__(self):
        # ... existing initialization ...
        
        # Initialize backend
        try:
            self._initialize_backend()
            self._load_available_strategies()
        except Exception as e:
            self.show_error(f"Initialization failed: {e}")
            self.status_bar.config(text="🔴 Initialization failed")
    
    def _initialize_backend(self):
        """Initialize service registry and strategy pair manager."""
        # Load service registry
        try:
            config_path = "config/services.yaml"
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"{config_path} not found. Please create service configuration."
                )
            
            self.status_bar.config(text="⚫ Loading services...")
            self.service_registry = ServiceRegistry(config_path)
            
            # Verify required services
            required_services = ["db_main"]
            for service in required_services:
                if service not in self.service_registry.list_services():
                    raise ServiceMissingError(f"Missing required service: {service}")
                    
        except FileNotFoundError as e:
            raise Exception(str(e))
        except Exception as e:
            raise Exception(f"Service registry error: {e}")
        
        # Initialize strategy pair manager
        try:
            self.strategy_manager = StrategyPairManager(
                service_registry=self.service_registry,
                config_dir="strategies/",
                alembic_config="alembic.ini"
            )
            self.status_bar.config(text="⚫ Ready")
            
        except Exception as e:
            raise Exception(f"Strategy manager error: {e}")
    
    def _load_available_strategies(self):
        """Scan strategies/ directory and populate dropdown."""
        strategy_files = []
        
        # Find all .yaml files in strategies/
        strategies_dir = "strategies/"
        if not os.path.exists(strategies_dir):
            self.show_warning(
                f"Strategies directory not found: {strategies_dir}\n"
                "Please create the directory and add strategy configurations."
            )
            return
        
        for file in os.listdir(strategies_dir):
            if file.endswith(".yaml") and file != "README.yaml":
                strategy_name = file.replace(".yaml", "")
                strategy_files.append(strategy_name)
        
        if len(strategy_files) == 0:
            self.show_warning(
                "No strategy pairs found in strategies/ directory\n"
                "Please add at least one strategy configuration."
            )
            return
        
        # Sort alphabetically
        strategy_files.sort()
        
        # Populate dropdown
        self.strategy_dropdown.config(values=strategy_files)
        
        # Select first strategy by default
        self.strategy_dropdown.set(strategy_files[0])
        self.on_strategy_selected()  # Trigger load
    
    def on_strategy_selected(self, event=None):
        """Handler for strategy dropdown selection."""
        selected_strategy = self.strategy_dropdown.get()
        
        if selected_strategy == "":
            return
        
        self.status_bar.config(text=f"⚫ Loading strategy: {selected_strategy}...")
        self.root.update_idletasks()  # Force UI update
        
        try:
            # Load strategy configuration (YAML content)
            config_path = f"strategies/{selected_strategy}.yaml"
            with open(config_path, 'r') as f:
                config_yaml = f.read()
            
            # Display in config preview textbox
            self.config_textbox.config(state="normal")
            self.config_textbox.delete("1.0", "end")
            self.config_textbox.insert("1.0", config_yaml)
            self.config_textbox.config(state="disabled")
            
            # Instantiate strategy pair using StrategyPairManager
            self.indexing_pipeline, self.retrieval_pipeline = (
                self.strategy_manager.load_pair(selected_strategy)
            )
            
            self.current_strategy = selected_strategy
            self.status_bar.config(text=f"🟢 Strategy loaded: {selected_strategy}")
            
            # Enable operation buttons
            self.update_button_states()
            
        except MigrationError as e:
            # Missing migrations
            error_msg = (
                f"Missing migrations for {selected_strategy}:\n"
                f"{', '.join(e.missing_revisions)}\n\n"
                f"Please run: alembic upgrade head"
            )
            
            self.show_error(error_msg)
            self.status_bar.config(text=f"🔴 Migration error: {selected_strategy}")
            
            # Keep config visible but disable operations
            self.current_strategy = None
            self.update_button_states()
            
        except ServiceMissingError as e:
            # Missing services
            error_msg = (
                f"Missing services for {selected_strategy}:\n"
                f"{', '.join(e.missing_services)}\n\n"
                f"Please check config/services.yaml"
            )
            
            self.show_error(error_msg)
            self.status_bar.config(text=f"🔴 Service error: {selected_strategy}")
            
            self.current_strategy = None
            self.update_button_states()
            
        except yaml.YAMLError as e:
            # Invalid YAML
            error_msg = (
                f"Invalid YAML configuration for {selected_strategy}:\n"
                f"{str(e)}\n\n"
                f"Please check the strategy configuration file."
            )
            
            self.show_error(error_msg)
            self.status_bar.config(text=f"🔴 YAML error: {selected_strategy}")
            
            self.current_strategy = None
            self.update_button_states()
            
        except Exception as e:
            error_msg = f"Failed to load strategy: {str(e)}"
            self.show_error(error_msg)
            self.status_bar.config(text=f"🔴 Error loading: {selected_strategy}")
            
            self.current_strategy = None
            self.update_button_states()
    
    def on_reload_configs(self):
        """Reload strategy configurations from disk."""
        self.status_bar.config(text="⚫ Reloading configurations...")
        self.root.update_idletasks()
        
        try:
            # Reload service registry (in case services.yaml changed)
            self.service_registry.reload_all()
            
            # Rescan strategies directory
            self._load_available_strategies()
            
            self.status_bar.config(text="🟢 Configurations reloaded")
            
        except Exception as e:
            self.show_error(f"Reload failed: {str(e)}")
            self.status_bar.config(text="🔴 Reload failed")
    
    def show_error(self, message: str):
        """Display error in popup dialog."""
        messagebox.showerror("Error", message)
    
    def show_warning(self, message: str):
        """Display warning in popup dialog."""
        messagebox.showwarning("Warning", message)
    
    def show_info(self, message: str):
        """Display info in popup dialog."""
        messagebox.showinfo("Info", message)
```

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/gui/test_backend_integration.py
import pytest
from unittest.mock import Mock, MagicMock, patch
from rag_factory.gui.main_window import RAGFactoryGUI
from rag_factory.core.exceptions import MigrationError, ServiceMissingError

class TestBackendIntegration:
    """Unit tests for backend integration."""
    
    @patch('rag_factory.gui.main_window.ServiceRegistry')
    @patch('rag_factory.gui.main_window.StrategyPairManager')
    def test_backend_initialization_success(self, mock_manager, mock_registry):
        """Test successful backend initialization."""
        app = RAGFactoryGUI()
        
        assert app.service_registry is not None
        assert app.strategy_manager is not None
        
        app.root.destroy()
    
    @patch('rag_factory.gui.main_window.ServiceRegistry')
    def test_missing_service_config(self, mock_registry):
        """Test error handling for missing service config."""
        mock_registry.side_effect = FileNotFoundError("config/services.yaml not found")
        
        app = RAGFactoryGUI()
        
        # Should show error but not crash
        assert "Initialization failed" in app.status_bar.cget("text")
        
        app.root.destroy()
    
    def test_strategy_loading_success(self):
        """Test successful strategy loading."""
        app = RAGFactoryGUI()
        
        # Mock backend
        app.strategy_manager = Mock()
        app.strategy_manager.load_pair.return_value = (Mock(), Mock())
        
        # Simulate strategy selection
        app.strategy_dropdown.set("semantic-local-pair")
        app.on_strategy_selected()
        
        assert app.current_strategy == "semantic-local-pair"
        assert app.indexing_pipeline is not None
        assert app.retrieval_pipeline is not None
        
        app.root.destroy()
    
    def test_missing_migrations_error(self):
        """Test error handling for missing migrations."""
        app = RAGFactoryGUI()
        
        # Mock backend
        app.strategy_manager = Mock()
        app.strategy_manager.load_pair.side_effect = MigrationError(
            missing_revisions=["abc123", "def456"]
        )
        
        # Simulate strategy selection
        app.strategy_dropdown.set("test-strategy")
        app.on_strategy_selected()
        
        assert app.current_strategy is None
        assert "Migration error" in app.status_bar.cget("text")
        
        app.root.destroy()
```

### Integration Tests

```python
# tests/integration/gui/test_strategy_loading.py
import pytest
from rag_factory.gui.main_window import RAGFactoryGUI

@pytest.mark.integration
def test_load_real_strategy():
    """Test loading a real strategy configuration."""
    app = RAGFactoryGUI()
    
    # Assume semantic-local-pair exists
    app.strategy_dropdown.set("semantic-local-pair")
    app.on_strategy_selected()
    
    # Verify strategy loaded
    assert app.current_strategy == "semantic-local-pair"
    assert app.indexing_pipeline is not None
    assert app.retrieval_pipeline is not None
    
    # Verify config displayed
    config_text = app.config_textbox.get("1.0", "end")
    assert "semantic-local-pair" in config_text
    
    app.root.destroy()
```

---

## Story Points Breakdown

- **Service Registry Integration:** 2 points
- **StrategyPairManager Integration:** 2 points
- **Strategy Loading Implementation:** 2 points
- **Error Handling:** 1 point
- **Testing:** 1 point

**Total:** 8 points

---

## Dependencies

- Story 18.2 (Core GUI Framework) - MUST BE COMPLETED
- Epic 17 (Strategy Pair Configuration) - COMPLETED ✅
- `config/services.yaml` must exist
- At least one strategy configuration in `strategies/` directory

---

## Notes

- This story connects the GUI to the backend functionality
- All business logic is delegated to Epic 17 components
- Focus on error handling and user feedback
- No new business logic should be implemented in the GUI
- Keep the GUI as a thin wrapper around existing functionality
