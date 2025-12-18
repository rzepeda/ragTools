import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from rag_factory.config.strategy_pair_manager import StrategyPairManager, ConfigurationError, CompatibilityError
from rag_factory.services.dependencies import StrategyDependencies, ServiceDependency
from rag_factory.registry.service_registry import ServiceRegistry

# Mocks for strategies must simulate behavior of actual strategies
class MockIndexingStrategy:
    def __init__(self, config, dependencies):
        self.config = config
        self.deps = dependencies
        
    def produces(self):
        return {"VECTOR"}

class MockRetrievalStrategy:
    def __init__(self, config, dependencies):
        self.config = config
        self.deps = dependencies
        
    def requires(self):
        return {"VECTOR"}
        
    def requires_capabilities(self):
        return self.requires()

class MockIncompatibleRetrievalStrategy:
    def __init__(self, config, dependencies):
        self.config = config
        self.deps = dependencies
        
    def requires(self):
        return {"VECTOR", "GRAPH"}
    
    def requires_capabilities(self):
        return self.requires()

@pytest.fixture
def mock_registry():
    registry = MagicMock(spec=ServiceRegistry)
    # Mock some services
    registry.get.side_effect = lambda x: MagicMock(name=x)
    return registry

@pytest.fixture
def mock_loader():
    with patch("rag_factory.config.strategy_pair_manager.StrategyPairLoader") as MockLoader:
        loader_instance = MockLoader.return_value
        yield loader_instance

@pytest.fixture
def manager(mock_registry, mock_loader):
    # Mock MigrationValidator to avoid actual DB connection
    with patch("rag_factory.config.strategy_pair_manager.MigrationValidator") as MockMV:
        manager = StrategyPairManager(mock_registry, config_dir="/tmp/strategies")
        # Ensure migration validator instance is set (the mocks might confuse __init__ logic slightly)
        manager.migration_validator = MockMV.return_value
        manager.migration_validator.validate.return_value = (True, [])
        return manager

def test_load_pair_success(manager, mock_loader, mock_registry):
    # Setup Loader return
    mock_config = {
        "strategy_name": "test-pair",
        "indexer": {
            "strategy": "indexer.module.Class",
            "services": {"llm": "$gpt4", "db": "$db1"},
            "config": {"some": "param"}
        },
        "retriever": {
            "strategy": "retriever.module.Class",
            "services": {"embedding": "$embed1"},
            "config": {"top_k": 5}
        },
        "migrations": {
            "required_revisions": ["1234"]
        }
    }
    mock_loader.load_config.return_value = mock_config

    # Mock _import_strategy_class
    with patch.object(manager, "_import_strategy_class") as mock_import:
        mock_import.side_effect = lambda x: MockIndexingStrategy if "indexer" in x else MockRetrievalStrategy

        # Run
        idx, ret = manager.load_pair("test-pair")

        # Assertions
        assert isinstance(idx, MockIndexingStrategy)
        assert isinstance(ret, MockRetrievalStrategy)
        
        # Verify loader usage
        mock_loader.load_config.assert_called_with(Path("/tmp/strategies/test-pair.yaml"))
        
        # Verify Migration Validation
        manager.migration_validator.validate.assert_called_with(["1234"])
        
        # Verify Service Resolution ($ prefix is stripped by manager)
        mock_registry.get.assert_any_call("gpt4")
        mock_registry.get.assert_any_call("db1")
        mock_registry.get.assert_any_call("embed1")

def test_load_pair_compatibility_error(manager, mock_loader):
    mock_config = {
        "strategy_name": "bad-pair",
        "indexer": {
            "strategy": "indexer.module.Class",
            "services": {},
            "config": {}
        },
        "retriever": {
            "strategy": "retriever.module.Class",
            "services": {},
            "config": {}
        }
    }
    mock_loader.load_config.return_value = mock_config

    with patch.object(manager, "_import_strategy_class") as mock_import:
        # Return Incompatible strategy for retriever
        mock_import.side_effect = lambda x: MockIndexingStrategy if "indexer" in x else MockIncompatibleRetrievalStrategy
        
        with pytest.raises(CompatibilityError, match="Retrieval requires"):
            manager.load_pair("bad-pair")

def test_migration_error(manager, mock_loader):
    mock_config = {
        "strategy_name": "test-pair",
        "migrations": {"required_revisions": ["1234"]},
        "indexer": {}, "retriever": {} # Need valid structure or loader mock handles it
    }
    mock_loader.load_config.return_value = mock_config
    
    manager.migration_validator.validate.return_value = (False, ["1234"])
    
    with pytest.raises(ConfigurationError, match="Missing migrations"):
        manager.load_pair("test-pair")

def test_db_context_creation(manager, mock_loader, mock_registry):
    mock_config = {
        "strategy_name": "db-context-pair",
        "indexer": {
            "strategy": "indexer.module.Class",
            "services": {"db": "$db1"},
            "db_config": {
                "tables": {"logical": "physical"},
                "fields": {"f1": "col1"}
            }
        },
        "retriever": { "strategy": "retriever.module.Class" }
    }
    mock_loader.load_config.return_value = mock_config

    # Mock DB service with get_context
    mock_db = MagicMock()
    mock_context = MagicMock()
    mock_db.get_context.return_value = mock_context
    
    # Configure registry to return mock_db ($ prefix is stripped by manager)
    def get_service(ref):
        if ref == "db1": return mock_db
        return MagicMock()
    mock_registry.get.side_effect = get_service

    with patch.object(manager, "_import_strategy_class") as mock_import:
        mock_import.return_value = MagicMock() # Generic mock strategy

        manager.load_pair("db-context-pair")

        # Verify get_context called
        mock_db.get_context.assert_called_with(
            table_mapping={"logical": "physical"},
            field_mapping={"f1": "col1"}
        )
