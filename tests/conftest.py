import sys
import importlib.util
from unittest.mock import MagicMock
import pytest

# Mock numpy and other dependencies GLOBALLY before any test collection
try:
    import numpy
except ImportError:
    sys.modules["numpy"] = MagicMock()

# Mock the services package and submodules that cause issues
# sys.modules["rag_factory.services"] = MagicMock()
# sys.modules["rag_factory.services.onnx"] = MagicMock()
# sys.modules["rag_factory.services.onnx.embedding"] = MagicMock()
# sys.modules["rag_factory.services.embedding"] = MagicMock()
# sys.modules["rag_factory.services.embedding.providers"] = MagicMock()
# sys.modules["rag_factory.services.embedding.providers.onnx_local"] = MagicMock()
# sys.modules["rag_factory.services.embedding.service"] = MagicMock()
# sys.modules["rag_factory.services.api"] = MagicMock()
# sys.modules["rag_factory.services.database"] = MagicMock()
# sys.modules["rag_factory.services.local"] = MagicMock()

# Manually load the modules we actually need
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Mock rag_factory package to prevent init
# sys.modules["rag_factory"] = MagicMock()
# sys.modules["rag_factory.strategies"] = MagicMock()

# Load modules in dependency order to avoid cycles and init triggers
load_module("rag_factory.core.capabilities", "/mnt/MCPProyects/ragTools/rag_factory/core/capabilities.py")
load_module("rag_factory.services.interfaces", "/mnt/MCPProyects/ragTools/rag_factory/services/interfaces.py")
load_module("rag_factory.services.dependencies", "/mnt/MCPProyects/ragTools/rag_factory/services/dependencies.py")
load_module("rag_factory.services.consistency", "/mnt/MCPProyects/ragTools/rag_factory/services/consistency.py")
load_module("rag_factory.core.indexing_interface", "/mnt/MCPProyects/ragTools/rag_factory/core/indexing_interface.py")
load_module("rag_factory.core.retrieval_interface", "/mnt/MCPProyects/ragTools/rag_factory/core/retrieval_interface.py")
load_module("rag_factory.core.pipeline", "/mnt/MCPProyects/ragTools/rag_factory/core/pipeline.py")
load_module("rag_factory.strategies.base", "/mnt/MCPProyects/ragTools/rag_factory/strategies/base.py")
load_module("rag_factory.exceptions", "/mnt/MCPProyects/ragTools/rag_factory/exceptions.py")
load_module("rag_factory.factory", "/mnt/MCPProyects/ragTools/rag_factory/factory.py")
load_module("rag_factory.strategies.indexing", "/mnt/MCPProyects/ragTools/rag_factory/strategies/indexing/__init__.py")
load_module("rag_factory.strategies.indexing.context_aware", "/mnt/MCPProyects/ragTools/rag_factory/strategies/indexing/context_aware.py")

# Load CLI modules for CLI tests
load_module("rag_factory.cli", "/mnt/MCPProyects/ragTools/rag_factory/cli/__init__.py")
load_module("rag_factory.cli.formatters", "/mnt/MCPProyects/ragTools/rag_factory/cli/formatters/__init__.py")
load_module("rag_factory.cli.formatters.validation", "/mnt/MCPProyects/ragTools/rag_factory/cli/formatters/validation.py")
load_module("rag_factory.cli.utils", "/mnt/MCPProyects/ragTools/rag_factory/cli/utils/__init__.py")
load_module("rag_factory.cli.utils.validation", "/mnt/MCPProyects/ragTools/rag_factory/cli/utils/validation.py")
load_module("rag_factory.cli.commands", "/mnt/MCPProyects/ragTools/rag_factory/cli/commands/__init__.py")
load_module("rag_factory.cli.commands.validate_pipeline", "/mnt/MCPProyects/ragTools/rag_factory/cli/commands/validate_pipeline.py")
load_module("rag_factory.cli.main", "/mnt/MCPProyects/ragTools/rag_factory/cli/main.py")
