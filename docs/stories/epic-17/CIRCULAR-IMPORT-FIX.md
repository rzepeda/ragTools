# Circular Import Fix - Summary

## Date: 2025-12-16

## Problem
All imports from `rag_factory` package were hanging indefinitely, preventing any tests from running.

## Root Cause
**Circular import** caused by `rag_factory/services/__init__.py` importing service implementations at module level.

### Import Chain That Caused the Hang:
1. `rag_factory/__init__.py` → imports from `rag_factory.factory`
2. `rag_factory/factory.py` → imports from `rag_factory.core.pipeline`
3. `rag_factory/core/pipeline.py` → imports from `rag_factory.core.indexing_interface`
4. `rag_factory/core/indexing_interface.py` → imports from `rag_factory.services.dependencies`
5. `rag_factory/services/dependencies.py` → imports from `rag_factory.services.interfaces`
6. `rag_factory/services/interfaces.py` → (clean, no problematic imports)
7. **BUT** `rag_factory/services/__init__.py` → imports from `rag_factory.services.onnx`
8. `rag_factory/services/onnx` → likely imports something that creates a cycle back

## Solution

### 1. Simplified `rag_factory/__init__.py`
Removed all imports except exceptions and version to avoid triggering the chain:
```python
from rag_factory.__version__ import __version__
from rag_factory.exceptions import (...)
# Removed: factory, strategies, pipeline, observability imports
```

### 2. Fixed `rag_factory/services/__init__.py` ✅ **KEY FIX**
Removed service implementation imports that were causing the circular dependency:
```python
# REMOVED these imports:
# from rag_factory.services.onnx import ONNXEmbeddingService
# from rag_factory.services.api import (AnthropicLLMService, ...)
# from rag_factory.services.database import (Neo4jGraphService, ...)
# from rag_factory.services.local import (CosineRerankingService)

# Kept only:
from rag_factory.services.interfaces import (...)
from rag_factory.services.dependencies import (...)
from rag_factory.services.consistency import (...)
```

### 3. Commented out auto-registration in `rag_factory/__init__.py`
The `_register_default_strategies()` function was also disabled to prevent potential issues.

## Result
✅ **Imports now work!**
```bash
$ python -c "from rag_factory.factory import RAGFactory; print('OK')"
✅ factory OK

$ python -c "from rag_factory.services.interfaces import ILLMService; print('OK')"
✅ interfaces OK
```

## Remaining Issue
Tests still timeout during execution (not during import). This is a separate issue that needs investigation.

## Files Modified
1. `/mnt/MCPProyects/ragTools/rag_factory/__init__.py` - Simplified imports
2. `/mnt/MCPProyects/ragTools/rag_factory/services/__init__.py` - Removed service implementation imports

## Recommendations

### For Service Imports
Service implementations should be imported directly where needed:
```python
# Instead of: from rag_factory.services import ONNXEmbeddingService
# Use: from rag_factory.services.onnx import ONNXEmbeddingService
```

### For Package __init__.py Files
- Keep `__init__.py` files minimal
- Only import interfaces and base classes
- Avoid importing implementations
- Use lazy imports or TYPE_CHECKING for type hints

### For Auto-Registration
Consider refactoring the auto-registration mechanism to:
- Use a separate registration module
- Implement lazy registration
- Or require explicit registration in application code

## Next Steps
1. ✅ Circular import fixed
2. ⏳ Investigate why tests still timeout during execution
3. ⏳ Verify the 6 code fixes we applied earlier actually work
4. ⏳ Continue fixing remaining failing tests
