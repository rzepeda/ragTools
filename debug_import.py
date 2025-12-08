import sys
import os

print(f"CWD: {os.getcwd()}")
print(f"sys.path: {sys.path}")

try:
    import rag_factory
    print(f"rag_factory: {rag_factory}")
    print(f"rag_factory file: {rag_factory.__file__}")
except ImportError as e:
    print(f"Error importing rag_factory: {e}")

try:
    import rag_factory.strategies
    print(f"rag_factory.strategies: {rag_factory.strategies}")
    print(f"rag_factory.strategies file: {rag_factory.strategies.__file__}")
except ImportError as e:
    print(f"Error importing rag_factory.strategies: {e}")

try:
    import rag_factory.strategies.indexing
    print(f"rag_factory.strategies.indexing: {rag_factory.strategies.indexing}")
except ImportError as e:
    print(f"Error importing rag_factory.strategies.indexing: {e}")
