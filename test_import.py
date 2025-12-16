"""Minimal test to isolate import issue"""

# Test 1: Can we import built-in modules?
print("Test 1: Importing sys...")
import sys
print("✅ sys imported")

# Test 2: Can we import from site-packages?
print("\nTest 2: Importing numpy...")
try:
    import numpy
    print("✅ numpy imported")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")

# Test 3: Can we import rag_factory.exceptions directly?
print("\nTest 3: Importing rag_factory.exceptions...")
try:
    import rag_factory.exceptions
    print("✅ rag_factory.exceptions imported")
except Exception as e:
    print(f"❌ rag_factory.exceptions import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ All tests completed")
