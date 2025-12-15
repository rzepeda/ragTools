
try:
    from rag_factory.services.embedding.providers.onnx_local import ONNXLocalProvider
    print("Import successful")
    config = {"model": "Xenova/all-MiniLM-L6-v2"}
    try:
        provider = ONNXLocalProvider(config)
        print("Provider instantiated")
    except Exception as e:
        print(f"Instantiation failed: {e}")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Other error: {e}")
