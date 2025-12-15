"""Schema version tracking for configuration validation."""

# Current schema versions
SERVICE_REGISTRY_VERSION = "1.0.0"
STRATEGY_PAIR_VERSION = "1.0.0"

# Schema evolution history
VERSION_HISTORY = {
    "service_registry": {
        "1.0.0": "Initial service registry schema with LLM, embedding, and database services"
    },
    "strategy_pair": {
        "1.0.0": "Initial strategy pair schema with indexer/retriever configuration"
    }
}


def is_compatible(schema_type: str, version: str) -> bool:
    """
    Check if a configuration version is compatible with current schema.
    
    Args:
        schema_type: Type of schema ("service_registry" or "strategy_pair")
        version: Version string to check
        
    Returns:
        True if version is compatible, False otherwise
    """
    if schema_type == "service_registry":
        current = SERVICE_REGISTRY_VERSION
    elif schema_type == "strategy_pair":
        current = STRATEGY_PAIR_VERSION
    else:
        return False
    
    # For now, only exact version match is supported
    # Future: implement semver compatibility checking
    return version == current
