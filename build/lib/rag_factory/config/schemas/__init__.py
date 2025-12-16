"""Configuration schemas package."""

from .version import (
    SERVICE_REGISTRY_VERSION,
    STRATEGY_PAIR_VERSION,
    VERSION_HISTORY,
    is_compatible,
)

__all__ = [
    "SERVICE_REGISTRY_VERSION",
    "STRATEGY_PAIR_VERSION",
    "VERSION_HISTORY",
    "is_compatible",
]
