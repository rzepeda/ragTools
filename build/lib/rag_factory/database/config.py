"""Database configuration for RAG Factory.

This module provides configuration management for database connections,
including connection pooling parameters and SSL/TLS settings.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database connection configuration.

    All settings can be configured via environment variables with DB_ prefix.
    Example: DB_DATABASE_URL, DB_POOL_SIZE, etc.

    Attributes:
        database_url: PostgreSQL connection URL (format: postgresql://user:pass@host:port/dbname)
        pool_size: Number of persistent connections to maintain in the pool
        max_overflow: Maximum number of connections that can be created beyond pool_size
        pool_timeout: Maximum time in seconds to wait for a connection from the pool
        pool_recycle: Time in seconds after which connections are recycled
        echo: Enable SQLAlchemy query logging for debugging
        pool_pre_ping: Test connections for liveness before using them
        vector_dimensions: Dimensions for vector embeddings (default: 1536 for OpenAI)
    """

    database_url: str = Field(
        description="PostgreSQL connection URL",
        examples=["postgresql://user:pass@localhost:5432/rag_factory"]
    )

    pool_size: int = Field(
        default=10,
        description="Number of persistent connections in the pool",
        ge=1,
        le=100
    )

    max_overflow: int = Field(
        default=20,
        description="Maximum overflow connections beyond pool_size",
        ge=0,
        le=100
    )

    pool_timeout: int = Field(
        default=30,
        description="Timeout in seconds for acquiring a connection",
        ge=1,
        le=300
    )

    pool_recycle: int = Field(
        default=3600,
        description="Recycle connections after this many seconds",
        ge=60,
        le=86400
    )

    echo: bool = Field(
        default=False,
        description="Enable SQLAlchemy query logging"
    )

    pool_pre_ping: bool = Field(
        default=True,
        description="Test connections for liveness before using"
    )

    vector_dimensions: int = Field(
        default=1536,
        description="Vector embedding dimensions (1536 for OpenAI)",
        ge=1,
        le=4096
    )

    class Config:
        """Pydantic configuration."""
        env_prefix = "DB_"
        env_file = ".env"
        case_sensitive = False
        extra = "allow"
