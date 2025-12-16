"""Core logger implementation for RAG Factory."""

import structlog
import logging
import time
import re
from typing import Dict, Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class LogLevel(Enum):
    """Log levels for structured logging."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogContext:
    """Context for a logging session with timing and metadata tracking.

    Attributes:
        strategy_name: Name of the RAG strategy being executed
        operation: Type of operation (e.g., 'retrieve', 'rerank', 'expand')
        query: Optional query string
        start_time: Timestamp when operation started
        metadata: Additional metadata to include in logs
    """

    strategy_name: str
    operation: str
    query: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds since operation start.

        Returns:
            Elapsed time in milliseconds
        """
        return (time.time() - self.start_time) * 1000


class RAGLogger:
    """Centralized structured logger for RAG operations.

    Features:
    - Structured logging with JSON output
    - Context management for operations with automatic timing
    - Performance tracking
    - Error tracking with full stack traces
    - PII filtering to prevent sensitive data leakage
    - File and console logging with rotation
    - Async-friendly design

    Example:
        ```python
        logger = RAGLogger()

        # Use context manager for automatic timing
        with logger.operation("retrieve", strategy="vector_search", query="test") as ctx:
            # Your retrieval operation
            results = perform_retrieval()
            ctx.metadata["results_count"] = len(results)
        # Automatically logs duration and results

        # Manual logging
        logger.log_query(
            query="test query",
            strategy="vector_search",
            results_count=5,
            latency_ms=45.2,
            success=True
        )
        ```
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        log_file: Optional[str] = None,
        log_level: LogLevel = LogLevel.INFO,
    ):
        """Initialize logger with configuration.

        Args:
            config: Optional configuration dictionary
            log_file: Optional path to log file for file-based logging
            log_level: Minimum log level to record
        """
        self.config = config or {}
        self.log_file = log_file
        self.log_level = log_level
        self._setup_structlog()
        self.log = structlog.get_logger()

    def _setup_structlog(self):
        """Configure structlog with processors and formatters."""
        # Set up standard Python logging
        logging.basicConfig(
            format="%(message)s",
            level=getattr(logging, self.log_level.value),
            handlers=self._get_handlers(),
        )

        # Configure structlog
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    def _get_handlers(self) -> list:
        """Get logging handlers based on configuration.

        Returns:
            List of logging handlers
        """
        handlers = []

        # Console handler (always enabled)
        console_handler = logging.StreamHandler()
        handlers.append(console_handler)

        # File handler (if log_file specified)
        if self.log_file:
            from logging.handlers import RotatingFileHandler

            # Create log directory if it doesn't exist
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Rotating file handler with 10MB max size, 5 backups
            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
            )
            handlers.append(file_handler)

        return handlers

    @contextmanager
    def operation(
        self,
        operation: str,
        strategy: str,
        query: Optional[str] = None,
        **metadata,
    ):
        """Context manager for logging an operation with automatic timing.

        Args:
            operation: Type of operation (e.g., "retrieve", "rerank", "expand")
            strategy: Strategy name
            query: Optional query string
            **metadata: Additional metadata to include in logs

        Yields:
            LogContext: Context object with timing and metadata

        Example:
            ```python
            with logger.operation("retrieve", strategy="vector_search", query="test") as ctx:
                results = perform_retrieval()
                ctx.metadata["results_count"] = len(results)
            ```
        """
        context = LogContext(
            strategy_name=strategy,
            operation=operation,
            query=query,
            metadata=metadata,
        )

        # Log operation start
        self.log.info(
            "operation_start",
            operation=operation,
            strategy=strategy,
            query=self._sanitize_query(query),
            **metadata,
        )

        try:
            yield context

            # Log operation success
            self.log.info(
                "operation_complete",
                operation=operation,
                strategy=strategy,
                duration_ms=context.elapsed_ms(),
                **context.metadata,
            )

        except Exception as e:
            # Log operation failure
            self.log.error(
                "operation_failed",
                operation=operation,
                strategy=strategy,
                duration_ms=context.elapsed_ms(),
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise

    def log_query(
        self,
        query: str,
        strategy: str,
        results_count: int,
        latency_ms: float,
        success: bool = True,
        **metadata,
    ):
        """Log a query execution with results.

        Args:
            query: The query string
            strategy: Strategy name
            results_count: Number of results returned
            latency_ms: Query latency in milliseconds
            success: Whether query succeeded
            **metadata: Additional metadata
        """
        self.log.info(
            "query_executed",
            query=self._sanitize_query(query),
            strategy=strategy,
            results_count=results_count,
            latency_ms=latency_ms,
            success=success,
            **metadata,
        )

    def log_metrics(
        self, strategy: str, operation: str, metrics: Dict[str, Any]
    ):
        """Log performance metrics for an operation.

        Args:
            strategy: Strategy name
            operation: Operation type
            metrics: Dictionary of metrics (latency, tokens, cost, etc.)
        """
        self.log.info(
            "metrics",
            strategy=strategy,
            operation=operation,
            **metrics,
        )

    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: LogLevel = LogLevel.ERROR,
    ):
        """Log an error with full context and stack trace.

        Args:
            error: The exception that occurred
            context: Additional context about the error
            severity: Log level for the error
        """
        log_func = getattr(self.log, severity.value.lower())
        log_func(
            "error_occurred",
            error=str(error),
            error_type=type(error).__name__,
            exc_info=True,
            **context,
        )

    def debug(self, message: str, **kwargs):
        """Log a debug message.

        Args:
            message: Debug message
            **kwargs: Additional context
        """
        self.log.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log an info message.

        Args:
            message: Info message
            **kwargs: Additional context
        """
        self.log.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log a warning message.

        Args:
            message: Warning message
            **kwargs: Additional context
        """
        self.log.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log an error message.

        Args:
            message: Error message
            **kwargs: Additional context
        """
        self.log.error(message, **kwargs)

    def _sanitize_query(self, query: Optional[str]) -> Optional[str]:
        """Sanitize query to remove PII and limit length.

        Args:
            query: Query string to sanitize

        Returns:
            Sanitized query string or None
        """
        if not query:
            return None

        # Truncate long queries
        max_length = self.config.get("max_query_length", 500)
        if len(query) > max_length:
            query = query[:max_length] + "..."

        # Remove potential email addresses
        query = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "[EMAIL]",
            query,
        )

        # Remove potential phone numbers (basic pattern)
        query = re.sub(
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "[PHONE]",
            query,
        )

        # Remove potential SSNs
        query = re.sub(
            r"\b\d{3}-\d{2}-\d{4}\b",
            "[SSN]",
            query,
        )

        return query


# Global logger instance for convenience
_global_logger: Optional[RAGLogger] = None


def get_logger(
    config: Optional[Dict[str, Any]] = None,
    log_file: Optional[str] = None,
) -> RAGLogger:
    """Get or create the global logger instance.

    Args:
        config: Optional configuration for logger
        log_file: Optional path to log file

    Returns:
        RAGLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = RAGLogger(config=config, log_file=log_file)
    return _global_logger
