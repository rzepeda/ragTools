"""Unit tests for RAG logger."""

import pytest
import time
from unittest.mock import Mock, patch
import structlog

from rag_factory.observability.logging.logger import (
    RAGLogger,
    LogContext,
    LogLevel,
    get_logger,
)


@pytest.fixture
def logger():
    """Create a RAGLogger instance for testing."""
    return RAGLogger()


@pytest.fixture
def reset_global_logger():
    """Reset global logger after each test."""
    yield
    import rag_factory.observability.logging.logger as logger_module
    logger_module._global_logger = None


class TestLogContext:
    """Tests for LogContext dataclass."""

    def test_log_context_creation(self):
        """Test creating a log context."""
        ctx = LogContext(
            strategy_name="test_strategy",
            operation="retrieve",
            query="test query",
        )

        assert ctx.strategy_name == "test_strategy"
        assert ctx.operation == "retrieve"
        assert ctx.query == "test query"
        assert ctx.start_time > 0
        assert isinstance(ctx.metadata, dict)

    def test_elapsed_ms(self):
        """Test elapsed time calculation."""
        ctx = LogContext(
            strategy_name="test_strategy",
            operation="retrieve",
        )

        time.sleep(0.01)  # Sleep 10ms
        elapsed = ctx.elapsed_ms()

        assert elapsed >= 10
        assert elapsed < 100  # Should be less than 100ms


class TestRAGLogger:
    """Tests for RAGLogger class."""

    def test_logger_initialization(self, logger):
        """Test logger initializes correctly."""
        assert logger.log is not None
        assert logger.config is not None
        assert logger.log_level == LogLevel.INFO

    def test_logger_with_config(self):
        """Test logger initialization with custom config."""
        config = {"max_query_length": 100}
        logger = RAGLogger(config=config)

        assert logger.config == config
        assert logger.config["max_query_length"] == 100

    def test_logger_with_log_file(self, tmp_path):
        """Test logger with file output."""
        log_file = tmp_path / "test.log"
        logger = RAGLogger(log_file=str(log_file))

        assert logger.log_file == str(log_file)

    def test_operation_context_success(self, logger):
        """Test successful operation logging."""
        with logger.operation(
            "retrieve", strategy="vector_search", query="test query"
        ) as ctx:
            assert ctx.strategy_name == "vector_search"
            assert ctx.operation == "retrieve"
            assert ctx.query == "test query"

            # Simulate work
            time.sleep(0.01)

        # Check elapsed time
        assert ctx.elapsed_ms() >= 10

    def test_operation_context_failure(self, logger):
        """Test operation logging with exception."""
        with pytest.raises(ValueError):
            with logger.operation(
                "retrieve", strategy="vector_search"
            ) as ctx:
                raise ValueError("Test error")

    def test_operation_context_metadata(self, logger):
        """Test adding metadata to operation context."""
        with logger.operation(
            "retrieve",
            strategy="test",
            custom_field="value",
            num_field=123,
        ) as ctx:
            ctx.metadata["result_count"] = 5
            ctx.metadata["tokens"] = 150

        assert ctx.metadata["result_count"] == 5
        assert ctx.metadata["tokens"] == 150

    def test_log_query(self, logger):
        """Test query logging."""
        # This should not raise any exceptions
        logger.log_query(
            query="test query",
            strategy="vector_search",
            results_count=5,
            latency_ms=45.2,
            success=True,
        )

    def test_log_metrics(self, logger):
        """Test metrics logging."""
        metrics = {
            "latency_ms": 45.2,
            "tokens": 150,
            "cost": 0.001,
        }

        logger.log_metrics(
            strategy="vector_search",
            operation="retrieve",
            metrics=metrics,
        )

    def test_log_error(self, logger):
        """Test error logging."""
        error = ValueError("Test error")
        context = {"strategy": "vector_search", "query": "test"}

        logger.log_error(error, context, LogLevel.ERROR)

    def test_convenience_methods(self, logger):
        """Test convenience logging methods."""
        logger.debug("Debug message", extra_field="value")
        logger.info("Info message", extra_field="value")
        logger.warning("Warning message", extra_field="value")
        logger.error("Error message", extra_field="value")

    def test_query_sanitization_truncation(self, logger):
        """Test query truncation for long queries."""
        long_query = "x" * 1000
        sanitized = logger._sanitize_query(long_query)

        assert len(sanitized) <= 503  # 500 + "..."
        assert sanitized.endswith("...")

    def test_query_sanitization_email(self, logger):
        """Test email removal from queries."""
        query = "Contact john.doe@example.com for details"
        sanitized = logger._sanitize_query(query)

        assert "@example.com" not in sanitized
        assert "[EMAIL]" in sanitized

    def test_query_sanitization_phone(self, logger):
        """Test phone number removal from queries."""
        query = "Call 555-123-4567 for support"
        sanitized = logger._sanitize_query(query)

        assert "555-123-4567" not in sanitized
        assert "[PHONE]" in sanitized

    def test_query_sanitization_ssn(self, logger):
        """Test SSN removal from queries."""
        query = "SSN: 123-45-6789"
        sanitized = logger._sanitize_query(query)

        assert "123-45-6789" not in sanitized
        assert "[SSN]" in sanitized

    def test_query_sanitization_none(self, logger):
        """Test sanitization with None query."""
        sanitized = logger._sanitize_query(None)
        assert sanitized is None


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_singleton(self, reset_global_logger):
        """Test that get_logger returns singleton instance."""
        logger1 = get_logger()
        logger2 = get_logger()

        assert logger1 is logger2

    def test_get_logger_with_config(self, reset_global_logger):
        """Test get_logger with configuration."""
        config = {"max_query_length": 100}
        logger = get_logger(config=config)

        assert logger.config == config


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_level_values(self):
        """Test log level enum values."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"
