"""Configuration for logging system."""

from typing import Optional
from dataclasses import dataclass
from enum import Enum


class LogFormat(Enum):
    """Log output formats."""

    JSON = "json"
    TEXT = "text"


@dataclass
class LoggingConfig:
    """Configuration for the logging system.

    Attributes:
        log_level: Minimum log level to record
        log_format: Output format (JSON or text)
        log_file: Path to log file (None for console only)
        max_file_size_mb: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        max_query_length: Maximum query length in logs (truncated after)
        enable_pii_filtering: Whether to filter PII from logs
        console_output: Whether to output logs to console
    """

    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.JSON
    log_file: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5
    max_query_length: int = 500
    enable_pii_filtering: bool = True
    console_output: bool = True
