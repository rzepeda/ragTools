"""Log filters for PII protection and sampling."""

import re
import random
from typing import Any, Dict, Optional


class PIIFilter:
    """Filter to remove personally identifiable information from logs.

    This filter removes or masks common PII patterns including:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - API keys and tokens
    """

    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )
    PHONE_PATTERN = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    CREDIT_CARD_PATTERN = re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")
    API_KEY_PATTERN = re.compile(r"\b[A-Za-z0-9_-]{32,}\b")

    @classmethod
    def filter_text(cls, text: str) -> str:
        """Filter PII from a text string.

        Args:
            text: Text to filter

        Returns:
            Filtered text with PII masked
        """
        if not text:
            return text

        # Replace patterns
        text = cls.EMAIL_PATTERN.sub("[EMAIL]", text)
        text = cls.PHONE_PATTERN.sub("[PHONE]", text)
        text = cls.SSN_PATTERN.sub("[SSN]", text)
        text = cls.CREDIT_CARD_PATTERN.sub("[CREDIT_CARD]", text)
        # Only mask very long strings that look like API keys
        if len(text) > 32:
            text = cls.API_KEY_PATTERN.sub("[API_KEY]", text)

        return text

    @classmethod
    def filter_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter PII from dictionary values.

        Args:
            data: Dictionary to filter

        Returns:
            Filtered dictionary
        """
        filtered = {}
        for key, value in data.items():
            if isinstance(value, str):
                filtered[key] = cls.filter_text(value)
            elif isinstance(value, dict):
                filtered[key] = cls.filter_dict(value)
            elif isinstance(value, list):
                filtered[key] = [
                    cls.filter_text(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                filtered[key] = value
        return filtered


class SamplingFilter:
    """Filter to sample logs based on configured rate.

    Useful for high-volume scenarios where logging every event
    would be too expensive.
    """

    def __init__(self, sample_rate: float = 1.0):
        """Initialize sampling filter.

        Args:
            sample_rate: Fraction of logs to keep (0.0 to 1.0)
                        1.0 = keep all, 0.1 = keep 10%
        """
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0.0 and 1.0")
        self.sample_rate = sample_rate

    def should_log(self) -> bool:
        """Determine if this log should be kept based on sample rate.

        Returns:
            True if log should be kept, False otherwise
        """
        if self.sample_rate >= 1.0:
            return True
        return random.random() < self.sample_rate
