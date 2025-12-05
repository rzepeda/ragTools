"""Unit tests for log filters."""

import pytest

from rag_factory.observability.logging.filters import PIIFilter, SamplingFilter


class TestPIIFilter:
    """Tests for PIIFilter class."""

    def test_filter_email(self):
        """Test email filtering."""
        text = "Contact john.doe@example.com for details"
        filtered = PIIFilter.filter_text(text)

        assert "@example.com" not in filtered
        assert "[EMAIL]" in filtered

    def test_filter_multiple_emails(self):
        """Test filtering multiple emails."""
        text = "Email john@example.com or jane@test.org"
        filtered = PIIFilter.filter_text(text)

        assert "john@example.com" not in filtered
        assert "jane@test.org" not in filtered
        assert filtered.count("[EMAIL]") == 2

    def test_filter_phone(self):
        """Test phone number filtering."""
        text = "Call 555-123-4567 or 555.987.6543"
        filtered = PIIFilter.filter_text(text)

        assert "555-123-4567" not in filtered
        assert "555.987.6543" not in filtered
        assert "[PHONE]" in filtered

    def test_filter_ssn(self):
        """Test SSN filtering."""
        text = "SSN: 123-45-6789"
        filtered = PIIFilter.filter_text(text)

        assert "123-45-6789" not in filtered
        assert "[SSN]" in filtered

    def test_filter_credit_card(self):
        """Test credit card filtering."""
        text = "Card: 1234-5678-9012-3456"
        filtered = PIIFilter.filter_text(text)

        assert "1234-5678-9012-3456" not in filtered
        assert "[CREDIT_CARD]" in filtered

    def test_filter_credit_card_no_dashes(self):
        """Test credit card filtering without dashes."""
        text = "Card: 1234567890123456"
        filtered = PIIFilter.filter_text(text)

        assert "1234567890123456" not in filtered
        assert "[CREDIT_CARD]" in filtered

    def test_filter_none_text(self):
        """Test filtering None text."""
        filtered = PIIFilter.filter_text(None)
        assert filtered is None

    def test_filter_empty_text(self):
        """Test filtering empty text."""
        filtered = PIIFilter.filter_text("")
        assert filtered == ""

    def test_filter_dict(self):
        """Test filtering dictionary values."""
        data = {
            "email": "john@example.com",
            "phone": "555-123-4567",
            "message": "Contact me",
        }

        filtered = PIIFilter.filter_dict(data)

        assert "[EMAIL]" in filtered["email"]
        assert "[PHONE]" in filtered["phone"]
        assert filtered["message"] == "Contact me"

    def test_filter_nested_dict(self):
        """Test filtering nested dictionary."""
        data = {
            "user": {
                "email": "john@example.com",
                "phone": "555-123-4567",
            },
            "message": "Hello",
        }

        filtered = PIIFilter.filter_dict(data)

        assert "[EMAIL]" in filtered["user"]["email"]
        assert "[PHONE]" in filtered["user"]["phone"]
        assert filtered["message"] == "Hello"

    def test_filter_dict_with_list(self):
        """Test filtering dictionary with list values."""
        data = {
            "emails": ["john@example.com", "jane@test.org"],
            "message": "Contact list",
        }

        filtered = PIIFilter.filter_dict(data)

        assert "[EMAIL]" in filtered["emails"][0]
        assert "[EMAIL]" in filtered["emails"][1]

    def test_filter_preserves_non_pii(self):
        """Test that non-PII data is preserved."""
        text = "The weather is nice today at 3:45 PM"
        filtered = PIIFilter.filter_text(text)

        assert filtered == text

    def test_filter_mixed_content(self):
        """Test filtering text with mixed content."""
        text = "Call 555-123-4567 or email john@example.com. SSN: 123-45-6789"
        filtered = PIIFilter.filter_text(text)

        assert "[PHONE]" in filtered
        assert "[EMAIL]" in filtered
        assert "[SSN]" in filtered
        assert "555-123-4567" not in filtered
        assert "john@example.com" not in filtered
        assert "123-45-6789" not in filtered


class TestSamplingFilter:
    """Tests for SamplingFilter class."""

    def test_sampling_filter_initialization(self):
        """Test initializing sampling filter."""
        filter = SamplingFilter(sample_rate=0.5)
        assert filter.sample_rate == 0.5

    def test_sampling_filter_invalid_rate_too_high(self):
        """Test invalid sample rate (too high)."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            SamplingFilter(sample_rate=1.5)

    def test_sampling_filter_invalid_rate_negative(self):
        """Test invalid sample rate (negative)."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            SamplingFilter(sample_rate=-0.1)

    def test_sampling_filter_100_percent(self):
        """Test 100% sampling rate."""
        filter = SamplingFilter(sample_rate=1.0)

        # Should always return True
        for _ in range(100):
            assert filter.should_log() is True

    def test_sampling_filter_0_percent(self):
        """Test 0% sampling rate."""
        filter = SamplingFilter(sample_rate=0.0)

        # Should always return False
        for _ in range(100):
            assert filter.should_log() is False

    def test_sampling_filter_50_percent(self):
        """Test 50% sampling rate (probabilistic)."""
        filter = SamplingFilter(sample_rate=0.5)

        # Run many iterations and check approximate rate
        iterations = 10000
        logged_count = sum(1 for _ in range(iterations) if filter.should_log())

        # Should be around 50% (allow 5% variance)
        rate = logged_count / iterations
        assert 0.45 <= rate <= 0.55

    def test_sampling_filter_10_percent(self):
        """Test 10% sampling rate (probabilistic)."""
        filter = SamplingFilter(sample_rate=0.1)

        # Run many iterations and check approximate rate
        iterations = 10000
        logged_count = sum(1 for _ in range(iterations) if filter.should_log())

        # Should be around 10% (allow 3% variance)
        rate = logged_count / iterations
        assert 0.07 <= rate <= 0.13
