"""Unit tests for rate limiter."""

import pytest
import time
import threading
from rag_factory.services.embedding.rate_limiter import RateLimiter


def test_rate_limiter_initialization():
    """Test rate limiter initializes correctly."""
    limiter = RateLimiter({"requests_per_minute": 60})
    assert limiter.min_interval == 1.0


def test_rate_limiter_allows_first_request():
    """Test first request is not delayed."""
    limiter = RateLimiter({"requests_per_second": 10})

    start = time.time()
    limiter.wait_if_needed()
    duration = time.time() - start

    assert duration < 0.01  # Should be immediate


def test_rate_limiter_enforces_limit():
    """Test rate limiter enforces rate limit."""
    limiter = RateLimiter({"requests_per_second": 2})

    start = time.time()
    limiter.wait_if_needed()
    limiter.wait_if_needed()
    duration = time.time() - start

    # Second request should wait ~0.5 seconds
    assert duration >= 0.4


def test_rate_limiter_requests_per_minute():
    """Test rate limiter with requests per minute."""
    limiter = RateLimiter({"requests_per_minute": 120})

    # Should allow 2 requests per second
    start = time.time()
    limiter.wait_if_needed()
    limiter.wait_if_needed()
    duration = time.time() - start

    assert duration >= 0.4


def test_rate_limiter_multiple_requests():
    """Test multiple requests respect rate limit."""
    limiter = RateLimiter({"requests_per_second": 5})

    start = time.time()
    for _ in range(5):
        limiter.wait_if_needed()
    duration = time.time() - start

    # 5 requests at 5 req/sec should take ~0.8 seconds
    assert duration >= 0.7


def test_rate_limiter_thread_safety():
    """Test rate limiter is thread-safe."""
    limiter = RateLimiter({"requests_per_second": 10})
    results = []

    def worker():
        start = time.time()
        limiter.wait_if_needed()
        results.append(time.time() - start)

    threads = [threading.Thread(target=worker) for _ in range(5)]

    start = time.time()
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    total_duration = time.time() - start

    # 5 requests at 10 req/sec should take ~0.4 seconds
    assert total_duration >= 0.3


def test_rate_limiter_prefers_stricter_limit():
    """Test that stricter limit is used when both are specified."""
    # requests_per_second=1 is stricter than requests_per_minute=120
    limiter = RateLimiter({
        "requests_per_second": 1,
        "requests_per_minute": 120
    })

    assert limiter.min_interval == 1.0  # Should use per-second limit
