"""
Unit tests for CostTracker.

Tests cost calculation, tracking, and budget management functionality.
"""

import pytest
from unittest.mock import Mock

from rag_factory.strategies.contextual.cost_tracker import CostTracker
from rag_factory.strategies.contextual.config import ContextualRetrievalConfig


@pytest.fixture
def config():
    """Default configuration for testing."""
    return ContextualRetrievalConfig(
        cost_per_1k_input_tokens=0.0015,
        cost_per_1k_output_tokens=0.002,
        budget_alert_threshold=1.0
    )


@pytest.fixture
def cost_tracker(config):
    """Cost tracker instance for testing."""
    return CostTracker(config)


def test_calculate_cost(cost_tracker):
    """Test cost calculation."""
    cost = cost_tracker.calculate_cost(input_tokens=1000, output_tokens=500)
    
    expected = (1000 / 1000) * 0.0015 + (500 / 1000) * 0.002
    assert cost == pytest.approx(expected)


def test_record_chunk_cost(cost_tracker):
    """Test recording chunk cost."""
    cost_tracker.record_chunk_cost(
        chunk_id="chunk_1",
        input_tokens=500,
        output_tokens=100,
        cost=0.001
    )
    
    assert cost_tracker.total_input_tokens == 500
    assert cost_tracker.total_output_tokens == 100
    assert cost_tracker.total_cost == 0.001
    assert "chunk_1" in cost_tracker.chunk_costs


def test_cost_summary(cost_tracker):
    """Test cost summary generation."""
    # Record multiple chunks
    for i in range(10):
        cost_tracker.record_chunk_cost(
            chunk_id=f"chunk_{i}",
            input_tokens=100,
            output_tokens=50,
            cost=0.0005
        )
    
    summary = cost_tracker.get_summary()
    
    assert summary["total_chunks"] == 10
    assert summary["total_cost"] == pytest.approx(0.005)
    assert summary["avg_cost_per_chunk"] == pytest.approx(0.0005)
    assert summary["total_input_tokens"] == 1000
    assert summary["total_output_tokens"] == 500


def test_budget_alert(cost_tracker, caplog):
    """Test budget alert threshold."""
    import logging
    caplog.set_level(logging.WARNING)
    
    # Exceed budget threshold
    for i in range(100):
        cost_tracker.record_chunk_cost(
            chunk_id=f"chunk_{i}",
            input_tokens=1000,
            output_tokens=200,
            cost=0.02  # High cost
        )
    
    # Should trigger alert
    assert "Cost alert" in caplog.text


def test_reset(cost_tracker):
    """Test resetting cost tracker."""
    cost_tracker.record_chunk_cost("chunk_1", 100, 50, 0.001)
    
    cost_tracker.reset()
    
    assert cost_tracker.total_cost == 0.0
    assert len(cost_tracker.chunk_costs) == 0
    assert cost_tracker.total_input_tokens == 0
    assert cost_tracker.total_output_tokens == 0


def test_check_budget_limit(cost_tracker):
    """Test budget limit checking."""
    cost_tracker.record_chunk_cost("chunk_1", 100, 50, 0.5)
    
    assert cost_tracker.check_budget_limit(1.0) is True
    assert cost_tracker.check_budget_limit(0.3) is False


def test_multiple_chunk_costs(cost_tracker):
    """Test tracking multiple chunks."""
    chunks = [
        ("chunk_1", 100, 50, 0.001),
        ("chunk_2", 200, 75, 0.002),
        ("chunk_3", 150, 60, 0.0015),
    ]
    
    for chunk_id, input_tokens, output_tokens, cost in chunks:
        cost_tracker.record_chunk_cost(chunk_id, input_tokens, output_tokens, cost)
    
    summary = cost_tracker.get_summary()
    
    assert summary["total_chunks"] == 3
    assert summary["total_input_tokens"] == 450
    assert summary["total_output_tokens"] == 185
    assert summary["total_cost"] == pytest.approx(0.0045)


def test_cost_per_1k_chunks(cost_tracker):
    """Test cost per 1k chunks calculation."""
    for i in range(5):
        cost_tracker.record_chunk_cost(f"chunk_{i}", 100, 50, 0.002)
    
    summary = cost_tracker.get_summary()
    
    # avg_cost_per_chunk = 0.002, so cost_per_1k = 2.0
    assert summary["cost_per_1k_chunks"] == pytest.approx(2.0)


def test_empty_tracker_summary(cost_tracker):
    """Test summary with no chunks."""
    summary = cost_tracker.get_summary()
    
    assert summary["total_chunks"] == 0
    assert summary["total_cost"] == 0.0
    assert summary["avg_cost_per_chunk"] == 0.0
    assert summary["cost_per_1k_chunks"] == 0.0
