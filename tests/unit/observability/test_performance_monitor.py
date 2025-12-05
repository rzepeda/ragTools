"""Unit tests for performance monitor."""

import pytest
import time

from rag_factory.observability.metrics.performance import (
    PerformanceMonitor,
    PerformanceSnapshot,
    get_performance_monitor,
)


@pytest.fixture
def monitor():
    """Create a PerformanceMonitor instance for testing."""
    return PerformanceMonitor()


@pytest.fixture
def reset_global_monitor():
    """Reset global monitor after each test."""
    yield
    import rag_factory.observability.metrics.performance as perf_module
    perf_module._global_monitor = None


class TestPerformanceSnapshot:
    """Tests for PerformanceSnapshot dataclass."""

    def test_performance_snapshot_creation(self):
        """Test creating a performance snapshot."""
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            cpu_percent=25.5,
            memory_percent=50.0,
            memory_used_mb=1024.0,
            memory_available_mb=2048.0,
        )

        assert snapshot.cpu_percent == 25.5
        assert snapshot.memory_percent == 50.0
        assert snapshot.memory_used_mb == 1024.0
        assert snapshot.memory_available_mb == 2048.0


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""

    def test_monitor_initialization(self, monitor):
        """Test monitor initializes correctly."""
        assert isinstance(monitor._snapshots, dict)
        assert isinstance(monitor._durations, dict)

    def test_track_operation(self, monitor):
        """Test tracking an operation."""
        with monitor.track("test_operation"):
            time.sleep(0.02)  # 20ms

        stats = monitor.get_stats("test_operation")

        assert stats["executions"] == 1
        assert stats["avg_duration_ms"] >= 20
        assert "avg_cpu_percent" in stats
        assert "avg_memory_percent" in stats

    def test_track_multiple_executions(self, monitor):
        """Test tracking multiple executions of same operation."""
        for _ in range(5):
            with monitor.track("test_operation"):
                time.sleep(0.01)

        stats = monitor.get_stats("test_operation")

        assert stats["executions"] == 5
        assert stats["avg_duration_ms"] >= 10

    def test_track_multiple_operations(self, monitor):
        """Test tracking different operations."""
        operations = ["op1", "op2", "op3"]

        for operation in operations:
            with monitor.track(operation):
                time.sleep(0.01)

        all_stats = monitor.get_all_stats()

        assert len(all_stats) == 3
        for operation in operations:
            assert operation in all_stats
            assert all_stats[operation]["executions"] == 1

    def test_get_stats_nonexistent(self, monitor):
        """Test getting stats for non-existent operation."""
        stats = monitor.get_stats("nonexistent")
        assert stats == {}

    def test_stats_calculation(self, monitor):
        """Test statistics calculation."""
        durations = [10, 20, 30, 40, 50]  # ms

        for duration_ms in durations:
            with monitor.track("test"):
                time.sleep(duration_ms / 1000)  # Convert to seconds

        stats = monitor.get_stats("test")

        assert stats["executions"] == 5
        assert stats["min_duration_ms"] >= 10
        assert stats["max_duration_ms"] >= 50
        assert "total_duration_ms" in stats

    def test_reset_specific_operation(self, monitor):
        """Test resetting specific operation."""
        with monitor.track("op1"):
            time.sleep(0.01)
        with monitor.track("op2"):
            time.sleep(0.01)

        monitor.reset("op1")

        stats_op1 = monitor.get_stats("op1")
        stats_op2 = monitor.get_stats("op2")

        assert stats_op1 == {}
        assert stats_op2["executions"] == 1

    def test_reset_all(self, monitor):
        """Test resetting all operations."""
        with monitor.track("op1"):
            time.sleep(0.01)
        with monitor.track("op2"):
            time.sleep(0.01)

        monitor.reset()

        all_stats = monitor.get_all_stats()
        assert len(all_stats) == 0

    def test_get_current_system_stats(self, monitor):
        """Test getting current system statistics."""
        stats = monitor.get_current_system_stats()

        assert "timestamp" in stats
        assert "cpu_percent" in stats
        assert "memory_percent" in stats
        assert "memory_used_mb" in stats
        assert "memory_available_mb" in stats
        assert "disk_percent" in stats
        assert "disk_used_gb" in stats
        assert "disk_free_gb" in stats

        # Verify values are reasonable
        assert 0 <= stats["cpu_percent"] <= 100
        assert 0 <= stats["memory_percent"] <= 100
        assert stats["memory_used_mb"] >= 0
        assert stats["memory_available_mb"] >= 0

    def test_track_with_exception(self, monitor):
        """Test tracking when exception occurs."""
        try:
            with monitor.track("error_operation"):
                time.sleep(0.01)
                raise ValueError("Test error")
        except ValueError:
            pass

        stats = monitor.get_stats("error_operation")

        # Should still record the operation
        assert stats["executions"] == 1
        assert stats["avg_duration_ms"] >= 10


class TestGetPerformanceMonitor:
    """Tests for get_performance_monitor function."""

    def test_get_monitor_singleton(self, reset_global_monitor):
        """Test that get_performance_monitor returns singleton instance."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()

        assert monitor1 is monitor2

    def test_get_monitor_persistence(self, reset_global_monitor):
        """Test that monitor persists data across calls."""
        monitor1 = get_performance_monitor()

        with monitor1.track("test"):
            time.sleep(0.01)

        monitor2 = get_performance_monitor()
        stats = monitor2.get_stats("test")

        assert stats["executions"] == 1
