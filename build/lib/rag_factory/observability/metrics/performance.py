"""Performance monitoring utilities."""

import time
import psutil
from typing import Dict, Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance metrics.

    Attributes:
        timestamp: When snapshot was taken
        cpu_percent: CPU utilization percentage
        memory_percent: Memory utilization percentage
        memory_used_mb: Memory used in megabytes
        memory_available_mb: Memory available in megabytes
    """

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float


class PerformanceMonitor:
    """Monitor system performance during operations.

    Tracks CPU, memory usage, and operation timing.

    Example:
        ```python
        monitor = PerformanceMonitor()

        with monitor.track("embedding_generation"):
            # Your operation
            embeddings = generate_embeddings(texts)

        stats = monitor.get_stats("embedding_generation")
        print(f"Duration: {stats['duration_ms']:.2f}ms")
        print(f"CPU: {stats['avg_cpu_percent']:.1f}%")
        ```
    """

    def __init__(self):
        """Initialize performance monitor."""
        self._snapshots: Dict[str, list] = {}
        self._durations: Dict[str, list] = {}

    @contextmanager
    def track(self, operation_name: str, interval: float = 0.1):
        """Track performance for an operation.

        Args:
            operation_name: Name of the operation to track
            interval: Sampling interval in seconds

        Yields:
            None
        """
        if operation_name not in self._snapshots:
            self._snapshots[operation_name] = []
        if operation_name not in self._durations:
            self._durations[operation_name] = []

        start_time = time.time()
        snapshots = []

        try:
            # Take initial snapshot
            snapshots.append(self._take_snapshot())

            yield

            # Take final snapshot
            snapshots.append(self._take_snapshot())

        finally:
            duration = (time.time() - start_time) * 1000  # Convert to ms
            self._durations[operation_name].append(duration)
            self._snapshots[operation_name].extend(snapshots)

    def _take_snapshot(self) -> PerformanceSnapshot:
        """Take a snapshot of current system performance.

        Returns:
            PerformanceSnapshot with current metrics
        """
        memory = psutil.virtual_memory()

        return PerformanceSnapshot(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
        )

    def get_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get performance statistics for an operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Dictionary of performance statistics
        """
        if operation_name not in self._snapshots:
            return {}

        snapshots = self._snapshots[operation_name]
        durations = self._durations[operation_name]

        if not snapshots:
            return {}

        avg_cpu = sum(s.cpu_percent for s in snapshots) / len(snapshots)
        avg_memory = sum(s.memory_percent for s in snapshots) / len(snapshots)
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "operation": operation_name,
            "executions": len(durations),
            "avg_duration_ms": round(avg_duration, 2),
            "total_duration_ms": round(sum(durations), 2),
            "min_duration_ms": round(min(durations), 2) if durations else 0,
            "max_duration_ms": round(max(durations), 2) if durations else 0,
            "avg_cpu_percent": round(avg_cpu, 2),
            "avg_memory_percent": round(avg_memory, 2),
            "snapshots_count": len(snapshots),
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all tracked operations.

        Returns:
            Dictionary mapping operation names to their statistics
        """
        return {
            operation: self.get_stats(operation)
            for operation in self._snapshots.keys()
        }

    def reset(self, operation_name: Optional[str] = None):
        """Reset performance data.

        Args:
            operation_name: Specific operation to reset, or None for all
        """
        if operation_name:
            if operation_name in self._snapshots:
                del self._snapshots[operation_name]
            if operation_name in self._durations:
                del self._durations[operation_name]
        else:
            self._snapshots.clear()
            self._durations.clear()

    def get_current_system_stats(self) -> Dict[str, Any]:
        """Get current system performance statistics.

        Returns:
            Dictionary of current system metrics
        """
        snapshot = self._take_snapshot()
        disk = psutil.disk_usage("/")

        return {
            "timestamp": snapshot.timestamp,
            "cpu_percent": snapshot.cpu_percent,
            "memory_percent": snapshot.memory_percent,
            "memory_used_mb": round(snapshot.memory_used_mb, 2),
            "memory_available_mb": round(snapshot.memory_available_mb, 2),
            "disk_percent": disk.percent,
            "disk_used_gb": round(disk.used / (1024**3), 2),
            "disk_free_gb": round(disk.free / (1024**3), 2),
        }


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor instance.

    Returns:
        PerformanceMonitor instance
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor
