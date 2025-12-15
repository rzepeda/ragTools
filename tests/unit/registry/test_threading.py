"""Thread safety tests for ServiceRegistry."""

import pytest
import threading
import time
from unittest.mock import Mock, patch

from rag_factory.registry.service_registry import ServiceRegistry


@pytest.fixture
def mock_services_file(tmp_path):
    """Create temporary services.yaml file."""
    content = """
services:
  llm1:
    name: "test-llm"
    type: "llm"
    url: "http://localhost:1234/v1"
    model: "test-model"

  embedding1:
    name: "test-embedding"
    type: "embedding"
    provider: "onnx"
    model: "Xenova/all-MiniLM-L6-v2"

  db1:
    name: "test-db"
    type: "postgres"
    connection_string: "postgresql://user:pass@localhost:5432/test"
"""
    services_file = tmp_path / "services.yaml"
    services_file.write_text(content)
    return str(services_file)


class TestConcurrentServiceAccess:
    """Tests for concurrent access to the same service."""

    def test_concurrent_service_access_creates_single_instance(self, mock_services_file):
        """Test concurrent access to same service creates only one instance."""
        registry = ServiceRegistry(mock_services_file)

        # Track number of times create_service is called
        call_count = 0
        call_lock = threading.Lock()

        def slow_create_service(service_name, config):
            """Simulate slow service creation."""
            nonlocal call_count
            time.sleep(0.1)  # Simulate slow creation
            with call_lock:
                call_count += 1
            return Mock()

        with patch.object(registry._factory, 'create_service', side_effect=slow_create_service):
            # Launch multiple threads accessing same service
            threads = []
            results = []

            def get_service():
                service = registry.get("llm1")
                results.append(service)

            for _ in range(10):
                thread = threading.Thread(target=get_service)
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Should only create service once despite concurrent access
            assert call_count == 1

            # All threads should get same instance
            assert len(set(id(s) for s in results)) == 1

    def test_concurrent_service_access_returns_same_instance(self, mock_services_file):
        """Test all concurrent requests get the same instance."""
        registry = ServiceRegistry(mock_services_file)

        mock_service = Mock()

        with patch.object(registry._factory, 'create_service', return_value=mock_service):
            results = []
            threads = []

            def get_service():
                service = registry.get("llm1")
                results.append(id(service))

            # Launch 20 concurrent threads
            for _ in range(20):
                thread = threading.Thread(target=get_service)
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            # All should have same instance ID
            assert len(set(results)) == 1
            assert all(result_id == id(mock_service) for result_id in results)

    def test_concurrent_access_with_slow_instantiation(self, mock_services_file):
        """Test concurrent access with very slow service instantiation."""
        registry = ServiceRegistry(mock_services_file)

        instantiation_count = 0
        lock = threading.Lock()

        def very_slow_create(service_name, config):
            """Very slow service creation."""
            nonlocal instantiation_count
            time.sleep(0.2)  # Longer delay
            with lock:
                instantiation_count += 1
            return Mock()

        with patch.object(registry._factory, 'create_service', side_effect=very_slow_create):
            threads = []
            results = []

            def get_service():
                service = registry.get("llm1")
                results.append(service)

            # Launch 15 threads
            for _ in range(15):
                thread = threading.Thread(target=get_service)
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            # Should only instantiate once
            assert instantiation_count == 1
            # All should get same instance
            assert len(set(id(s) for s in results)) == 1


class TestConcurrentDifferentServices:
    """Tests for concurrent access to different services."""

    def test_concurrent_different_services(self, mock_services_file):
        """Test concurrent access to different services works correctly."""
        registry = ServiceRegistry(mock_services_file)

        with patch.object(registry._factory, 'create_service', return_value=Mock()):
            results = {
                'llm1': [],
                'embedding1': [],
                'db1': []
            }

            def get_service(service_name):
                service = registry.get(service_name)
                results[service_name].append(service)

            threads = []
            for service_name in ['llm1', 'embedding1', 'db1']:
                for _ in range(5):
                    thread = threading.Thread(target=get_service, args=(service_name,))
                    threads.append(thread)
                    thread.start()

            for thread in threads:
                thread.join()

            # Each service should have consistent instances
            for service_name, instances in results.items():
                assert len(instances) == 5
                assert len(set(id(i) for i in instances)) == 1

    def test_concurrent_different_services_independent_locks(self, mock_services_file):
        """Test that different services use independent locks."""
        registry = ServiceRegistry(mock_services_file)

        instantiation_times = {}
        lock = threading.Lock()

        def timed_create(service_name, config):
            """Track instantiation timing."""
            start = time.time()
            time.sleep(0.1)  # Simulate work
            end = time.time()
            with lock:
                instantiation_times[service_name] = (start, end)
            return Mock()

        with patch.object(registry._factory, 'create_service', side_effect=timed_create):
            threads = []

            def get_service(service_name):
                registry.get(service_name)

            # Start threads for different services simultaneously
            for service_name in ['llm1', 'embedding1', 'db1']:
                thread = threading.Thread(target=get_service, args=(service_name,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            # Check that services were instantiated concurrently
            # (overlapping time windows indicate concurrent execution)
            times = list(instantiation_times.values())
            assert len(times) == 3

            # At least some services should have overlapping instantiation times
            # This proves they weren't blocked by a global lock
            overlaps = 0
            for i in range(len(times)):
                for j in range(i + 1, len(times)):
                    start1, end1 = times[i]
                    start2, end2 = times[j]
                    # Check if time windows overlap
                    if (start1 <= start2 <= end1) or (start2 <= start1 <= end2):
                        overlaps += 1

            # Should have at least one overlap (proves concurrent execution)
            assert overlaps >= 1


class TestConcurrentReload:
    """Tests for concurrent reload operations."""

    def test_reload_while_getting_service(self, mock_services_file):
        """Test reloading service while other threads are getting it."""
        registry = ServiceRegistry(mock_services_file)

        old_service = Mock()
        old_service.close = Mock()
        new_service = Mock()

        call_count = 0

        def create_service_side_effect(service_name, config):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return old_service
            else:
                return new_service

        with patch.object(registry._factory, 'create_service', side_effect=create_service_side_effect):
            # Get service initially
            registry.get("llm1")

            results = []
            threads = []

            def get_service():
                service = registry.get("llm1")
                results.append(service)

            def reload_service():
                time.sleep(0.05)  # Small delay
                registry.reload("llm1")

            # Start threads getting service
            for _ in range(5):
                thread = threading.Thread(target=get_service)
                threads.append(thread)
                thread.start()

            # Start reload thread
            reload_thread = threading.Thread(target=reload_service)
            threads.append(reload_thread)
            reload_thread.start()

            # Wait for all
            for thread in threads:
                thread.join()

            # Should have called close on old service
            old_service.close.assert_called_once()

            # Results should contain either old or new service
            # (depending on timing)
            unique_services = set(id(s) for s in results)
            assert len(unique_services) <= 2  # At most old and new


class TestConcurrentShutdown:
    """Tests for concurrent shutdown operations."""

    def test_shutdown_while_getting_services(self, mock_services_file):
        """Test shutdown while other threads are getting services."""
        registry = ServiceRegistry(mock_services_file)

        mock_service = Mock()
        mock_service.close = Mock()

        with patch.object(registry._factory, 'create_service', return_value=mock_service):
            threads = []
            errors = []

            def get_service():
                try:
                    registry.get("llm1")
                except Exception as e:
                    errors.append(e)

            def shutdown_registry():
                time.sleep(0.05)
                registry.shutdown()

            # Start threads getting service
            for _ in range(5):
                thread = threading.Thread(target=get_service)
                threads.append(thread)
                thread.start()

            # Start shutdown thread
            shutdown_thread = threading.Thread(target=shutdown_registry)
            threads.append(shutdown_thread)
            shutdown_thread.start()

            for thread in threads:
                thread.join()

            # Shutdown should have been called
            # Some gets might succeed, some might fail depending on timing
            # But no crashes should occur


class TestRaceConditions:
    """Tests for potential race conditions."""

    def test_no_duplicate_instantiation_race(self, mock_services_file):
        """Test that race conditions don't cause duplicate instantiation."""
        registry = ServiceRegistry(mock_services_file)

        instantiation_count = 0
        lock = threading.Lock()

        def counted_create(service_name, config):
            nonlocal instantiation_count
            # Simulate slow creation to increase chance of race
            time.sleep(0.05)
            with lock:
                instantiation_count += 1
            return Mock()

        with patch.object(registry._factory, 'create_service', side_effect=counted_create):
            threads = []

            def get_service():
                registry.get("llm1")

            # Launch many threads simultaneously
            for _ in range(50):
                thread = threading.Thread(target=get_service)
                threads.append(thread)

            # Start all at once
            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            # Should only instantiate once despite 50 concurrent requests
            assert instantiation_count == 1

    def test_cache_consistency_under_load(self, mock_services_file):
        """Test cache remains consistent under heavy concurrent load."""
        registry = ServiceRegistry(mock_services_file)

        services_created = {
            'llm1': Mock(),
            'embedding1': Mock(),
            'db1': Mock()
        }

        def create_service(service_name, config):
            time.sleep(0.01)  # Small delay
            return services_created[service_name]

        with patch.object(registry._factory, 'create_service', side_effect=create_service):
            results = {
                'llm1': [],
                'embedding1': [],
                'db1': []
            }
            threads = []

            def get_random_service(service_name):
                service = registry.get(service_name)
                results[service_name].append(id(service))

            # Create many threads accessing different services
            import random
            service_names = ['llm1', 'embedding1', 'db1']
            for _ in range(100):
                service_name = random.choice(service_names)
                thread = threading.Thread(target=get_random_service, args=(service_name,))
                threads.append(thread)

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            # Each service should have exactly one unique instance
            for service_name, instance_ids in results.items():
                if instance_ids:  # If service was accessed
                    unique_ids = set(instance_ids)
                    assert len(unique_ids) == 1
                    assert unique_ids.pop() == id(services_created[service_name])
