# =============================================================================
# FILE: tests/models/test_state_manager.py
# =============================================================================
"""
Thread safety tests for StateManager singleton.

These tests verify that the StateManager singleton pattern is thread-safe
under concurrent access scenarios.

Author: PaleoAST Development Team
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pytest

from models.state_manager import StateManager, get_state_manager


class TestStateManagerThreadSafety:
    """Tests for thread-safe singleton behavior."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        StateManager.reset_instance()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        StateManager.reset_instance()

    def test_concurrent_get_instance_returns_same_instance(self) -> None:
        """
        Test that concurrent calls to get_instance() return the same instance.

        Uses threading.Barrier to synchronize threads so they all call
        get_instance() at approximately the same time, maximizing race condition
        potential.
        """
        num_threads = 20
        barrier = threading.Barrier(num_threads)
        instances: List[StateManager | None] = [None] * num_threads
        errors: List[Exception] = []

        def get_instance_task(index: int) -> None:
            try:
                # Wait for all threads to be ready, then call get_instance()
                barrier.wait()
                instances[index] = StateManager.get_instance()
            except Exception as e:
                errors.append(e)

        # Run threads concurrently
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=get_instance_task, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors during concurrent get_instance: {errors}"

        # Verify all instances are the same object
        first_instance = instances[0]
        assert first_instance is not None
        for i, instance in enumerate(instances):
            assert instance is first_instance, (
                f"Thread {i} got different instance than thread 0"
            )

    def test_concurrent_get_instance_via_executor(self) -> None:
        """
        Alternative test using ThreadPoolExecutor for concurrent access.

        This test uses a thread pool which has different timing characteristics
        than manual thread management.
        """
        num_threads = 50

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(StateManager.get_instance) for _ in range(num_threads)]
            instances = [f.result() for f in futures]

        # All instances should be the same
        first_instance = instances[0]
        assert first_instance is not None
        for i, instance in enumerate(instances):
            assert instance is first_instance, (
                f"Future {i} got different instance"
            )

    def test_reset_instance_creates_new_instance(self) -> None:
        """
        Test that reset_instance() followed by get_instance() creates a new instance.
        """
        # Get first instance
        instance1 = StateManager.get_instance()
        assert instance1 is not None

        # Reset singleton
        StateManager.reset_instance()

        # Get new instance - should be a different object
        instance2 = StateManager.get_instance()
        assert instance2 is not None
        assert instance1 is not instance2, "reset_instance should create a new instance"

    def test_concurrent_state_modifications(self) -> None:
        """
        Test that concurrent state modifications don't cause data races.

        This test has multiple threads modifying visualization settings
        concurrently and verifies that all modifications are applied correctly.
        """
        num_threads = 10
        iterations = 50
        barrier = threading.Barrier(num_threads)
        errors: List[Exception] = []

        state = StateManager.get_instance()

        def modify_settings_task(thread_id: int) -> None:
            try:
                barrier.wait()  # Synchronize start
                for i in range(iterations):
                    key = f"setting_{thread_id}_{i}"
                    value = f"value_{thread_id}_{i}"
                    state.set_visualization_setting(key, value)

                    # Verify the setting was set
                    retrieved = state.get_visualization_setting(key)
                    assert retrieved == value, (
                        f"Thread {thread_id}, iteration {i}: "
                        f"expected {value}, got {retrieved}"
                    )
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=modify_settings_task, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent modifications: {errors}"

        # Verify all settings were persisted
        for thread_id in range(num_threads):
            for i in range(iterations):
                key = f"setting_{thread_id}_{i}"
                value = f"value_{thread_id}_{i}"
                retrieved = state.get_visualization_setting(key)
                assert retrieved == value, (
                    f"Final verification failed for {key}: expected {value}, got {retrieved}"
                )

    def test_concurrent_read_write_operations(self) -> None:
        """
        Test that concurrent reads and writes don't cause deadlocks or errors.

        Multiple threads perform reads while others perform writes,
        verifying thread-safe access to state.
        """
        num_readers = 5
        num_writers = 5
        iterations = 30
        barrier = threading.Barrier(num_readers + num_writers)
        errors: List[Exception] = []

        state = StateManager.get_instance()

        # Pre-populate some settings for reading
        for i in range(10):
            state.set_visualization_setting(f"readable_{i}", f"data_{i}")

        def reader_task(reader_id: int) -> None:
            try:
                barrier.wait()
                for _ in range(iterations):
                    # Read operations
                    _ = state.get_visualization_setting("readable_0")
                    _ = state.get_all_visualization_settings()
                    _ = state.has_data
                    _ = state.is_modified
            except Exception as e:
                errors.append(e)

        def writer_task(writer_id: int) -> None:
            try:
                barrier.wait()
                for i in range(iterations):
                    state.set_visualization_setting(
                        f"writer_{writer_id}_iter_{i}",
                        f"value_{writer_id}_{i}"
                    )
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(num_readers):
            t = threading.Thread(target=reader_task, args=(i,))
            threads.append(t)
            t.start()

        for i in range(num_writers):
            t = threading.Thread(target=writer_task, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent read/write: {errors}"

    def test_nested_lock_acquisition(self) -> None:
        """
        Test that nested lock acquisition (RLock behavior) works correctly.

        RLock allows the same thread to acquire the lock multiple times.
        This tests that read operations can be nested safely.
        """
        state = StateManager.get_instance()

        # Perform nested reads
        with state.read_lock():
            settings1 = state.get_all_visualization_settings()
            with state.read_lock():  # Nested read lock
                settings2 = state.get_all_visualization_settings()
                with state.read_lock():  # Triple nested
                    settings3 = state.get_all_visualization_settings()
                    assert settings1 == settings2 == settings3

    def test_get_state_manager_convenience_function(self) -> None:
        """
        Test that get_state_manager() returns the same instance as get_instance().
        """
        instance1 = StateManager.get_instance()
        instance2 = get_state_manager()

        assert instance1 is instance2

    def test_rapid_sequential_get_instance(self) -> None:
        """
        Test rapid sequential calls to get_instance() return the same instance.
        """
        instances = [StateManager.get_instance() for _ in range(1000)]
        first = instances[0]
        assert first is not None
        for i, inst in enumerate(instances):
            assert inst is first, f"Instance {i} differs from first"


class TestStateManagerSingletonIntegrity:
    """Tests for singleton pattern integrity."""

    def setup_method(self) -> None:
        StateManager.reset_instance()

    def teardown_method(self) -> None:
        StateManager.reset_instance()

    def test_single_instance_enforced(self) -> None:
        """Verify that calling __new__ directly doesn't break singleton."""
        # Direct construction should return same instance
        instance1 = StateManager.get_instance()
        instance2 = StateManager()  # Direct construction
        instance3 = StateManager.get_instance()

        assert instance1 is instance2, "Direct construction should return singleton"
        assert instance2 is instance3, "get_instance should return same singleton"
