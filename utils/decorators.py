# =============================================================================
# FILE: utils/decorators.py
# =============================================================================
"""
Decorator Module for PaleoAST

This module provides decorators for common operations such as
thread safety, memoization, execution timing, and input validation.

Decorator Functions:
    - thread_safe: Ensure method execution is thread-safe
    - memoize: Cache function results
    - log_execution_time: Log function execution duration
    - validate_inputs: Validate function inputs before execution
    - cache_result: Cache results to disk or memory

Author: PaleoAST Development Team
version: 1.0.1
"""

import functools
import inspect
import logging
import threading
import time
from collections.abc import Callable, Hashable
from typing import Any, ParamSpec, TypeVar

import numpy as np

# Configure logger
logger = logging.getLogger(__name__)

# Type variables for generic decorators
P = ParamSpec("P")
T = TypeVar("T")


def thread_safe(method: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator to make a method thread-safe using a lock.

    This decorator wraps a method to ensure only one thread can
    execute it at a time. Each instance gets its own lock.

    Usage:
        Apply to methods that access shared state:

        >>> class DataProcessor:
        ...     def __init__(self):
        ...         self._lock = threading.Lock()
        ...
        ...     @thread_safe
        ...     def process(self, data):
        ...         # Thread-safe operations
        ...         pass

    Note:
        The decorated method must be part of a class with a '_lock'
        attribute (typically threading.RLock or threading.Lock).
        The lock is created automatically if not present.
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
        # Get or create lock
        lock = getattr(self, "_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._lock = lock

        with lock:
            return method(self, *args, **kwargs)

    return wrapper


def memoize(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator to memoize (cache) function results.

    This decorator caches function results based on arguments,
    avoiding redundant computation for pure functions.

    Mathematical Context:
        A pure function f(x, y) always returns the same result
        for the same inputs. Memoization exploits this property:

        memoized_f(x, y) = cache[x, y] if exists
                          = f(x, y) otherwise (then cache)

    Usage:
        >>> @memoize
        ... def expensive_computation(n):
        ...     # Simulate expensive calculation
        ...     return sum(range(n))
        >>> expensive_computation(1000)  # Computed
        499500
        >>> expensive_computation(1000)  # Retrieved from cache
        499500

    Warning:
        Only use with pure functions (no side effects, no random values).
        Results are stored in memory and persist until object deletion.
    """
    cache: dict = {}
    cache_lock = threading.Lock()

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # Create cache key from arguments
        # Need to make args hashable
        key_args = tuple(tuple(arg) if isinstance(arg, (list, np.ndarray)) else arg for arg in args)
        key_kwargs = tuple((k, tuple(v) if isinstance(v, (list, np.ndarray)) else v) for k, v in sorted(kwargs.items()))
        key = (key_args, key_kwargs)

        # Try to get from cache
        try:
            with cache_lock:
                if key in cache:
                    return cache[key]
        except TypeError:
            # Arguments not hashable, skip cache
            pass

        # Compute result
        result = func(*args, **kwargs)

        # Store in cache
        try:
            with cache_lock:
                cache[key] = result
        except TypeError:
            # Result not hashable, skip cache
            pass

        return result

    # Add cache management methods
    wrapper.cache_clear = lambda: cache.clear()
    wrapper.cache_info = lambda: {"size": len(cache)}

    return wrapper


def log_execution_time(
    func: Callable[P, T] | None = None, logger_instance: logging.Logger | None = None, level: int = logging.INFO
) -> Callable:
    """
    Decorator to log function execution time.

    This decorator measures and logs the time taken to execute
    a function, useful for performance monitoring and optimization.

    Usage:
        >>> @log_execution_time
        ... def slow_function():
        ...     time.sleep(1)
        ...     return "done"
        >>> slow_function()
        # INFO - slow_function completed in 1.002s
        'done'

        >>> @log_execution_time(logger_instance=my_logger, level=logging.DEBUG)
        ... def another_function():
        ...     pass

    Mathematical Note:
        Execution time measurement uses:
        - start = time.perf_counter()  # High-resolution timer
        - duration = end - start

        For n repeated calls, average time:
        T̄ = (1/n) * Σ_i=1^n T_i
    """

    def decorator(f: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            nonlocal logger_instance
            log = logger_instance or logger

            # ``result`` must be initialised so the final ``return``
            # works even when ``f`` raises before assignment. The
            # previous version re-raised inside the ``except`` block
            # and then hit ``return result`` with ``result`` unbound,
            # which replaced the real exception with a confusing
            # ``UnboundLocalError``.
            result: T | None = None

            start_time = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                success = True
                error: BaseException | None = None
            except BaseException as e:
                success = False
                error = e
                raise
            finally:
                end_time = time.perf_counter()
                duration = end_time - start_time

                # Format duration
                if duration < 0.001:
                    duration_str = f"{duration * 1_000_000:.2f}μs"
                elif duration < 1:
                    duration_str = f"{duration * 1_000:.2f}ms"
                else:
                    duration_str = f"{duration:.3f}s"

                # Log message
                if success:
                    log.log(level, f"{f.__name__} completed in {duration_str}")
                else:
                    log.log(level, f"{f.__name__} failed after {duration_str}: {error}")

            assert result is not None  # for type checkers; only reached on success
            return result

        return wrapper

    # Handle both @log_execution_time and @log_execution_time()
    if func is not None:
        return decorator(func)
    return decorator


def validate_inputs(**validators: dict) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator factory for validating function inputs.

    This decorator validates function inputs against specified
    validators before executing the function.

    Usage:
        >>> @validate_inputs(
        ...     x={"type": np.ndarray, "ndim": 2},
        ...     y={"type": (int, float), "min": 0}
        ... )
        ... def process(x, y):
        ...     return x * y

        >>> @validate_inputs(
        ...     data={"shape": (None, 3)}  # (any, 3)
        ... )
        ... def transform(data):
        ...     return data * 2

    Validator Options:
        - type: Single type or tuple of types
        - ndim: Exact number of dimensions (for arrays)
        - shape: Tuple of dimension constraints (None = any)
        - min/max: Numeric bounds
        - choices: List of valid choices
        - custom: Custom validation function

    Mathematical Context:
        Input validation ensures mathematical operations are valid:
        - Matrix multiplication: A.shape[1] == B.shape[0]
        - Division: denominator != 0
        - Logarithm: argument > 0
        - Square root: argument >= 0
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Get function signature
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Validate each specified parameter
            for param_name, rules in validators.items():
                if param_name not in bound.arguments:
                    continue

                value = bound.arguments[param_name]

                # Type validation
                if "type" in rules:
                    expected_type = rules["type"]
                    if not isinstance(expected_type, tuple):
                        expected_type = (expected_type,)
                    if not isinstance(value, expected_type):
                        raise TypeError(f"Parameter '{param_name}' must be {expected_type}, got {type(value)}")

                # Array dimension validation
                if "ndim" in rules and isinstance(value, (np.ndarray,)):
                    if value.ndim != rules["ndim"]:
                        raise ValueError(
                            f"Parameter '{param_name}' must have {rules['ndim']} dimensions, got {value.ndim}"
                        )

                # Array shape validation
                if "shape" in rules and isinstance(value, np.ndarray):
                    expected_shape = rules["shape"]
                    if len(expected_shape) != value.ndim:
                        raise ValueError(
                            f"Parameter '{param_name}' shape mismatch: "
                            f"expected {len(expected_shape)}D, got {value.ndim}D"
                        )
                    for i, (exp, actual) in enumerate(zip(expected_shape, value.shape)):
                        if exp is not None and exp != actual:
                            raise ValueError(f"Parameter '{param_name}' dimension {i}: expected {exp}, got {actual}")

                # Numeric range validation
                if "min" in rules:
                    if value < rules["min"]:
                        raise ValueError(f"Parameter '{param_name}' must be >= {rules['min']}, got {value}")
                if "max" in rules:
                    if value > rules["max"]:
                        raise ValueError(f"Parameter '{param_name}' must be <= {rules['max']}, got {value}")

                # Choice validation
                if "choices" in rules:
                    if value not in rules["choices"]:
                        raise ValueError(f"Parameter '{param_name}' must be one of {rules['choices']}, got {value}")

                # Custom validation function
                if "custom" in rules:
                    custom_validator = rules["custom"]
                    if not custom_validator(value):
                        raise ValueError(f"Parameter '{param_name}' failed custom validation")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def cache_result(
    maxsize: int | None = 128, key_func: Callable[..., Hashable] | None = None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to cache function results using LRU cache.

    This is similar to memoize but provides more control over
    cache size and key generation.

    Usage:
        >>> @cache_result(maxsize=256)
        ... def compute_eigenvalues(matrix):
        ...     # Expensive computation
        ...     return np.linalg.eigvals(matrix)

    Parameters:
        maxsize: Maximum number of cached results (None = unlimited)
        key_func: Function to generate cache key from arguments.
                  If None, uses default argument hashing.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        if key_func is not None:
            _cache: dict = {}

            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                cache_key = key_func(*args, **kwargs)
                if cache_key in _cache:
                    return _cache[cache_key]
                result = func(*args, **kwargs)
                if maxsize is not None and len(_cache) >= maxsize:
                    oldest = next(iter(_cache))
                    del _cache[oldest]
                _cache[cache_key] = result
                return result

            wrapper.cache_clear = _cache.clear
            wrapper.cache_info = lambda: {"size": len(_cache)}
        else:
            if maxsize is not None:
                cached_func = functools.lru_cache(maxsize=maxsize)(func)
            else:
                cached_func = functools.lru_cache(maxsize=None)(func)

            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                return cached_func(*args, **kwargs)

            wrapper.cache_clear = cached_func.cache_clear
            wrapper.cache_info = cached_func.cache_info

        return wrapper

    return decorator


def deprecated(
    message: str | None = None, removal_version: str | None = None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to mark functions as deprecated.

    Usage:
        >>> @deprecated("Use new_function instead", removal_version="2.0")
        ... def old_function():
        ...     pass

    Parameters:
        message: Deprecation message explaining the alternative
        removal_version: Version in which the function will be removed
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            warning_msg = f"{func.__name__} is deprecated"
            if message:
                warning_msg += f": {message}"
            if removal_version:
                warning_msg += f" (will be removed in {removal_version})"

            import warnings

            warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def requires_gui(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator to ensure GUI is available before executing.

    Usage:
        >>> @requires_gui
        ... def show_plot():
        ...     # Requires active GUI context
        ...     plt.show()
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            from PyQt6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is None:
                raise RuntimeError("No QApplication instance found. Create QApplication before calling this function.")
        except ImportError:
            raise ImportError("PyQt6 is required for this function. Install with: pip install PyQt6")

        return func(*args, **kwargs)

    return wrapper
