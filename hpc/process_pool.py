"""
================================================================================
PaleoAST HPC - Process Pool Module
================================================================================

多进程池实现，支持任务分割、进度回调、错误处理。

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from multiprocessing import Manager, Pool
from typing import Any, TypeVar

import numpy as np

from utils.exceptions import ComputationError

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class TaskStatus(Enum):
    """任务状态枚举"""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class Task:
    """
    任务数据类

    属性:
        task_id: 唯一标识
        func: 要执行的函数
        args: 位置参数
        kwargs: 关键字参数
        status: 当前状态
        result: 执行结果
        error: 错误信息
        start_time: 开始时间
        end_time: 结束时间
    """

    task_id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    progress: float = 0.0

    def __post_init__(self):
        if not callable(self.func):
            raise TypeError("func must be callable")


class ProcessPool:
    """
    多进程任务池

    提供高性能并行计算能力，支持进度回调和错误处理。

    核心功能:
        1. 自动进程管理
        2. 任务队列和进度跟踪
        3. 结果收集和聚合
        4. 异常处理和日志

    使用示例:
        >>> pool = ProcessPool(n_workers=4)
        >>> results = pool.map(
        ...     func=compute_distance,
        ...     items=distance_pairs,
        ...     chunk_size=100
        ... )
    """

    def __init__(
        self,
        n_workers: int | None = None,
        max_tasks_per_worker: int = 10,
        progress_callback: Callable[[float, str], None] | None = None,
    ):
        """
        初始化进程池

        参数:
            n_workers: 工作进程数 (默认CPU核心数)
            max_tasks_per_worker: 每个worker的最大任务数
            progress_callback: 进度回调函数 (progress: float, message: str)
        """
        self._n_workers = n_workers or mp.cpu_count()
        self._max_tasks = max_tasks_per_worker
        self._progress_callback = progress_callback
        self._pool: Pool | None = None
        self._manager = Manager()
        self._task_queue = self._manager.Queue()
        self._result_queue = self._manager.Queue()
        self._progress_queue = self._manager.Queue()
        self._tasks: dict[str, Task] = {}
        self._results: dict[str, Any] = {}
        # Failures recorded by error_callback, so a crashed task can be told
        # apart from one that has not finished yet.
        self._errors: dict[str, str] = {}
        self._shutdown = False
        self._logger = logging.getLogger(f"{__name__}.ProcessPool")

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()
        return False

    def start(self) -> None:
        """启动进程池"""
        if self._pool is not None:
            return

        self._pool = Pool(processes=self._n_workers, initializer=_worker_init, maxtasksperchild=self._max_tasks)

        self._logger.info(f"Started process pool with {self._n_workers} workers")

    def shutdown(self, timeout: float = 30.0) -> None:
        """关闭进程池"""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

        # 关闭Manager进程
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None

        self._logger.info("Process pool shut down")
        self._shutdown = True

    def map(
        self,
        func: Callable[[Any], R],
        items: list[Any],
        chunk_size: int = 1,
        callback: Callable[[R], None] | None = None,
        raise_on_error: bool = True,
        timeout: float | None = 300.0,
    ) -> list[R]:
        """
        并行映射

        参数:
            func: 要应用的函数
            items: 输入项列表
            chunk_size: 每个任务处理的项数
            callback: 结果回调
            raise_on_error: 默认 True。任一 item 失败即抛错。
                设为 False 则返回带 ``None`` 空洞、部分 chunk 可能缺失的
                列表（仅在你确实要"尽力而为"时使用）。
            timeout: 单个 chunk 的等待上限（秒）。默认 300 秒，
                即此前的硬编码值；传 ``None`` 表示一直等待。
                此前没有这个参数：bootstrap 重复抽样只要跑得慢，
                就会被 300 秒的硬上限判成"chunk 失败"，一个只是
                慢的重复和真正报错的重复在调用方看来完全一样。
                超时会单独记录（``mp.TimeoutError``，而非
                ``concurrent.futures.TimeoutError``）并写进最终错误
                信息，同时仍按失败 chunk 计入；能取消就取消，
                取消不了（``ApplyResult`` 没有 ``cancel()``）会
                在日志里明说。

        返回:
            结果列表

        Raises:
            ComputationError: 有 item 失败或 chunk 整体失败，且
                ``raise_on_error`` 为 True。

        Note:
            旧实现在 item 失败时只写一条日志就把 ``None`` 放进结果里，
            chunk 整体失败时甚至直接跳过——调用方拿到的是一个"看起来
            完整"的列表，无法分辨哪个输入失败了。bootstrap 统计量算出来
            就会少掉那些重复。
        """
        if self._shutdown:
            # shutdown() released the Manager as well as the pool, so
            # start() here would silently resurrect a set of processes the
            # caller had just torn down. Say so instead.
            raise ComputationError(
                "This ProcessPool has been shut down and cannot accept more work; construct a new ProcessPool."
            )

        if self._pool is None:
            self.start()

        if not items:
            return []

        # 分块
        chunks = self._chunk_items(items, chunk_size)

        total = len(chunks)
        failed_chunks: list[int] = []
        timed_out_chunks: list[int] = []

        # (chunk_index, chunk, future) triples rather than a bare list of
        # futures. A bare list desynchronises from `chunks` as soon as one
        # submission raises -- the failed chunk is not appended -- and then
        # `enumerate(results)` indexes the wrong chunk for every later failure.
        # That misattribution is exactly what this function now exists to
        # prevent, so the pairing is carried explicitly instead of implied.
        submitted: list[tuple[int, list, Any]] = []
        for i, chunk in enumerate(chunks):
            try:
                result = self._pool.apply_async(_worker_map, args=(func, chunk))
            except Exception as e:
                self._logger.error(f"Failed to submit chunk {i}: {e}")
                failed_chunks.append(i)
                continue
            submitted.append((i, chunk, result))

        # 收集结果
        output: list[Any] = []
        failed_items: list[Any] = []
        for i, chunk, result in submitted:
            try:
                chunk_result = result.get(timeout=timeout)
                output.extend(chunk_result)

                if callback:
                    for item in chunk_result:
                        if item is not None:
                            callback(item)

                # 更新进度
                for position, value in enumerate(chunk_result):
                    if value is None:
                        failed_items.append(chunk[position])
                progress = (i + 1) / total
                self._report_progress(progress, f"Processed chunk {i + 1}/{total}")

            except mp.TimeoutError:
                # A chunk that merely ran long is not a chunk that failed.
                # The exception here is multiprocessing's own TimeoutError
                # (what ApplyResult.get raises) -- it is unrelated to
                # concurrent.futures.TimeoutError, and letting it fall
                # through to the generic handler below is exactly how a
                # slow replicate came to be reported as a failing chunk.
                # Try to stop the work: the result object is asked to
                # cancel if it supports it. multiprocessing's ApplyResult
                # has no cancel() (that is concurrent.futures), so the
                # chunk keeps running there and we say so in the log
                # rather than pretending it was stopped.
                items_preview = f"{chunk[:5]}{' ...' if len(chunk) > 5 else ''}"
                cancel = getattr(result, "cancel", None)
                if callable(cancel):
                    cancel()
                    self._logger.error(
                        f"Chunk {i} timed out after {timeout}s (items: {items_preview}); future cancelled"
                    )
                else:
                    self._logger.error(
                        f"Chunk {i} timed out after {timeout}s (items: {items_preview}); this pool's "
                        f"result object cannot cancel a running chunk, so it keeps occupying a worker "
                        f"until it finishes. Raise the timeout, or shrink the chunk, if it is genuinely slow."
                    )
                timed_out_chunks.append(i)
                failed_chunks.append(i)
                failed_items.extend(chunk)

            except Exception as e:
                self._logger.error(f"Chunk {i} failed: {e}")
                failed_chunks.append(i)
                failed_items.extend(chunk)

        if (failed_items or failed_chunks) and raise_on_error:
            detail = f"{len(timed_out_chunks)} of them timed out after {timeout}s" if timed_out_chunks else ""
            raise ComputationError(
                f"map() had {len(failed_items)} failing item(s) "
                f"and {len(failed_chunks)} failing chunk(s)"
                f"{'; ' + detail if detail else ''}; returning the "
                f"survivors would silently misrepresent which inputs were "
                f"processed. Failing items: {failed_items[:10]}"
                f"{' ...' if len(failed_items) > 10 else ''}. Pass "
                f"raise_on_error=False to get a best-effort list instead."
            )

        return output

    def submit_task(self, task_id: str, func: Callable, *args, **kwargs) -> Task:
        """
        提交单个任务

        参数:
            task_id: 任务ID
            func: 函数
            *args: 位置参数
            **kwargs: 关键字参数

        返回:
            Task对象
        """
        task = Task(task_id=task_id, func=func, args=args, kwargs=kwargs)

        self._tasks[task_id] = task

        if self._pool is None:
            self.start()

        # 提交到进程池，添加回调收集结果
        # ``error_callback`` receives only the exception, so the id is bound
        # through a closure -- otherwise a transport-level failure cannot be
        # attributed to a task and stays invisible.
        self._pool.apply_async(
            _worker_execute,
            args=(func, args, kwargs, task_id),
            callback=self._on_task_complete,
            error_callback=lambda exc, tid=task_id: self._on_task_error(exc, tid),
        )

        return task

    def _on_task_complete(self, result: tuple) -> None:
        """任务完成回调"""
        task_id, value, error = result
        if task_id in self._tasks:
            task = self._tasks[task_id]
            if error is None:
                task.status = TaskStatus.COMPLETED
                task.result = value
                self._results[task_id] = value
            else:
                task.status = TaskStatus.FAILED
                task.error = error

    def _on_task_error(self, error: Exception, task_id: str | None = None) -> None:
        """Mark a task FAILED instead of only logging.

        This is the ``error_callback`` of ``apply_async``, so it fires when the
        failure is at the transport level (the worker died, the payload could
        not be pickled). The old version only logged, which left the task
        PENDING forever: ``get_result`` then raised ``KeyError`` that read
        like "not ready yet", and ``wait_all`` never saw it finish. A crashed
        task was indistinguishable from a slow one.
        """
        self._logger.error(f"Task {task_id} failed with exception: {error}")
        if task_id is not None and task_id in self._tasks:
            task = self._tasks[task_id]
            task.status = TaskStatus.FAILED
            task.error = str(error)
            # Recorded so get_result can raise the real cause, and so
            # wait_all counts it as finished.
            self._errors[task_id] = error

    def get_result(self, task_id: str, timeout: float | None = None) -> Any:
        """
        获取任务结果，**等待**至完成

        参数:
            task_id: 任务ID
            timeout: 最长等待秒数。``None`` 表示一直等。

        返回:
            任务结果

        Raises:
            KeyError: 任务ID未提交
            TimeoutError: 超时仍未完成
            RuntimeError: 任务失败（附原因）

        Note:
            ``submit_task`` is asynchronous, so the natural
            ``submit_task(...)`` then ``get_result(...)`` sequence used to
            raise ``KeyError: not found or not completed`` whenever the
            worker had not finished yet -- the ``timeout`` parameter was
            accepted and never used.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} was never submitted to this pool")

        deadline = None if timeout is None else time.time() + timeout
        while True:
            if task_id in self._results:
                return self._results[task_id]
            if task_id in self._errors:
                raise RuntimeError(f"Task {task_id} failed: {self._errors[task_id]}")
            task = self._tasks.get(task_id)
            if task is not None and task.status is TaskStatus.FAILED:
                raise RuntimeError(f"Task {task_id} failed: {task.error}")
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            time.sleep(0.01)

    def wait_all(self, timeout: float | None = None) -> dict[str, Any]:
        """
        等待所有任务完成

        参数:
            timeout: 最大等待时间

        返回:
            {task_id: result} 字典
        """
        if self._pool is None:
            return {}

        self._pool.close()
        self._pool.join()
        self._pool = None

        return self._results.copy()

    def _chunk_items(self, items: list[Any], chunk_size: int) -> list[list[Any]]:
        """将列表分块"""
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    def _report_progress(self, progress: float, message: str) -> None:
        """报告进度"""
        if self._progress_callback:
            self._progress_callback(progress, message)
        else:
            self._logger.debug(f"Progress: {progress * 100:.1f}% - {message}")

    def compute_parallel_distance(self, matrix: np.ndarray, metric: str = "euclidean") -> np.ndarray:
        """
        并行计算距离矩阵

        参数:
            matrix: 输入矩阵 (n_samples, n_features)
            metric: 距离度量

        返回:
            距离矩阵 (n_samples, n_samples)
        """
        n = matrix.shape[0]

        # 生成所有对的索引
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))

        # 并行计算 (pass matrix as second element of each item)
        items = [((i, j), matrix) for (i, j) in pairs]
        distances = self.map(
            func=_compute_pair_distance,
            items=items,
            chunk_size=max(1, len(pairs) // (self._n_workers * 4)),
            callback=None,
        )

        # 构建距离矩阵
        dist_matrix = np.zeros((n, n), dtype=np.float64)
        for (i, j), d in zip(pairs, distances, strict=True):
            if d is None:
                raise RuntimeError(f"Distance computation failed for pair ({i}, {j})")
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

        return dist_matrix

    def bootstrap_parallel(
        self, data: np.ndarray, n_bootstraps: int, statistic_func: Callable[[np.ndarray], float]
    ) -> list[float]:
        """
        并行Bootstrap分析

        参数:
            data: 输入数据
            n_bootstraps: Bootstrap次数
            statistic_func: 统计函数。

                **必须可 pickle**——它在 worker 进程里执行，闭包、
                lambda、局部函数都无法跨进程传递。

        返回:
            Bootstrap统计量列表

        Raises:
            ComputationError: 任一重复失败。旧实现在这种情况下把
                ``None`` 放进结果里，调用方算出的 ``np.mean`` 是 NaN，
                却看不出有 40 个重复没跑成。

        Notes:
            ``statistic_func`` is honoured. It used to be accepted, documented
            and then ignored: every replicate went through ``_bootstrap_single``,
            which is hard-wired to ``np.mean``, so a caller asking for a median
            or a percentile interval silently got means.
        """
        results = self.map(
            func=_bootstrap_with_statistic,
            items=[(data, statistic_func)] * n_bootstraps,
            chunk_size=max(1, n_bootstraps // self._n_workers),
            callback=None,
        )

        return results


def _worker_init() -> None:
    """Worker进程初始化"""
    pass


def _worker_execute(func: Callable, args: tuple, kwargs: dict, task_id: str) -> tuple[str, Any, str | None]:
    """
    Worker执行函数

    返回: (task_id, result, error)
    """
    try:
        result = func(*args, **kwargs)
        return (task_id, result, None)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e!s}\n{traceback.format_exc()}"
        return (task_id, None, error_msg)


def _worker_map(func: Callable[[Any], Any], items: list[Any]) -> list[Any]:
    """Worker映射函数 — 逐项执行，单个 item 失败不影响整块结果。

    旧实现使用 ``[func(item) for item in items]``，任一 item 抛出异常
    会导致整个 chunk 的所有结果丢失。改为逐项 try/except，失败项
    记录日志并返回 ``None``，调用方可按需过滤。
    """
    results: list[Any] = []
    for item in items:
        try:
            results.append(func(item))
        except Exception as e:
            logger.error(f"_worker_map: func({item!r}) failed: {type(e).__name__}: {e}")
            results.append(None)
    return results


def _compute_pair_distance(pair_and_matrix: tuple) -> float:
    """计算一对样本的距离"""
    (i, j), matrix = pair_and_matrix
    vec_i = matrix[i]
    vec_j = matrix[j]
    return float(np.sqrt(np.sum((vec_i - vec_j) ** 2)))


def _bootstrap_single(data: np.ndarray) -> float:
    """单次Bootstrap采样(均值)"""
    n = data.shape[0]
    indices = np.random.randint(0, n, size=n)
    sample = data[indices]
    return float(np.mean(sample))


def _bootstrap_with_statistic(payload: tuple[np.ndarray, Callable[[np.ndarray], float]]) -> float:
    """One bootstrap replicate evaluated with the caller's statistic.

    Takes a single ``(data, statistic_func)`` tuple because ``ProcessPool.map``
    forwards each item as one positional argument; it does not unpack
    sequences.

    ``statistic_func`` must be importable (a module-level function, or a
    functools.partial of one) so it survives pickling into the worker.
    """
    data, statistic_func = payload
    n = data.shape[0]
    indices = np.random.randint(0, n, size=n)
    return float(statistic_func(data[indices]))
