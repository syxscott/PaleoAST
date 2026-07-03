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
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from multiprocessing import Manager, Pool
from typing import Any, TypeVar

import numpy as np

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

    def map(
        self,
        func: Callable[[Any], R],
        items: list[Any],
        chunk_size: int = 1,
        callback: Callable[[R], None] | None = None,
    ) -> list[R]:
        """
        并行映射

        参数:
            func: 要应用的函数
            items: 输入项列表
            chunk_size: 每个任务处理的项数
            callback: 结果回调

        返回:
            结果列表
        """
        if self._pool is None:
            self.start()

        if not items:
            return []

        # 分块
        chunks = self._chunk_items(items, chunk_size)

        results = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            try:
                result = self._pool.apply_async(_worker_map, args=(func, chunk))
                results.append(result)
            except Exception as e:
                self._logger.error(f"Failed to submit chunk {i}: {e}")

        # 收集结果
        output = []
        for i, result in enumerate(results):
            try:
                chunk_result = result.get(timeout=300)
                output.extend(chunk_result)

                if callback:
                    for item in chunk_result:
                        if item is not None:
                            callback(item)

                # 更新进度
                failed = sum(1 for item in chunk_result if item is None)
                if failed > 0:
                    self._logger.warning(f"Chunk {i}: {failed}/{len(chunk_result)} items failed")
                progress = (i + 1) / total
                self._report_progress(progress, f"Processed chunk {i + 1}/{total}")

            except Exception as e:
                self._logger.error(f"Chunk {i} failed: {e}")

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
        self._pool.apply_async(
            _worker_execute,
            args=(func, args, kwargs, task_id),
            callback=self._on_task_complete,
            error_callback=self._on_task_error,
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

    def _on_task_error(self, error: Exception) -> None:
        """任务错误回调"""
        self._logger.error(f"Task failed with exception: {error}")

    def get_result(self, task_id: str, timeout: float | None = None) -> Any:
        """
        获取任务结果

        参数:
            task_id: 任务ID
            timeout: 超时时间

        返回:
            任务结果
        """
        if task_id not in self._results:
            raise KeyError(f"Task {task_id} not found or not completed")

        return self._results[task_id]

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
            statistic_func: 统计函数

        返回:
            Bootstrap统计量列表
        """
        results = self.map(
            func=_bootstrap_single,
            items=[data] * n_bootstraps,
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
    """单次Bootstrap采样"""
    n = data.shape[0]
    indices = np.random.randint(0, n, size=n)
    sample = data[indices]
    return float(np.mean(sample))
