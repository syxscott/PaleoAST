"""
================================================================================
PaleoAST HPC - Process Pool Module
================================================================================

多进程池实现，支持任务分割、进度回调、错误处理。

作者: PaleoAST Development Team
"""

from __future__ import annotations
from typing import (
    Dict, List, Optional, Callable, Any, Tuple, TypeVar
)
from dataclasses import dataclass, field
from enum import Enum, auto
import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager, Process
import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, Future
from queue import Empty
import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


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
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
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
        n_workers: Optional[int] = None,
        max_tasks_per_worker: int = 10,
        progress_callback: Optional[Callable[[float, str], None]] = None
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
        self._pool: Optional[Pool] = None
        self._manager = Manager()
        self._task_queue = self._manager.Queue()
        self._result_queue = self._manager.Queue()
        self._progress_queue = self._manager.Queue()
        self._tasks: Dict[str, Task] = {}
        self._results: Dict[str, Any] = {}
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
        
        self._pool = Pool(
            processes=self._n_workers,
            initializer=_worker_init,
            maxtasksperchild=self._max_tasks
        )
        
        self._logger.info(f"Started process pool with {self._n_workers} workers")
    
    def shutdown(self, timeout: float = 30.0) -> None:
        """关闭进程池"""
        if self._pool is None:
            return
        
        self._pool.close()
        self._pool.join()
        self._pool = None
        
        self._logger.info("Process pool shut down")
    
    def map(
        self,
        func: Callable[[Any], R],
        items: List[Any],
        chunk_size: int = 1,
        callback: Optional[Callable[[R], None]] = None
    ) -> List[R]:
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
                result = self._pool.apply_async(
                    _worker_map,
                    args=(func, chunk)
                )
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
                        callback(item)
                
                # 更新进度
                progress = (i + 1) / total
                self._report_progress(progress, f"Processed chunk {i+1}/{total}")
                
            except Exception as e:
                self._logger.error(f"Chunk {i} failed: {e}")
        
        return output
    
    def submit_task(
        self,
        task_id: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Task:
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
        task = Task(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs
        )
        
        self._tasks[task_id] = task
        
        if self._pool is None:
            self.start()
        
        # 提交到进程池
        async_result = self._pool.apply_async(
            _worker_execute,
            args=(func, args, kwargs, task_id)
        )
        
        return task
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
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
    
    def wait_all(self, timeout: Optional[float] = None) -> Dict[str, Any]:
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
        
        return self._results.copy()
    
    def _chunk_items(self, items: List[Any], chunk_size: int) -> List[List[Any]]:
        """将列表分块"""
        return [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
    
    def _report_progress(self, progress: float, message: str) -> None:
        """报告进度"""
        if self._progress_callback:
            self._progress_callback(progress, message)
        else:
            self._logger.debug(f"Progress: {progress*100:.1f}% - {message}")
    
    def compute_parallel_distance(
        self,
        matrix: np.ndarray,
        metric: str = 'euclidean'
    ) -> np.ndarray:
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
            callback=None
        )

        # 构建距离矩阵
        dist_matrix = np.zeros((n, n), dtype=np.float64)
        for (i, j), d in zip(pairs, distances):
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
        
        return dist_matrix
    
    def bootstrap_parallel(
        self,
        data: np.ndarray,
        n_bootstraps: int,
        statistic_func: Callable[[np.ndarray], float]
    ) -> List[float]:
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
            callback=None
        )
        
        return results


def _worker_init() -> None:
    """Worker进程初始化"""
    pass


def _worker_execute(
    func: Callable,
    args: Tuple,
    kwargs: Dict,
    task_id: str
) -> Tuple[str, Any, Optional[str]]:
    """
    Worker执行函数
    
    返回: (task_id, result, error)
    """
    try:
        result = func(*args, **kwargs)
        return (task_id, result, None)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return (task_id, None, error_msg)


def _worker_map(
    func: Callable[[Any], Any],
    items: List[Any]
) -> List[Any]:
    """Worker映射函数"""
    return [func(item) for item in items]


def _compute_pair_distance(pair_and_matrix: Tuple) -> float:
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
