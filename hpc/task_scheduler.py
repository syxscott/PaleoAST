"""
================================================================================
PaleoAST HPC - Task Scheduler Module
================================================================================

任务调度器，支持依赖关系、优先级、动态负载均衡。

作者: PaleoAST Development Team
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """任务优先级"""

    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


class TaskState(Enum):
    """任务状态"""

    CREATED = auto()
    WAITING = auto()
    READY = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    BLOCKED = auto()


@dataclass
class ScheduledTask:
    """
    调度任务

    属性:
        task_id: 唯一标识
        func: 执行函数
        args: 位置参数
        kwargs: 关键字参数
        dependencies: 依赖任务ID
        priority: 优先级
        state: 当前状态
        result: 执行结果
        error: 错误信息
    """

    task_id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    dependencies: set[str] = field(default_factory=set)
    priority: TaskPriority = TaskPriority.NORMAL
    state: TaskState = TaskState.CREATED
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=lambda: __import__("time").time())


class TaskScheduler:
    """
    任务调度器

    支持:
        1. 任务依赖管理
        2. 优先级调度
        3. 动态负载均衡
        4. 任务取消和重试

    算法:
        - 使用拓扑排序确定任务执行顺序
        - 优先级队列管理就绪任务
        - 依赖追踪实现DAG执行
    """

    def __init__(self, n_workers: int = 4):
        """
        初始化调度器

        参数:
            n_workers: 工作线程数
        """
        self._n_workers = n_workers
        self._tasks: dict[str, ScheduledTask] = {}
        self._ready_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._completed: set[str] = set()
        self._failed: set[str] = set()
        self._results: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"{__name__}.TaskScheduler")
        self._running = False
        self._worker_threads: list[threading.Thread] = []
        self._shutdown_event = threading.Event()

    def add_task(
        self,
        func: Callable,
        task_id: str | None = None,
        dependencies: list[str] | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        *args,
        **kwargs,
    ) -> str:
        """
        添加任务

        参数:
            func: 执行函数
            task_id: 任务ID (可选，自动生成)
            dependencies: 依赖任务ID列表
            priority: 优先级
            *args: 位置参数
            **kwargs: 关键字参数

        返回:
            任务ID
        """
        if task_id is None:
            task_id = str(uuid.uuid4())

        if task_id in self._tasks:
            raise ValueError(f"Task {task_id} already exists")

        deps = set(dependencies) if dependencies else set()

        # 验证依赖存在
        for dep_id in deps:
            if dep_id not in self._tasks and dep_id not in self._completed:
                raise ValueError(f"Dependency '{dep_id}' not found for task '{task_id}'")

        task = ScheduledTask(task_id=task_id, func=func, args=args, kwargs=kwargs, dependencies=deps, priority=priority)

        with self._lock:
            self._tasks[task_id] = task

        self._logger.debug(f"Added task {task_id} with priority {priority.name}")

        # 检查是否就绪
        self._check_ready(task_id)

        return task_id

    def _check_ready(self, task_id: str) -> None:
        """检查任务是否就绪"""
        with self._lock:
            if task_id not in self._tasks:
                return

            task = self._tasks[task_id]

            if task.state != TaskState.CREATED:
                return

            # 检查依赖是否完成
            deps_completed = all(dep_id in self._completed for dep_id in task.dependencies)

            if deps_completed:
                task.state = TaskState.READY
                self._ready_queue.put((task.priority.value, task.created_at, task.task_id))
                self._logger.debug(f"Task {task_id} is now READY")

    def start(self) -> None:
        """启动调度器"""
        if self._running:
            return

        self._running = True
        self._shutdown_event.clear()

        for i in range(self._n_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"SchedulerWorker-{i}", daemon=True)
            worker.start()
            self._worker_threads.append(worker)

        self._logger.info(f"Started scheduler with {self._n_workers} workers")

    def shutdown(self, wait: bool = True) -> None:
        """关闭调度器"""
        self._running = False
        self._shutdown_event.set()

        if wait:
            for worker in self._worker_threads:
                worker.join(timeout=5.0)

        self._worker_threads.clear()
        self._logger.info("Scheduler shut down")

    def _worker_loop(self) -> None:
        """Worker线程主循环"""
        while self._running and not self._shutdown_event.is_set():
            try:
                # 从就绪队列获取任务
                _priority, _created_at, task_id = self._ready_queue.get(timeout=0.1)

                # 获取任务
                with self._lock:
                    if task_id not in self._tasks:
                        continue

                    task = self._tasks[task_id]

                    if task.state != TaskState.READY:
                        continue

                    task.state = TaskState.RUNNING

                # 执行任务
                try:
                    self._logger.debug(f"Executing task {task_id}")

                    result = task.func(*task.args, **task.kwargs)

                    with self._lock:
                        task.result = result
                        task.state = TaskState.COMPLETED
                        self._completed.add(task_id)
                        self._results[task_id] = result

                    self._logger.debug(f"Task {task_id} completed")

                    # 触发依赖此任务的其他任务
                    self._unblock_dependents(task_id)

                except Exception as e:
                    with self._lock:
                        task.error = str(e)
                        task.state = TaskState.FAILED
                        self._failed.add(task_id)

                    self._logger.error(f"Task {task_id} failed: {e}")

                self._ready_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self._logger.error(f"Worker error: {e}")

    def _unblock_dependents(self, completed_id: str) -> None:
        """解除依赖任务的阻塞"""
        with self._lock:
            for task_id, task in self._tasks.items():
                if completed_id in task.dependencies:
                    self._check_ready(task_id)

    def get_result(self, task_id: str, timeout: float | None = None) -> Any:
        """
        获取任务结果

        参数:
            task_id: 任务ID
            timeout: 超时时间

        返回:
            任务结果
        """
        start_time = __import__("time").time()

        while True:
            with self._lock:
                if task_id in self._completed:
                    return self._results.get(task_id)

                if task_id in self._failed:
                    raise RuntimeError(f"Task {task_id} failed: {self._tasks[task_id].error}")

            if timeout and (time.time() - start_time) > timeout:
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
        start_time = time.time()

        while True:
            with self._lock:
                pending = len(self._tasks) - len(self._completed) - len(self._failed)

                if pending == 0:
                    return self._results.copy()

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Timeout waiting for all tasks")

            time.sleep(0.1)

    def get_status(self) -> dict[str, Any]:
        """获取调度器状态"""
        with self._lock:
            return {
                "total_tasks": len(self._tasks),
                "completed": len(self._completed),
                "failed": len(self._failed),
                "ready": self._ready_queue.qsize(),
                "running": sum(1 for t in self._tasks.values() if t.state == TaskState.RUNNING),
            }
