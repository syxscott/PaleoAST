"""
================================================================================
PaleoAST Phase 3 - HPC Module
================================================================================

高性能计算模块，提供多进程并行计算能力。

作者: PaleoAST Development Team
版本: 3.0.0
"""

from .process_pool import ProcessPool, Task
from .task_scheduler import TaskScheduler, TaskStatus
from .shared_memory import SharedMemoryManager
from .memory_map import MemoryMappedMatrix
from .progress_queue import ProgressQueue

__all__ = [
    'ProcessPool',
    'Task',
    'TaskScheduler',
    'TaskStatus',
    'SharedMemoryManager',
    'MemoryMappedMatrix',
    'ProgressQueue',
]
