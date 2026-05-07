"""
================================================================================
PaleoAST Phase 5 - Startup Loader
================================================================================

本模块实现启动加载器，管理应用程序的初始化过程。

作者: PaleoAST Development Team
"""

from __future__ import annotations
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock

logger = logging.getLogger(__name__)


class StartupPhase(Enum):
    """启动阶段枚举"""
    INITIALIZATION = "initialization"
    DEPENDENCY_CHECK = "dependency_check"
    CONFIG_LOADING = "config_loading"
    DATA_LOADING = "data_loading"
    MODULE_IMPORT = "module_import"
    UI_INITIALIZATION = "ui_initialization"
    READY = "ready"


@dataclass
class StartupProgress:
    """启动进度信息"""
    phase: StartupPhase
    message: str
    progress: int  # 0-100
    details: Optional[str] = None
    error: Optional[str] = None
    
    def is_complete(self) -> bool:
        """是否完成"""
        return self.phase == StartupPhase.READY


@dataclass
class StartupModule:
    """启动模块信息"""
    name: str
    import_func: Callable[[], Any]
    weight: int = 1
    dependencies: List[str] = field(default_factory=list)
    _instance: Any = field(default=None, init=False, repr=False)
    
    def load(self) -> Any:
        """加载模块"""
        if self._instance is None:
            logger.info(f"Loading module: {self.name}")
            self._instance = self.import_func()
            logger.info(f"Module loaded: {self.name}")
        return self._instance


class StartupLoader:
    """
    启动加载器
    
    管理应用程序的启动过程，支持进度追踪和模块依赖管理。
    
    使用示例:
        >>> loader = StartupLoader()
        >>> loader.register_module("numpy", lambda: __import__('numpy'))
        >>> loader.add_callback(lambda p: print(f"Progress: {p.progress}"))
        >>> loader.start()
    """
    
    def __init__(self):
        """初始化加载器"""
        self._modules: Dict[str, StartupModule] = {}
        self._callbacks: List[Callable[[StartupProgress], None]] = []
        self._progress_callbacks: List[Callable[[int, str], None]] = []
        self._current_progress = StartupProgress(
            phase=StartupPhase.INITIALIZATION,
            message="Initializing...",
            progress=0
        )
        self._is_loading = False
        self._is_cancelled = False
        self._lock = Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._loaded_modules: Dict[str, Any] = {}
        
        logger.info("StartupLoader initialized")
    
    def register_module(
        self,
        name: str,
        import_func: Callable[[], Any],
        weight: int = 1,
        dependencies: List[str] = None
    ) -> None:
        """
        注册模块
        
        参数:
            name: 模块名称
            import_func: 导入函数
            weight: 权重
            dependencies: 依赖列表
        """
        with self._lock:
            self._modules[name] = StartupModule(
                name=name,
                import_func=import_func,
                weight=weight,
                dependencies=dependencies or []
            )
            logger.debug(f"Registered module: {name} (weight: {weight})")
    
    def add_callback(
        self, 
        callback: Callable[[StartupProgress], None]
    ) -> None:
        """
        添加进度回调
        
        参数:
            callback: 回调函数
        """
        self._callbacks.append(callback)
    
    def add_progress_callback(
        self,
        callback: Callable[[int, str], None]
    ) -> None:
        """
        添加简化进度回调
        
        参数:
            callback: 回调函数 (progress: int, message: str)
        """
        self._progress_callbacks.append(callback)
    
    def _emit_progress(self, progress: StartupProgress) -> None:
        """发送进度更新"""
        self._current_progress = progress
        
        for callback in self._callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
        
        for callback in self._progress_callbacks:
            try:
                callback(progress.progress, progress.message)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    def _resolve_dependencies(self, module_name: str) -> List[str]:
        """解析依赖"""
        if module_name not in self._modules:
            return []
        
        module = self._modules[module_name]
        resolved = []
        visited = set()
        
        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            
            if name in self._modules:
                for dep in self._modules[name].dependencies:
                    visit(dep)
                if name not in resolved:
                    resolved.append(name)
        
        visit(module_name)
        return resolved
    
    def _calculate_total_weight(self, module_names: List[str]) -> int:
        """计算总权重"""
        return sum(
            self._modules[name].weight 
            for name in module_names 
            if name in self._modules
        )
    
    def start(self, splash_callback: Callable[[int, str], None] = None) -> Dict[str, Any]:
        """
        开始加载
        
        参数:
            splash_callback: 闪屏回调函数
        
        返回:
            加载的模块字典
        """
        if self._is_loading:
            logger.warning("Already loading")
            return self._loaded_modules
        
        self._is_loading = True
        self._is_cancelled = False
        self._loaded_modules = {}
        
        # 阶段1: 初始化
        self._emit_progress(StartupProgress(
            phase=StartupPhase.INITIALIZATION,
            message="Starting PaleoAST...",
            progress=0
        ))
        if splash_callback:
            splash_callback(0, "Starting PaleoAST...")
        
        time.sleep(0.1)
        
        # 阶段2: 依赖检查
        self._emit_progress(StartupProgress(
            phase=StartupPhase.DEPENDENCY_CHECK,
            message="Checking dependencies...",
            progress=10
        ))
        if splash_callback:
            splash_callback(10, "Checking dependencies...")
        
        self._check_dependencies()
        time.sleep(0.1)
        
        # 阶段3: 配置加载
        self._emit_progress(StartupProgress(
            phase=StartupPhase.CONFIG_LOADING,
            message="Loading configuration...",
            progress=20
        ))
        if splash_callback:
            splash_callback(20, "Loading configuration...")
        
        time.sleep(0.1)
        
        # 阶段4: 模块导入
        self._emit_progress(StartupProgress(
            phase=StartupPhase.MODULE_IMPORT,
            message="Importing modules...",
            progress=30
        ))
        if splash_callback:
            splash_callback(30, "Importing modules...")
        
        # 解析所有模块依赖
        all_module_names = list(self._modules.keys())
        
        # 按依赖顺序加载
        loaded = set()
        remaining = set(all_module_names)
        
        base_progress = 30
        end_progress = 80
        
        while remaining:
            progress_step = (end_progress - base_progress) / max(len(remaining), 1)
            
            # 找到没有未加载依赖的模块
            ready = [
                name for name in remaining
                if all(dep in loaded for dep in self._modules[name].dependencies)
            ]
            
            if not ready:
                # 循环依赖检测
                logger.error(f"Circular dependency detected in: {remaining}")
                break
            
            # 加载准备好的模块
            for name in ready:
                msg = f"Loading {name}..."
                current_progress = base_progress + progress_step * (len(all_module_names) - len(remaining))
                
                self._emit_progress(StartupProgress(
                    phase=StartupPhase.MODULE_IMPORT,
                    message=msg,
                    progress=int(current_progress)
                ))
                if splash_callback:
                    splash_callback(int(current_progress), msg)
                
                try:
                    self._loaded_modules[name] = self._modules[name].load()
                    loaded.add(name)
                except Exception as e:
                    logger.error(f"Failed to load module {name}: {e}")
                    self._emit_progress(StartupProgress(
                        phase=StartupPhase.MODULE_IMPORT,
                        message=f"Error loading {name}",
                        progress=int(current_progress),
                        error=str(e)
                    ))
                
                remaining.discard(name)
                time.sleep(0.05)
        
        # 阶段5: 数据加载
        self._emit_progress(StartupProgress(
            phase=StartupPhase.DATA_LOADING,
            message="Loading data...",
            progress=85
        ))
        if splash_callback:
            splash_callback(85, "Loading data...")
        
        time.sleep(0.1)
        
        # 阶段6: UI初始化
        self._emit_progress(StartupProgress(
            phase=StartupPhase.UI_INITIALIZATION,
            message="Initializing UI...",
            progress=90
        ))
        if splash_callback:
            splash_callback(90, "Initializing UI...")
        
        time.sleep(0.1)
        
        # 完成
        self._emit_progress(StartupProgress(
            phase=StartupPhase.READY,
            message="Ready!",
            progress=100
        ))
        if splash_callback:
            splash_callback(100, "Ready!")
        
        self._is_loading = False
        
        logger.info(f"Startup complete. Loaded {len(self._loaded_modules)} modules")
        return self._loaded_modules
    
    def _check_dependencies(self) -> None:
        """检查依赖"""
        missing = []
        
        for name, module in self._modules.items():
            try:
                # 尝试导入以验证
                module.load()
            except ImportError as e:
                missing.append((name, str(e)))
                logger.error(f"Missing dependency {name}: {e}")
        
        if missing:
            self._emit_progress(StartupProgress(
                phase=StartupPhase.DEPENDENCY_CHECK,
                message="Missing dependencies detected",
                progress=15,
                error=f"Missing: {[m[0] for m in missing]}"
            ))
    
    def cancel(self) -> None:
        """取消加载"""
        self._is_cancelled = True
        logger.info("Loading cancelled")
    
    @property
    def is_loading(self) -> bool:
        """是否正在加载"""
        return self._is_loading
    
    @property
    def loaded_modules(self) -> Dict[str, Any]:
        """获取已加载模块"""
        return self._loaded_modules.copy()


# 预定义的PaleoAST模块加载器
class PaleoASTLoader(StartupLoader):
    """
    PaleoAST专用加载器
    
    预配置了所有PaleoAST模块的加载顺序。
    """
    
    def __init__(self):
        """初始化PaleoAST加载器"""
        super().__init__()
        self._register_paleoast_modules()
    
    def _register_paleoast_modules(self) -> None:
        """注册PaleoAST模块"""
        # 核心配置
        self.register_module(
            "config",
            lambda: __import__('config'),
            weight=1
        )
        
        # 工具模块 (无依赖)
        self.register_module(
            "utils",
            lambda: __import__('utils'),
            weight=1
        )
        
        # 数据模型 (无依赖)
        self.register_module(
            "models",
            lambda: __import__('models'),
            weight=2,
            dependencies=["config"]
        )
        
        # 统计引擎
        self.register_module(
            "statistics",
            lambda: __import__('statistics'),
            weight=3,
            dependencies=["utils", "models"]
        )
        
        # 形态测量
        self.register_module(
            "morphometrics",
            lambda: __import__('morphometrics'),
            weight=3,
            dependencies=["utils", "models"]
        )
        
        # 3D形态测量
        self.register_module(
            "morpho3d",
            lambda: __import__('morpho3d'),
            weight=3,
            dependencies=["morphometrics", "utils"]
        )
        
        # 古生态学
        self.register_module(
            "ecology",
            lambda: __import__('ecology'),
            weight=3,
            dependencies=["utils", "models"]
        )
        
        # 生物地层学
        self.register_module(
            "stratigraphy",
            lambda: __import__('stratigraphy'),
            weight=3,
            dependencies=["utils", "models"]
        )
        
        # 可视化
        self.register_module(
            "visualization",
            lambda: __import__('visualization'),
            weight=2,
            dependencies=["models", "statistics"]
        )
        
        # 控制器
        self.register_module(
            "controllers",
            lambda: __import__('controllers'),
            weight=2,
            dependencies=["models", "statistics", "visualization"]
        )
        
        # 视图 (最后加载)
        self.register_module(
            "views",
            lambda: __import__('views'),
            weight=2,
            dependencies=["controllers", "visualization"]
        )
        
        logger.info("Registered PaleoAST modules")
