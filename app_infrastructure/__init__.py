"""
================================================================================
PaleoAST - Application Infrastructure
================================================================================

本包包含应用基础设施代码：
- 目录结构修复与AST源码审计 (audit)
- 商业级启动引擎与闪屏动画 (startup)
- 全局异常拦截与日志系统 (exception_handler)
- 巨型QSS样式引擎 (theme)

作者: PaleoAST Development Team
版本: 1.0.1
"""

from . import audit, startup, theme

__all__ = ["audit", "startup", "theme"]
