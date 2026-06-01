"""
PyInstaller Runtime Hook for PaleoAST

This hook is executed after PyInstaller creates the executable but before
the main application runs. It ensures Python can find all bundled packages.
"""

import sys
import os
from pathlib import Path

# 获取 _internal 目录路径
if getattr(sys, 'frozen', False):
    # 运行在打包后的环境中
    bundle_dir = Path(sys.executable).parent
    _internal = bundle_dir / "_internal"

    # 添加 _internal 到 sys.path
    # 这样 Python 就能找到所有打包的模块
    if str(_internal) not in sys.path:
        sys.path.insert(0, str(_internal))

    # 确保当前目录是 exe 所在目录
    os.chdir(str(bundle_dir))
