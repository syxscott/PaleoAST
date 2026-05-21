# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for PaleoAST

Build command:
    conda activate past
    pyinstaller PaleoAST.spec

Or use the build script:
    python build_exe.py
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# =============================================================================
# 项目路径
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
BUILD_ROOT = PROJECT_ROOT / "build"
DIST_ROOT = PROJECT_ROOT / "dist"

# =============================================================================
# 隐藏导入 (Hidden Imports)
# =============================================================================

# PyQt6 隐藏导入
hiddenimports = [
    # Core PyQt6
    "PyQt6",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "PyQt6.sip",

    # Config 模块
    "config",
    "config.colors",
    "config.constants",
    "config.design_system",
    "config.i18n",
    "config.i18n.translations_en",
    "config.i18n.translations_zh",

    # Models
    "models",
    "models.state_manager",
    "models.data_matrix",

    # Controllers
    "controllers",
    "controllers.data_controller",
    "controllers.statistics_controller",

    # Views
    "views",
    "views.ui_main_window",
    "views.ui_dialogs",
    "views.ui_navigation",
    "views.ui_plot_canvas",
    "views.ui_spreadsheet",
    "views.ui_pcm_dialogs",
    "views.ui_allometry_dialogs",
    "views.ui_beta_diversity_dialogs",
    "views.ui_evolution_rate_dialogs",
    "views.ui_extinction_dialogs",
    "views.ui_null_model_dialogs",

    # Statistics
    "statistics",

    # Ecology
    "ecology",

    # Morphometrics
    "morphometrics",
    "morphometrics.gpa",
    "morphometrics.evolution_rate",

    # Phylogenetics
    "phylogenetics",

    # Stratigraphy
    "stratigraphy",
    "stratigraphy.extinction",

    # Visualization
    "visualization",

    # Utils
    "utils",
    "utils.exceptions",
    "utils.event_bus",

    # App infrastructure
    "app_infrastructure",
    "app_infrastructure.exception_handler",

    # scipy / numpy 扩展
    "scipy",
    "scipy.linalg",
    "scipy.spatial",
    "scipy.stats",
    "numpy",
    "numpy.core",
    "numpy.linalg",
    "pandas",
    "matplotlib",
    "matplotlib.backends",
    "matplotlib.backends.backend_qtagg",
    "psutil",
]

# =============================================================================
# 数据文件 (Data Files)
# =============================================================================

datas = [
    # Logo
    (str(PROJECT_ROOT / "logo.png"), "."),

    # i18n 翻译文件
    (str(PROJECT_ROOT / "config" / "i18n" / "translations_en.py"), "config/i18n"),
    (str(PROJECT_ROOT / "config" / "i18n" / "translations_zh.py"), "config/i18n"),
]

# =============================================================================
# 收集子模块
# =============================================================================

# 收集所有子模块以确保完整打包
for module_name in ["scipy", "numpy", "pandas", "matplotlib", "sklearn"]:
    try:
        hiddenimports.extend(collect_submodules(module_name))
    except Exception:
        pass

# 收集 PyQt6 数据文件
try:
    from PyInstaller.utils.hooks import collect_data_files
    qt_datas, qt_binaries = collect_data_files("PyQt6", include_py_files=True)
    hiddenimports.append("PyQt6")
except Exception:
    qt_datas = []
    qt_binaries = []

# =============================================================================
# PyInstaller 分析
# =============================================================================

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=qt_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "test",
        "pytest",
        "IPython",
        "notebook",
        "jupyter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# =============================================================================
# PYZ 打包
# =============================================================================

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# =============================================================================
# EXE 可执行文件
# =============================================================================

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PaleoAST",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # 不显示控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "logo.png") if (PROJECT_ROOT / "logo.png").exists() else None,
)

# =============================================================================
# 收集文件到输出目录
# =============================================================================

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PaleoAST",
)

# =============================================================================
# 可选：创建 Windows 安装程序 (使用 --onedir 模式)
# =============================================================================

# 如果需要创建单个可执行文件，可以使用以下配置
# 或者使用 NSIS / InnoSetup 创建安装程序
