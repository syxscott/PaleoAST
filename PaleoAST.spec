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

# A spec file is EXECUTED, not imported: PyInstaller exec()s it in a namespace
# that has no __file__, so `Path(__file__).parent` raises NameError before a
# single line of the build runs. SPECPATH is the global PyInstaller injects for
# exactly this. The sys.path entry is what lets the spec import PyInstaller
# helpers at all, and matters just as much when the build is driven from a
# different working directory.
import os
import sys

sys.path.insert(0, os.path.dirname(SPECPATH) or ".")
PROJECT_ROOT = Path(SPECPATH)
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
    # NOTE: this was "statistics", which is a STDLIB module name -- there is no
    # statistics/ package in this project (it is `stats/`, renamed precisely
    # because the stdlib shadows it). The real analysis package was therefore
    # never collected, and PyInstaller would have shipped the stdlib one.
    "stats",

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
    # The R export path is reached through a local import inside
    # MainWindow._on_export_as_r_script, so nothing in the import graph
    # mentions it and PyInstaller's analysis would leave it out. Without the
    # entry the File > "Export PCA as R script" action raises ImportError in
    # the frozen build only.
    "visualization.r_export",
    "visualization.r_render",

    # Utils
    "utils",
    "utils.exceptions",
    "utils.event_bus",

    # NOTE: the `app_infrastructure` entries that used to sit here were
    # removed with the package -- nothing imported it, and its
    # exception_handler hard-imports `psutil`, which is only in the `full`
    # extra, so listing it could only ever fail on a minimal build.

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

# Collect submodules of the packages the project actually uses.
#
# This used to cover numpy, scipy, pandas, matplotlib AND sklearn with a bare
# collect_submodules(), which is indiscriminate: it walks every importable
# name in each distribution, including the optional ones. That is how a
# 1.5 GB build ended up carrying torch, polars, OpenCV and four CUDA DLLs.
# The explicit hiddenimports list above covers what the code imports; the
# scientific hooks PyInstaller ships for numpy/scipy/matplotlib/pandas/sklearn
# collect their own data files, so nothing real is lost.
for module_name in ("sklearn.utils", "scipy.spatial", "scipy.stats"):
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
        # --- Size, not correctness -------------------------------------
        # None of these is imported anywhere in the project (checked by
        # grepping every module: numpy, scipy, matplotlib, pandas, sklearn
        # and psutil are what the code actually uses). They arrived anyway
        # because they sit in site-packages and the analysis walks
        # importable modules. The first build shipped 1.49 GB, of which:
        #   torch          292 MB
        #   _polars_runtime 168 MB
        #   cv2            113 MB
        #   bitsandbytes    86 MB   (four CUDA DLLs, none of which can load
        #                                on a machine without a GPU)
        #   pyarrow          21 MB
        # Excluding them costs nothing at runtime and roughly halves the
        # installer.
        "torch",
        "torchvision",
        "torchaudio",
        "torchgen",
        "functorch",
        "cv2",
        "polars",
        "_polars_runtime_32",
        "_polars_runtime_64",
        "bitsandbytes",
        "pyarrow",
        "tensorflow",
        "jax",
        "jaxlib",
        "numba",
        "llvmlite",
        "h5py",
        "pyarrow.lib",
        "nvidia",
        "triton",
        "onnxruntime",
        "optuna",
        "ray",
    ],
    # NOTE: cipher= / win_no_prefer_redirects= / win_private_assemblies= were
    # 5.x-only Analysis kwargs and were REMOVED in PyInstaller 6.0, which
    # build_exe.py pins. Passing them here made the spec fail to load at all
    # -- and because build_exe.py installs PyInstaller itself, the error
    # arrived only on a machine that had never built before.
)

# =============================================================================
# Windows version resource
# =============================================================================
# Without it the exe reports 0.0.0.0 in its file properties and "Unknown" in
# the task manager. Kept as a separate .txt because PyInstaller parses the
# numeric version fields with int().
VERSION_FILE = PROJECT_ROOT / "packaging" / "version_info.txt"
version_file = str(VERSION_FILE) if VERSION_FILE.exists() else None

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
    version=version_file,
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
