"""
PaleoAST Build Script

使用 PyInstaller 打包 PaleoAST 为独立可执行文件

使用方法:
    conda activate <你用于开发的环境>   # 例如 dev
    python build_exe.py

构建用的解释器由 get_conda_python() 解析,可用 $PALEOAST_BUILD_PYTHON 覆盖。

或者双击运行:
    build.bat
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_conda_python():
    """Resolve the Python interpreter to build with.

    The target environment is overridable because the build used to be hardcoded
    to a conda env named ``past``, which is not the environment this project is
    developed in (``dev`` on this machine) -- so a build could silently use an
    interpreter with the wrong dependencies, or fail outright.

    Resolution order:
      1. ``$PALEOAST_BUILD_PYTHON`` (explicit, wins outright)
      2. the currently active interpreter, when it already has PyQt6 and numpy
      3. ``$CONDA_DEFAULT_ENV`` / ``$CONDA_PREFIX``'s env
      4. ``past`` then ``dev`` under the conda base, if they exist
    """
    override = os.environ.get("PALEOAST_BUILD_PYTHON")
    if override:
        candidate = Path(override)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"PALEOAST_BUILD_PYTHON points at {override!r}, which does not exist"
        )

    # Prefer whatever interpreter is running this script, provided it can
    # actually import the project's own dependencies.
    if _has_build_deps(sys.executable):
        return Path(sys.executable)

    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    prefix = os.environ.get("CONDA_PREFIX")
    candidates: list[Path] = []
    if env_name and prefix:
        candidates.append(Path(prefix) / ("python.exe" if sys.platform == "win32" else "bin/python"))
    base = Path(os.environ.get("CONDA_PREFIX") or (r"D:\Program Files\ananconda3" if sys.platform == "win32" else "/opt/anaconda3"))
    for name in ("past", "dev"):
        if sys.platform == "win32":
            candidates.append(base / "envs" / name / "python.exe")
        else:
            candidates.append(base / "envs" / name / "bin" / "python")
    for cand in candidates:
        if cand.exists() and _has_build_deps(str(cand)):
            return cand

    # Nothing verified; hand back the first candidate that merely EXISTS so the
    # error surfaces from PyInstaller with real diagnostics.
    for cand in candidates:
        if cand.exists():
            return cand
    return Path(sys.executable)


def _has_build_deps(python: str | Path) -> bool:
    """True when this interpreter can import what the app needs at runtime."""
    try:
        probe = "import PyQt6, numpy, scipy, pandas, matplotlib, sklearn"
        return subprocess.run(
            [str(python), "-c", probe], capture_output=True, text=True
        ).returncode == 0
    except (OSError, ValueError):
        return False


def check_pyinstaller():
    """检查 pyinstaller 是否安装"""
    python = get_conda_python()
    print(f"构建解释器: {python}")
    result = subprocess.run([str(python), "-m", "pip", "show", "pyinstaller"], capture_output=True, text=True)
    if result.returncode != 0:
        print("正在安装 PyInstaller...")
        # Pin >=6: PaleoAST.spec uses the 6.x Analysis() signature. The 5.x-era
        # kwargs (win_no_prefer_redirects, cipher, ...) were removed in 6.0.
        subprocess.run(
            [str(python), "-m", "pip", "install", "pyinstaller>=6"], check=True
        )
        print("PyInstaller 安装完成!")
    else:
        print("PyInstaller 已安装")


def clean_build():
    """清理之前的构建文件"""
    project_root = Path(__file__).parent
    build_dir = project_root / "build"
    dist_dir = project_root / "dist"

    print("清理旧构建文件...")

    if build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"  已删除: {build_dir}")

    if dist_dir.exists():
        shutil.rmtree(dist_dir)
        print(f"  已删除: {dist_dir}")


def run_pyinstaller():
    """运行 PyInstaller"""
    python = get_conda_python()
    project_root = Path(__file__).parent

    print("\n开始打包...")
    print(f"使用 Python: {python}")

    # 构建命令
    #
    # NOTE: no --onedir / --onefile here. Those are makespec-time options and
    # PyInstaller rejects them outright when a .spec is supplied:
    #   ERROR: option(s) not allowed: --onedir/--onefile
    #   makespec options not valid when a .spec file is given
    # The mode already lives in the spec -- EXE() is built with
    # exclude_binaries=True and handed to COLLECT(), which is onedir.
    cmd = [
        str(python),
        "-m",
        "PyInstaller",
        "--clean",  # 清理缓存
        "--noconfirm",  # 不询问确认
        f"--distpath={project_root / 'dist'}",
        f"--workpath={project_root / 'build'}",
        str(project_root / "PaleoAST.spec"),
    ]

    print(f"执行命令: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(project_root))

    if result.returncode != 0:
        print("构建失败!")
        sys.exit(1)

    print("\n构建成功!")


def create_launcher():
    """创建启动器脚本"""
    project_root = Path(__file__).parent
    dist_dir = project_root / "dist" / "PaleoAST"

    # Windows 批处理文件
    bat_content = """@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting PaleoAST...
PaleoAST.exe
"""

    bat_path = project_root / "dist" / "PaleoAST.bat"
    bat_path.write_text(bat_content, encoding="utf-8")
    print(f"已创建启动脚本: {bat_path}")

    # 创建桌面快捷方式 (Windows)
    if sys.platform == "win32":
        try:
            import pythoncom
            import win32com.client

            pythoncom.CoInitialize()

            # 创建快捷方式
            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortcut(str(project_root / "dist" / "PaleoAST.lnk"))
            shortcut.TargetPath = str(dist_dir / "PaleoAST.exe")
            shortcut.WorkingDirectory = str(dist_dir)
            shortcut.Description = "PaleoAST - Paleontological Advanced Statistical Toolkit"
            shortcut.Save()

            print(f"已创建快捷方式: {project_root / 'dist' / 'PaleoAST.lnk'}")

            pythoncom.CoUninitialize()
        except ImportError:
            print("提示: 安装 pywin32 可以创建桌面快捷方式")
            print(f"快捷方式位置: {dist_dir / 'PaleoAST.exe'}")


def print_summary():
    """打印构建摘要"""
    project_root = Path(__file__).parent
    dist_dir = project_root / "dist" / "PaleoAST"
    exe_path = dist_dir / "PaleoAST.exe"

    print("\n" + "=" * 60)
    print("PaleoAST 打包完成!")
    print("=" * 60)
    print(f"\n输出目录: {dist_dir}")
    print(f"可执行文件: {exe_path}")
    print(f"文件大小: {exe_path.stat().st_size / 1024 / 1024:.2f} MB")

    print("\n使用方法:")
    print(f"  1. 进入目录: cd {dist_dir}")
    print("  2. 运行程序: PaleoAST.exe")
    print("\n或者双击运行:")
    print(f"  {project_root / 'dist' / 'PaleoAST.bat'}")
    print("=" * 60)


def main():
    """主函数"""
    print("=" * 60)
    print("PaleoAST 打包工具")
    print("=" * 60)

    # 检查 pyinstaller
    check_pyinstaller()

    # 清理旧文件
    clean_build()

    # 运行打包
    run_pyinstaller()

    # 创建启动器
    create_launcher()

    # 打印摘要
    print_summary()


if __name__ == "__main__":
    main()
