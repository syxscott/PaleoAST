@echo off
chcp 65001 >nul
echo ============================================
echo   PaleoAST 打包工具
echo ============================================
echo.
echo 正在激活 conda 环境 'past'...

:: 检查 conda 是否可用
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到 conda，请确保已安装 Anaconda
    pause
    exit /b 1
)

:: 激活 conda 环境并运行打包脚本
call conda activate past
if %errorlevel% neq 0 (
    echo 错误: 无法激活 'past' 环境
    echo 请确保已创建该环境: conda create -n past python=3.11
    pause
    exit /b 1
)

echo 环境激活成功
echo.
echo 开始打包...
echo.

python "%~dp0build_exe.py"

if %errorlevel% neq 0 (
    echo.
    echo 打包失败!
    pause
    exit /b 1
)

echo.
echo ============================================
echo   打包完成！
echo ============================================
echo.
echo 输出目录: %~dp0dist\PaleoAST
echo.
echo 按任意键打开输出目录...
pause >nul
explorer "%~dp0dist"
