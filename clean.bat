@echo off
chcp 65001 >nul
echo ============================================
echo   PaleoAST 构建清理工具
echo ============================================
echo.

set DIST_DIR=%~dp0dist
set BUILD_DIR=%~dp0build

echo 正在清理构建文件...

if exist "%DIST_DIR%" (
    echo 删除: %DIST_DIR%
    rmdir /s /q "%DIST_DIR%"
    echo   已删除 dist 目录
) else (
    echo   dist 目录不存在，跳过
)

if exist "%BUILD_DIR%" (
    echo 删除: %BUILD_DIR%
    rmdir /s /q "%BUILD_DIR%"
    echo   已删除 build 目录
) else (
    echo   build 目录不存在，跳过
)

echo.
echo 清理完成！
pause
