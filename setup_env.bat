@echo off
chcp 65001 >nul
REM 创建 Model core 统一环境（版本见 requirements.txt）。路径需与 python_env.txt 第一行一致。
setlocal
set "HERE=%~dp0"
set "VENV=E:\venvs\model_core"
if not "%~1"=="" set "VENV=%~1"
where python >nul 2>nul || (echo [ERR] 找不到 python，请先安装 Python 3.11 & pause & exit /b 1)
python -m venv "%VENV%" || (pause & exit /b 1)
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV%\Scripts\python.exe" -m pip install -r "%HERE%requirements.txt" || (pause & exit /b 1)
echo.
echo [OK] 环境已就绪：%VENV%
echo      若路径不是 E:\venvs\model_core，请把它写进 python_env.txt
pause
