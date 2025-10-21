@echo off
echo ========================================
echo Qwen3-VL 视频分析服务启动脚本
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo [1/3] 检查依赖...
pip show flask >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在安装依赖...
    pip install -r requirements.txt
)

echo.
echo [2/3] 创建必要的目录...
if not exist "uploads" mkdir uploads
if not exist "templates" mkdir templates

echo.
echo [3/3] 启动 Flask 应用...
echo.
echo 应用将在 http://localhost:5000 运行
echo 按 Ctrl+C 可以停止服务
echo.
echo ========================================
echo.

python app.py

pause
