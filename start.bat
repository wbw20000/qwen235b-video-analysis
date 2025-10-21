@echo off
echo ============================================================
echo Qwen3-VL 视频分析服务启动脚本（虚拟环境版本）
echo ============================================================
echo.

REM 检查虚拟环境是否存在
if not exist "venv\Scripts\python.exe" (
    echo 错误: 虚拟环境不存在！
    echo 请先运行以下命令创建虚拟环境:
    echo   python -m venv venv
    echo   venv\Scripts\pip install flask openai
    pause
    exit /b 1
)

echo [1/2] 激活虚拟环境...
call venv\Scripts\activate.bat

echo.
echo [2/2] 启动 Flask 应用...
echo.
echo 应用将在 http://localhost:5000 运行
echo 按 Ctrl+C 可以停止服务
echo.
echo ============================================================
echo.

REM 启动应用
python app.py

pause
