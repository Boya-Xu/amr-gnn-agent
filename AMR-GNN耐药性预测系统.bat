@echo off
chcp 65001 >nul
title AMR-GNN 耐药性预测系统 演示启动器
cls
echo.
echo ==================================================
echo    🧬 AMR-GNN 细菌耐药性预测智能系统
echo    一键启动演示版
echo ==================================================
echo.

:: ========== 1. 环境检查 ==========
echo [1/5] 环境检查中...
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo 首次运行，正在安装依赖管理器...
    powershell -Command "irm https://astral.sh/uv/install.ps1 | iex" >nul
    echo 安装完成，正在继续...
)

:: ========== 2. 安装所有依赖 ==========
echo [2/5] 检查项目依赖...
uv sync >nul
cd frontend
uv sync >nul
cd ..

:: ========== 3. 启动后端服务 ==========
echo [3/5] 启动后端核心服务...
start "🔧 后端服务（请勿关闭）" cmd /k "uv run uvicorn app:app --host 0.0.0.0 --port 8000"

:: 等待后端启动完成
timeout /t 8 /nobreak >nul

:: ========== 4. 自动释放8502端口（已修正，无报错） ==========
echo [4/5] 清理前端端口...
powershell -Command "$process_id = (netstat -ano | findstr :8502 | findstr LISTENING | ForEach-Object { $_.Split()[-1] }); if ($process_id) { taskkill /F /PID $process_id | Out-Null; Write-Host '✅ 已释放8502端口' } else { Write-Host '✅ 8502端口空闲' }"

:: 启动前端服务（固定用8502端口）
echo 启动前端演示界面...
start "🎨 前端服务（请勿关闭）" cmd /k "cd frontend && uv run streamlit run frontend.py --server.headless true --server.port 8502"

:: 等待前端启动完成
timeout /t 12 /nobreak >nul

:: ========== 5. 只打开前端演示页面 ==========
echo [5/5] 启动完成，正在打开演示界面...
echo.
echo ✅ 系统启动完成！
echo 📍 演示地址：http://localhost:8502
echo ⚠️  请勿关闭上面的两个服务窗口！
echo ==================================================
echo.

:: 只打开前端页面，不打开API文档
start http://localhost:8502

pause