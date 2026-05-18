@echo off
chcp 65001 >nul
title AMR-GNN后端服务
echo ======================================
echo AMR-GNN 后端API服务启动中...
echo ======================================
echo.

:: 检查uv是否安装
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [1/4] 首次运行，正在安装uv包管理器...
    powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"
    echo.
    echo uv安装完成！请关闭本窗口，重新双击运行
    pause
    exit
)

:: 创建虚拟环境（如果不存在）
if not exist .venv (
    echo [2/4] 正在创建虚拟环境...
    uv venv
)

:: 安装依赖
echo [3/4] 检查并安装项目依赖...
uv sync

:: 启动服务
echo.
echo [4/4] 后端服务启动成功！
echo API文档地址：http://localhost:8000/docs
echo ======================================
echo.

:: 自动打开API文档
start http://localhost:8000/docs

:: 启动API服务（你验证过的正常命令）
uv run python src/api.py

pause