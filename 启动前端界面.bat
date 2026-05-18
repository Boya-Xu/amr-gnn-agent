@echo off
chcp 65001 >nul
title AMR-GNN前端界面
echo ======================================
echo AMR-GNN 前端界面启动中...
echo ======================================
echo.

:: 进入前端目录
cd /d %~dp0\frontend

:: 安装前端依赖
echo 检查前端依赖...
uv sync

:: 启动前端
echo 前端启动成功！
echo 前端地址：http://localhost:8502
echo ======================================
echo.

:: 自动打开前端界面（只开这一个）
start http://localhost:8502

:: 加--server.headless true，禁止streamlit自动开浏览器
uv run streamlit run frontend.py --server.headless true

pause