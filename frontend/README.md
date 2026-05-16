# AMR-GNN 耐药性预测智能助手 - 前端界面

基于 Streamlit 构建的细菌抗生素耐药性（AMR）预测智能助手，提供对话式交互界面。  
用户可上传 `.fasta` 基因组文件，通过自然语言请求耐药性预测或耐药原因解释，系统自动调用后端 Agent 统一接口，并可视化展示预测结果表格与热力图。

## 功能概览

- ✅ 聊天式对话界面（类 ChatGPT）
- ✅ 支持拖拽上传 `.fasta` / `.fa` 基因组文件
- ✅ 真实模式：通过 Agent 统一接口执行预测与解释
- ✅ 预测结果自动生成表格（含耐药概率渐变着色）
- ✅ 耐药原因解释时自动生成热力图（特征重要性柱状图）
- ✅ 历史记录保存、清空对话、清空文件
- ✅ 错误自动捕获，不中断界面

## 系统架构
用户浏览器
│
▼
Streamlit 前端 (localhost:8501)
│ POST /api/chat
▼
Agent 统一接口 (FastAPI, localhost:8000)
│
├─ 预处理工具 ──► 模拟预处理服务 (localhost:8001)
├─ 预测工具 ──► 预测服务 (localhost:8002)
└─ 解释工具 ──► 解释服务 (Agent 内部 / 预测服务)


> 前端仅与 Agent 通信，不直接调用预处理、预测、解释服务。

## 环境准备

- Python 3.10+ （推荐 3.11）
- [uv](https://docs.astral.sh/uv/) 包管理器

## 快速开始

### 1. 安装依赖

```bash
uv sync

2. 启动前端
uv run streamlit run frontend.py

3. 连接真实后端
需要以下服务均已启动：
服务	                  	默认地址	                	启动命令示例
模拟预处理		http://localhost:8001	uv run uvicorn fake_preprocess_api:app --host 0.0.0.0 --port 8001
预测服务	    http://localhost:8002	cd amr-gnn-agent && uv run uvicorn src.api:app --host 0.0.0.0 --port 8002
Agent 统一接口	http://localhost:8000	cd amr-gnn-agent && uv run uvicorn app:app --host 0.0.0.0 --port 8000
前端默认调用 http://localhost:8000/api/chat，可在 frontend.py 顶部 AGENT_API_URL 修改。

Agent 接口约定
前端通过 POST /api/chat 与 Agent 通信。

请求格式：
json
{
  "message": "已上传文件路径：E:/.../uploads/xxx.fasta\n用户问题：请预测万古霉素耐药性"
}

返回格式：
json
{
  "status": "success",
  "reply": "Markdown 格式的中文回复（直接展示在对话气泡中）",
  "chart_data": {
    "y_proba": [[0.05, 0.95], ...],
    "y_pred": [1, 0, ...],
    "isolate_ids": ["1352.10008", ...]
  },
  "explain_data": {
    "isolate_ids": ["1352.10008"],
    "attributions": [[0.63, 0.02, ...], ...],
    "attribution_shape": [1, 512]
  }
}
chart_data：预测结果表格，y_proba[i][1] 为耐药概率，y_pred[i]=1 为耐药。
explain_data：解释数据，attributions[i][j] 为第 i 个菌株第 j 个特征的 IG 分数。
两个字段在未触发对应功能时均为 null。

前端配置
AGENT_API_URL：Agent 接口地址，默认为 http://localhost:8000/api/chat
UPLOAD_DIR：上传文件保存目录，默认为 ./uploads

常见问题
Q：为什么表格/热力图不显示？
A：检查后端返回的 chart_data / explain_data 是否为 null。若为 null，请确认后端工具已正确返回数据；前端会安全跳过，不会报错。

Q：请求超时怎么办？
A：前端默认超时 300 秒，可在 frontend.py 中修改 timeout 参数。若 Agent 处理时间过长，请检查后端工具响应速度，或调整超时阈值。

Q：如何修改 Agent 地址？
A：编辑 frontend.py 顶部的 AGENT_API_URL = "http://localhost:8000/api/chat" 即可。