from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import warnings
from agent import chat_with_agent

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

from src.predict import get_prediction

class ChatRequest(BaseModel):
    message: str

app = FastAPI(
    title="AMR-GNN细菌耐药性预测API",
    version="1.0.0",
    description="基于多表示图神经网络的抗菌药物耐药性预测服务，支持铜绿假单胞菌、大肠杆菌等多种病原体"
)

class PredictRequest(BaseModel):
    feature_path: str = "./data/extracted_unitigs"
    antibiotic: str = "vancomycin"

class PredictResponse(BaseModel):
    y_proba: List[List[float]]
    y_pred: List[int]
    isolate_ids: List[str]

# 新增：Explain 接口的请求模型
class ExplainRequest(BaseModel):
    feature_path: str = "./data/extracted_unitigs"
    antibiotic: str = "vancomycin"

# 健康检查接口
@app.get("/health", summary="服务健康检查", tags=["系统接口"])
async def health_check():
    return {"status": "ok", "message": "AMR-GNN服务运行正常"}

# 核心预测接口
@app.post("/predict", summary="耐药性预测", tags=["核心功能"], response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        result = get_prediction(
            feature_path=request.feature_path,
            antibiotic=request.antibiotic
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

# 新增：模型解释接口
@app.post("/explain", summary="耐药性解释（IG热力图数据）", tags=["核心功能"])
async def explain(request: ExplainRequest):
    try:
        from src.explain_api import get_explanation
        result = get_explanation(
            feature_path=request.feature_path,
            antibiotic=request.antibiotic
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解释分析失败: {str(e)}")


import asyncio


@app.post("/api/chat")
async def agent_chat_endpoint(request: ChatRequest):
    explain_data = None

    # 第一步：在独立线程中调 /explain，避免阻塞事件循环
    try:
        import requests as req
        loop = asyncio.get_running_loop()
        explain_response = await loop.run_in_executor(
            None,
            lambda: req.post(
                "http://127.0.0.1:8000/explain",
                json={
                    "feature_path": "./data/extracted_unitigs",
                    "antibiotic": "vancomycin",
                    "isolate_ids": ["1352.10008"],
                    "n_steps": 20
                },
                timeout=300
            )
        )
        if explain_response.status_code == 200:
            explain_data = explain_response.json()
            print(f"✅ 成功获取 explain_data")
    except Exception as e:
        print(f"⚠️ 获取 explain_data 失败: {e}")

    # 第二步：调 Agent 获取文字回复
    try:
        agent_result = chat_with_agent(request.message)
        return {
            "status": "success",
            "reply": agent_result.get("reply", ""),
            "chart_data": agent_result.get("chart_data"),
            "explain_data": explain_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 运行出错: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)