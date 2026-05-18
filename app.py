# ============== 第一行就放路径修复，绝对不能动顺序 ==============
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ============== 后面才是所有import ==============
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

from agent import chat_with_agent
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
    try:
        # 只调 Agent，Agent 内部会调用 explain_amr_tool 获取解释数据
        agent_result = chat_with_agent(request.message)
        return {
            "status": "success",
            "reply": agent_result.get("reply", ""),
            "chart_data": agent_result.get("chart_data"),
            "explain_data": agent_result.get("explain_data")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 运行出错: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)