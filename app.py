from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import warnings
from agent import chat_with_agent

# 抑制pkg_resources过期警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

# 修正导入路径：从src.predict导入
from src.predict import get_prediction
from src.explain_api import get_explanation

# 定义前端传过来的数据格式
class ChatRequest(BaseModel):
    message: str

# 初始化FastAPI应用
app = FastAPI(
    title="AMR-GNN细菌耐药性预测API",
    version="1.0.0",
    description="基于多表示图神经网络的抗微生物药物耐药性预测服务，支持铜绿假单胞菌、大肠杆菌等多种病原体"
)

# 严格的请求体模型
class PredictRequest(BaseModel):
    feature_path: str = "./data/extracted_unitigs"
    antibiotic: str = "vancomycin"

# 严格的响应体模型（明确每个字段的类型）
class PredictResponse(BaseModel):
    y_proba: List[List[float]]  # 二维数组：[[敏感概率, 耐药概率], ...]
    y_pred: List[int]           # 预测标签：0=敏感，1=耐药
    isolate_ids: List[str]      # 菌株ID列表

# 解释接口的请求体模型
class ExplainRequest(BaseModel):
    feature_path: str = "./data/extracted_unitigs"
    antibiotic: str = "vancomycin"
    isolate_ids: list = None
    n_steps: int = 20

# 解释接口的响应体模型
class ExplainResponse(BaseModel):
    isolate_ids: list
    attributions: list
    attribution_shape: list
    error: str = None
    metadata: dict = None

# 健康检查接口
@app.get("/health", summary="服务健康检查", tags=["系统接口"])
async def health_check():
    return {"status": "ok", "message": "AMR-GNN服务运行正常"}

# 核心预测接口
@app.post("/predict", summary="耐药性预测", tags=["核心功能"], response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        # 调用预测函数
        result = get_prediction(
            feature_path=request.feature_path,
            antibiotic=request.antibiotic
        )
        return result
    except Exception as e:
        # 统一异常处理
        raise HTTPException(
            status_code=500,
            detail=f"预测失败: {str(e)}"
        )

# 特征重要性解释接口（热力图）
@app.post("/explain", summary="特征重要性解释（热力图）", tags=["核心功能"], response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    print(f"[DEBUG] 收到请求，feature_path: {request.feature_path}, antibiotic: {request.antibiotic}, isolate_ids: {request.isolate_ids}")
    try:
        result = get_explanation(
            feature_path=request.feature_path,
            antibiotic=request.antibiotic,
            isolate_ids=request.isolate_ids,
            n_steps=request.n_steps
        )
        print(f"[DEBUG] 返回结果 isolate_ids: {result.get('isolate_ids')}, error: {result.get('error', 'N/A')}")
        return result
    except Exception as e:
        print(f"[ERROR] 解释接口异常: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"解释失败: {str(e)}"
        )


@app.post("/api/chat")
async def agent_chat_endpoint(request: ChatRequest):
    try:
        agent_result = chat_with_agent(request.message)
        return {
            "status": "success",
            "reply": agent_result["reply"],
            "chart_data": agent_result["chart_data"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent 运行出错: {str(e)}"
        )

# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)