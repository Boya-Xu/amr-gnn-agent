# 解决ModuleNotFoundError问题：添加项目根目录到Python路径
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# 直接导入你已经写好的预测函数
from src.predict import get_prediction

app = FastAPI(title="AMR-GNN抗生素耐药性预测API", version="1.0")


# 请求模型
class PredictionRequest(BaseModel):
    feature_path: str = "./data/extracted_unitigs"
    antibiotic: str
    isolate_ids: Optional[List[str]] = None


# 响应模型
class PredictionResult(BaseModel):
    isolate_id: str
    sensitive_probability: float
    resistant_probability: float
    prediction: str  # "敏感" 或 "耐药"


class PredictionResponse(BaseModel):
    success: bool
    total_count: int
    results: List[PredictionResult]
    statistics: dict


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_antibiotic_resistance(request: PredictionRequest):
    """
    抗生素耐药性预测接口
    - **feature_path**: 特征文件夹路径，默认使用./data/extracted_unitigs
    - **antibiotic**: 抗生素名称，如"vancomycin"
    - **isolate_ids**: 可选，指定要预测的菌株ID列表
    """
    try:
        # 直接调用同步预测函数（FastAPI会自动在线程池中运行）
        result = get_prediction(
            feature_path=request.feature_path,
            antibiotic=request.antibiotic,
            isolate_ids=request.isolate_ids
        )

        # 格式化结果
        formatted_results = []
        for i in range(len(result["isolate_ids"])):
            formatted_results.append(PredictionResult(
                isolate_id=result["isolate_ids"][i],
                sensitive_probability=round(result["y_proba"][i][0], 4),
                resistant_probability=round(result["y_proba"][i][1], 4),
                prediction="耐药" if result["y_pred"][i] == 1 else "敏感"
            ))

        # 统计信息
        resistant_count = sum(result["y_pred"])
        sensitive_count = len(result["y_pred"]) - resistant_count

        return PredictionResponse(
            success=True,
            total_count=len(formatted_results),
            results=formatted_results,
            statistics={
                "resistant_count": resistant_count,
                "sensitive_count": sensitive_count,
                "resistant_ratio": round(resistant_count / len(result["y_pred"]) * 100, 1),
                "sensitive_ratio": round(sensitive_count / len(result["y_pred"]) * 100, 1)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.get("/")
async def root():
    return {"message": "AMR-GNN抗生素耐药性预测API服务已启动", "docs": "/docs"}


if __name__ == "__main__":
    # 直接运行API服务
    uvicorn.run(app, host="0.0.0.0", port=8000)