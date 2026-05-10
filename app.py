# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sys
from pathlib import Path
import traceback

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent))

from src.predict import get_prediction

app = FastAPI(title="耐药性预测API", description="AMR-GNN 预测服务")

class PredictRequest(BaseModel):
    feature_path: str
    antibiotic: str
    isolate_ids: list = None

class PredictResponse(BaseModel):
    y_proba: list
    y_pred: list
    isolate_ids: list = None

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        print(f"收到预测请求: feature_path={request.feature_path}, antibiotic={request.antibiotic}")
        result = get_prediction(
            feature_path=request.feature_path,
            antibiotic=request.antibiotic,
            isolate_ids=request.isolate_ids
        )
        print("预测成功")
        return PredictResponse(**result)
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print("=" * 50)
        print("预测失败，错误详情：")
        traceback.print_exc()
        print("=" * 50)
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)