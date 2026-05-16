from fastapi import FastAPI, UploadFile, File
import tempfile
import shutil
import time
import os

app = FastAPI()

# 动态获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXTRACTED_UNITIGS = os.path.join(DATA_DIR, "extracted_unitigs")
SNPS_ADJ_MATRIX = os.path.join(DATA_DIR, "snps_adj_matrix.csv")
FCGR_ADJ_MATRIX = os.path.join(DATA_DIR, "fcgr_adj_matrix.csv")

@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...)):
    # 保存上传的文件
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".fasta")
    shutil.copyfileobj(file.file, tmp)
    tmp.close()
    time.sleep(3)  # 模拟处理
    # 返回相对路径
    return {
        "feature_file": "./data/extracted_unitigs",
        "snps_adj_matrix": "./data/snps_adj_matrix.csv",
        "fcgr_adj_matrix": "./data/fcgr_adj_matrix.csv"
    }