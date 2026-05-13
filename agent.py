import os
import requests
import json
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

# ================= 1. 配置大模型 =================
os.environ["OPENAI_API_KEY"] = "sk-0d8675768a7e43bf800d5c11ff4a3668"
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"

llm = ChatOpenAI(model="deepseek-chat", temperature=0)

# ================= 2. 定义工具 (Tools) =================
@tool
def preprocess_fasta_tool(file_path: str) -> str:
    """
    当用户上传了基因组序列文件(.fasta)，要求进行预处理或提取特征时，必须先调用此工具。
    输入参数 file_path 是前端上传文件的相对路径。
    """
    url = "https://brick-glitch-sculptor.ngrok-free.dev/preprocess"
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path, f, 'application/octet-stream')}
            response = requests.post(url, files=files)
        if response.status_code == 200:
            return f"预处理成功，特征文件已就绪。详细信息: {response.text}"
        return f"预处理失败，状态码: {response.status_code}, 详情: {response.text}"
    except FileNotFoundError:
        return f"文件未找到: {file_path}，请确认文件路径是否正确"
    except Exception as e:
        return f"调用预处理接口异常: {str(e)}"

@tool
def predict_amr_tool(antibiotic: str) -> str:
    """
    当用户询问关于某种抗生素（如 'vancomycin' 或 '万古霉素'）的耐药性预测结果时调用此工具。
    注意：必须在预处理成功后才能调用此工具。
    """
    url = "http://127.0.0.1:8000/predict"
    try:
        response = requests.post(url, json={
            "feature_path": "./data/extracted_unitigs",
            "antibiotic": antibiotic
        })
        if response.status_code == 200:
            return response.text
        return f"预测失败，状态码: {response.status_code}, 详情: {response.text}"
    except Exception as e:
        return f"调用预测接口异常: {str(e)}"

@tool
def explain_amr_tool(feature_path: str = "./data/extracted_unitigs", antibiotic: str = "vancomycin") -> str:
    """
    当用户要求解释耐药性原因、想看热力图或IG分数时调用此工具。
    必须在预测完成后才能调用。
    """
    url = "http://127.0.0.1:8000/explain"
    try:
        response = requests.post(url, json={
            "feature_path": feature_path,
            "antibiotic": antibiotic
        })
        if response.status_code == 200:
            return response.text
        return f"解释分析失败，状态码: {response.status_code}, 详情: {response.text}"
    except Exception as e:
        return f"调用解释接口异常: {str(e)}"

tools = [preprocess_fasta_tool, predict_amr_tool, explain_amr_tool]

# ================= 3. 定义 Agent 大脑 =================
system_prompt = """
你是一个专业的基因组抗菌药物耐药性 (AMR) 分析助手。
你的工作流程如下：
1. 当用户提供文件路径并要求分析时，你必须先调用 preprocess_fasta_tool。
2. 只有在预处理成功后，你才能调用 predict_amr_tool 进行耐药性预测。
3. 获取预测结果后，请用清晰、专业的中文总结结果（包括耐药概率和最终判定）。
4. 如果用户想看热力图或解释耐药原因，调用 explain_amr_tool。
"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

def chat_with_agent(user_message: str) -> dict:
    try:
        response = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
        messages = response.get("messages", [])
        reply = ""
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                reply = last_message.content
            else:
                reply = str(last_message)

        chart_data = None
        for msg in messages:
            if hasattr(msg, 'name') and msg.name == "predict_amr_tool":
                try:
                    chart_data = json.loads(msg.content)
                except Exception:
                    pass

        explain_data = None
        for msg in messages:
            if hasattr(msg, 'name') and msg.name == "explain_amr_tool":
                try:
                    explain_data = json.loads(msg.content)
                except Exception:
                    pass

        return {"reply": reply, "chart_data": chart_data, "explain_data": explain_data}
    except Exception as e:
        return {"reply": f"Agent 思考过程中出现错误: {str(e)}", "chart_data": None, "explain_data": None}