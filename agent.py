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
    预测指定抗生素的耐药性。仅用于预测，不解释原因。
    参数 antibiotic: 抗生素名称，如 'vancomycin'、'penicillin'。
    注意：用户问“预测”、“耐药性”时使用此工具。用户问“为什么”、“解释”、“热力图”时绝对不能使用！
    """
    url = "http://127.0.0.1:8000/predict"
    try:
        response = requests.post(url, json={
            "feature_path": "./data/extracted_unitigs",
            "antibiotic": antibiotic
        }, timeout=30)
        if response.status_code == 200:
            return response.text
        return f"预测失败，状态码: {response.status_code}, 详情: {response.text}"
    except Exception as e:
        return f"调用预测接口异常: {str(e)}"


@tool
def explain_amr_tool(antibiotic: str, isolate_ids: list) -> str:
    """
    解释指定菌株耐药的原因，返回热力图数据。仅用于解释，不预测耐药性。
    参数 antibiotic: 抗生素名称，如 'vancomycin'。
    参数 isolate_ids: 需要解释的菌株ID列表，如 ['1352.10008', '1352.10011']。
    注意：用户问“为什么耐药”、“解释原因”、“热力图”、“特征重要性”时必须调用此工具！
    """
    url = "http://127.0.0.1:8000/explain"
    try:
        response = requests.post(url, json={
            "feature_path": "./data/extracted_unitigs",
            "antibiotic": antibiotic,
            "isolate_ids": isolate_ids,
            "n_steps": 20
        }, timeout=(60,600))
        if response.status_code == 200:
            return response.text
        return f"解释分析失败，状态码: {response.status_code}, 详情: {response.text}"
    except Exception as e:
        return f"调用解释接口异常: {str(e)}"


tools = [preprocess_fasta_tool, predict_amr_tool]

# ================= 3. 定义 Agent 大脑 =================
system_prompt = """
你是一个专业的基因组抗菌药物耐药性 (AMR) 分析助手。你必须严格遵守以下规则：

## 工具调用规则
1. 用户问“预测”、“耐药性” → 调用 preprocess_fasta_tool（如需要），然后调用 predict_amr_tool。
2. 用户问“为什么”、“解释”、“热力图”、“特征重要性”、“原因” → 不要调用任何工具，直接回复"好的，我正在为您生成耐药性解释和热力图数据，请稍候..."即可，后端会自动处理。
3. 用户问“分析”或“全面评估” → 调用 predict_amr_tool，然后直接回复"预测完成，解释数据正在生成中..."

## 参数传递规则
- predict_amr_tool 的 antibiotic 参数只接受抗生素名称（如 'vancomycin'），绝不能传入菌株ID。

## 输出规则
获取结果后，请用清晰、专业的中文总结。
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
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    continue
                if hasattr(msg, 'name') and msg.name:
                    continue
                reply = msg.content
                break

        chart_data = None
        explain_data = None
        for msg in messages:
            if hasattr(msg, 'name') and msg.name == "predict_amr_tool":
                try:
                    chart_data = json.loads(msg.content)
                except Exception:
                    pass

        return {"reply": reply, "chart_data": chart_data, "explain_data": explain_data}
    except Exception as e:
        return {"reply": f"Agent 思考过程中出现错误: {str(e)}", "chart_data": None, "explain_data": None}