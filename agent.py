import os
import requests
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
        response = requests.post(url, json={"file_path": file_path})
        if response.status_code == 200:
            return f"预处理成功，特征文件已就绪。详细信息: {response.text}"
        return f"预处理失败，状态码: {response.status_code}"
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
        response = requests.post(url, json={"antibiotic": antibiotic})
        if response.status_code == 200:
            return f"预测完成。结果数据: {response.json()}"
        return f"预测失败，状态码: {response.status_code}"
    except Exception as e:
        return f"调用预测接口异常: {str(e)}"

tools = [preprocess_fasta_tool, predict_amr_tool]

# ================= 3. 定义 Agent 大脑 =================
system_prompt = """
你是一个专业的基因组抗菌药物耐药性 (AMR) 分析助手。
你的工作流程如下：
1. 当用户提供文件路径并要求分析时，你必须先调用 preprocess_fasta_tool。
2. 只有在预处理成功后，你才能调用 predict_amr_tool 进行耐药性预测。
3. 获取预测结果后，请用清晰、专业的中文总结结果（包括耐药概率和最终判定）。
"""

# LangChain 1.x 使用 create_agent，参数为 model, tools, system_prompt
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

# 封装一个对外的函数
def chat_with_agent(user_message: str) -> str:
    try:
        # create_agent 返回的是 CompiledStateGraph，用 invoke
        response = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
        # 提取最后一条 AI 消息
        messages = response.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                return last_message.content
            return str(last_message)
        return "Agent 未返回有效响应"
    except Exception as e:
        return f"Agent 思考过程中出现错误: {str(e)}"