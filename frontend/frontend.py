"""
AMR-GNN 耐药性预测智能助手 - 前端界面（适配 Agent 同步接口）
修复：热力图稳定key、错误不中断、文件清理、表格key等
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import uuid
import urllib3
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================== 全局配置 ==================
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Agent 统一接口（同学 A 部署的 FastAPI）
AGENT_API_URL = "http://localhost:8000/api/chat"

# ================== 页面基础设置 ==================
st.set_page_config(
    page_title="AMR-GNN 智能体",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 AMR-GNN 耐药性预测智能助手")
st.caption("⚡️ 上传基因组 .fasta 文件，用自然语言提问，获得耐药性分析与解释")

# ================== 初始化会话状态 ==================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ================== 工具函数 ==================
def clear_uploaded_files():
    """清空本地保存的上传文件"""
    if st.session_state.uploaded_files:
        for f in st.session_state.uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, f.name)
            if os.path.exists(file_path):
                os.remove(file_path)
    st.session_state.uploaded_files = None
    st.session_state.uploader_key += 1

def plot_ig_heatmap(heatmap_data: dict, msg_index: int):
    """绘制IG归因热力图，使用消息索引作为稳定key"""
    unitigs = heatmap_data.get("unitigs", [])
    scores = heatmap_data.get("scores", [])
    isolate_id = heatmap_data.get("isolate_id", "未知菌株")

    if not unitigs or not scores:
        st.warning("热力图数据为空")
        return

    sorted_pairs = sorted(zip(unitigs, scores), key=lambda x: x[1], reverse=True)[:20]
    top_unitigs, top_scores = zip(*sorted_pairs)

    fig = px.bar(
        x=list(top_unitigs),
        y=list(top_scores),
        labels={"x": "特征", "y": "IG 归因分数"},
        title=f"🔬 对耐药性预测贡献最大的特征（{isolate_id}）",
        color=list(top_scores),
        color_continuous_scale="Tealgrn"
    )
    fig.add_hline(
        y=0.5, line_dash="dot", line_color="grey",
        annotation_text="通常关注阈值", annotation_position="top right"
    )
    fig.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
    st.plotly_chart(fig, width='stretch', key=f"heatmap_{msg_index}")

def convert_predict_response_to_table(predict_result: dict):
    """
    将同学 B 的预测 API 返回格式转换为前端表格数据。
    输入示例：{"y_proba": [[0.1,0.9],...], "y_pred": [1,0,...], "isolate_ids": ["ID1",...]}
    假设二分类：y_proba[i][0] 是敏感概率，y_proba[i][1] 是耐药概率；y_pred: 1=耐药,0=敏感
    """
    rows = []
    resistant_count = 0
    isolate_ids = predict_result.get("isolate_ids", [])
    y_proba = predict_result.get("y_proba", [])
    y_pred = predict_result.get("y_pred", [])

    for i, iso_id in enumerate(isolate_ids):
        prob = y_proba[i][1] if i < len(y_proba) and len(y_proba[i]) > 1 else 0.0
        pred_label = "耐药" if (i < len(y_pred) and y_pred[i] == 1) else "敏感"
        if pred_label == "耐药":
            resistant_count += 1
        rows.append([iso_id, prob, pred_label])

    total = len(rows)
    ratio = resistant_count / total if total > 0 else 0.0
    summary = f"（共{total}株，耐药{resistant_count}株，耐药率{ratio:.1%}）"
    return rows, summary

def convert_explain_response_to_heatmap(explain_result: dict):
    """
    尝试将解释结果转换为热力图所需格式。
    假设返回包含 'attributions' 和 'isolate_ids'。
    """
    isolate_ids = explain_result.get("isolate_ids", [])
    attributions = explain_result.get("attributions", [])
    if isinstance(attributions, list) and len(attributions) > 0:
        # 取第一个菌株的归因分数
        scores = attributions[0]
        if isinstance(scores, list):
            unitigs = [f"特征{j}" for j in range(len(scores))]
            return {
                "unitigs": unitigs,
                "scores": scores,
                "isolate_id": isolate_ids[0] if isolate_ids else "未知"
            }
    return None

# ================== 侧边栏：文件上传 + 控制 ==================
with st.sidebar:
    st.header("📁 上传基因组文件")
    st.markdown("支持 **.fasta** 格式，可一次上传多个文件")

    uploaded = st.file_uploader(
        "选择文件",
        type=["fasta", "fa", "fna"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}",
        help="上传您要分析的菌株基因组组装文件"
    )

    if uploaded:
        st.session_state.uploaded_files = uploaded
        for f in uploaded:
            save_path = os.path.join(UPLOAD_DIR, f.name)
            with open(save_path, "wb") as out_file:
                out_file.write(f.getbuffer())
        st.success(f"✅ 已上传并保存 {len(uploaded)} 个文件")

        file_df = pd.DataFrame(
            [(f.name, f"{f.size/1024:.1f} KB") for f in uploaded],
            columns=["文件名", "大小"]
        )
        st.dataframe(file_df, hide_index=True, width='stretch')

        if st.button("🗑️ 清空已上传的文件"):
            clear_uploaded_files()
            st.rerun()
    else:
        st.info("👆 请上传 .fasta 文件开始分析")

    st.divider()
    st.caption("💡 示例提问：")
    st.caption("“请预测这些菌株对万古霉素的耐药性”")
    st.caption("“为什么第一株菌是耐药的？”")

    st.divider()
    if st.session_state.messages:
        if st.button("🧹 清空对话记录"):
            st.session_state.messages = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.rerun()

# ================== 主区域：聊天历史 ==================
if not st.session_state.messages:
    st.markdown("""
    ## 👋 欢迎使用 AMR-GNN 智能助手
    在这里，您可以通过**自然语言**对细菌基因组进行耐药性分析。

    **快速开始：**
    1. 在左侧上传 `.fasta` 基因组文件
    2. 在底部对话框输入您的问题  
    3. 智能体将自动完成特征提取、模型预测和结果解释

    *我们的模型基于 AMR-GNN 图神经网络，支持对多种抗生素的耐药性预测。*
    """)
else:
    for idx, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        with st.chat_message(role):
            if msg.get("content"):
                st.markdown(msg["content"])

            # 渲染表格（使用稳定 key）
            if msg.get("table") and msg["table"].get("data"):
                df = pd.DataFrame(msg["table"]["data"], columns=msg["table"]["columns"])
                if "耐药概率" in df.columns:
                    styled = df.style.background_gradient(
                        subset=["耐药概率"], cmap="RdYlGn", vmin=0, vmax=1
                    ).format({"耐药概率": "{:.2%}"})
                    st.dataframe(styled, hide_index=True, width='stretch', key=f"table_{idx}")
                else:
                    st.dataframe(df, hide_index=True, width='stretch', key=f"table_{idx}")

            if msg.get("heatmap"):
                plot_ig_heatmap(msg["heatmap"], msg_index=idx)

# ================== 用户输入处理 ==================
if prompt := st.chat_input("💬 输入您的问题，例如“预测这些菌株对万古霉素的耐药性”..."):

    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 准备助手回复数据结构
    data = {"reply": "", "table": None, "heatmap": None}

    with st.chat_message("assistant"):
        try:
            # 构造消息：将上传文件路径嵌入文本（因为 Agent 接口只接受 message）
            final_message = prompt
            if st.session_state.uploaded_files:
                file_paths = [os.path.join(UPLOAD_DIR, f.name).replace("\\", "/")
                              for f in st.session_state.uploaded_files]
                # 用明确的上下文告诉 Agent 文件在哪
                file_context = "已上传文件路径：" + ", ".join(file_paths)
                final_message = f"{file_context}\n用户问题：{prompt}"

            with st.spinner("🤖 智能体正在分析您的问题（可能需要几分钟）..."):
                resp = requests.post(
                    AGENT_API_URL,
                    json={"message": final_message},
                    timeout=300  # 长时间等待
                )

            if resp.status_code != 200:
                raise Exception(f"Agent 返回错误状态码 {resp.status_code}：{resp.text}")

            result = resp.json()

            # 检查外层 status（同学 A 的接口包装）
            if result.get("status") != "success":
                raise Exception(f"Agent 处理失败：{result}")

            reply = result.get("reply", "（未获取到回复）")
            st.markdown(reply)
            data["reply"] = reply

            # ---- 解析表格：转换 B 的预测结果为前端格式 ----
            table_data = None
            chart_data = result.get("chart_data")
            if isinstance(chart_data, dict):
                # 尝试转换 B 的原始格式
                if "y_proba" in chart_data and "isolate_ids" in chart_data:
                    rows, summary = convert_predict_response_to_table(chart_data)
                    if rows:
                        st.caption(summary)
                        df = pd.DataFrame(rows, columns=["菌株编号", "耐药概率", "预测结果"])
                        styled = df.style.background_gradient(
                            subset=["耐药概率"], cmap="RdYlGn", vmin=0, vmax=1
                        ).format({"耐药概率": "{:.2%}"})
                        st.dataframe(styled, hide_index=True, width='stretch')
                        table_data = {
                            "columns": ["菌株编号", "耐药概率", "预测结果"],
                            "data": rows
                        }
                # 兼容可能的前端标准格式（若某天同学 A 改了）
                elif "results" in chart_data:
                    results = chart_data["results"]
                    if isinstance(results, list) and results:
                        rows = []
                        resistant_count = 0
                        for item in results:
                            if isinstance(item, dict):
                                iso_id = item.get("isolate_id", "未知")
                                prob = item.get("resistant_probability", 0)
                                pred = item.get("prediction", "未知")
                                if pred == "耐药":
                                    resistant_count += 1
                                rows.append([iso_id, prob, pred])
                        if rows:
                            total = len(rows)
                            ratio = resistant_count / total if total > 0 else 0.0
                            st.caption(f"（共{total}株，耐药{resistant_count}株，耐药率{ratio:.1%}）")
                            df = pd.DataFrame(rows, columns=["菌株编号", "耐药概率", "预测结果"])
                            styled = df.style.background_gradient(
                                subset=["耐药概率"], cmap="RdYlGn", vmin=0, vmax=1
                            ).format({"耐药概率": "{:.2%}"})
                            st.dataframe(styled, hide_index=True, width='stretch')
                            table_data = {
                                "columns": ["菌株编号", "耐药概率", "预测结果"],
                                "data": rows
                            }
            data["table"] = table_data

            # ---- 解析热力图 ----
            heatmap_data = None
            explain_data = result.get("explain_data")
            if isinstance(explain_data, dict):
                hm = convert_explain_response_to_heatmap(explain_data)
                if hm:
                    plot_ig_heatmap(hm, msg_index=len(st.session_state.messages))
                    heatmap_data = hm
            data["heatmap"] = heatmap_data

        except requests.exceptions.Timeout:
            error_msg = "❌ 请求超时，智能体处理时间过长，请稍后重试。"
            st.error(error_msg)
            data["reply"] = error_msg
        except Exception as e:
            error_msg = f"❌ 调用失败：{str(e)}"
            st.error(error_msg)
            data["reply"] = error_msg

    # 将助手回复存入历史（包含可能的错误消息）
    st.session_state.messages.append({
        "role": "assistant",
        "content": data.get("reply", ""),
        "table": data.get("table"),
        "heatmap": data.get("heatmap")
    })