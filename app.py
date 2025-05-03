"""
任务:基于大语言模型的对话式数据分析Agent
以开源大模型Llama3.1:8b为核心,通过Ollama实现本地化部署,结合LangChain和LangGraph框架构建自然语言交互Agent.
支持用户通过对话形式查询和分析本地数据库，还允许用户上传自定义业务背景描述及数据集，扩展选择其他业务场景。
设计动态数据可视化模块,利用Streamlit实现前端图表渲染,支持多维度数据展示与交互式分析.
同时通过持续对话优化分析流程，实现人机协作的高效数据处理。
目前已经在Agent.py文件中使用LangGraph实现Agent的基本框架, 并在app.py文件中使用streamlit实现的前端的设计,
要求在Agent.py的基础上增加检索本地数据库的功能, 保留Tavily工具, 将数据检索功能定义为新的工具。
本地数据在项目根目录data下, data目录下包含仅txt.csv文件, 给出完整的Agent.py文件。
注意:from langgraph.prebuilt.tool_executor import ToolExecutor 已经不存在
"""


# app_frontend.py
import streamlit as st
from Agent import graph, State, MemorySaver  # 导入原始业务逻辑
from langchain_core.messages import AIMessage, HumanMessage
import uuid

# 页面配置
st.set_page_config(
    page_title="智能研究助手",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 侧边栏设置
with st.sidebar:
    st.header("⚙️ 配置设置")
    
    # 会话管理
    st.subheader("会话管理")
    if st.button("新建会话"):
        st.session_state.clear()
    
    # 模型设置
    st.subheader("模型设置")
    ollama_model = st.selectbox(
        "选择AI模型",
        ["llama3.1:8b", "llama2", "mistral"],
        index=0,
        help="选择使用的Ollama模型"
    )
    
    # 高级设置
    st.subheader("高级设置")
    max_results = st.slider("最大搜索结果", 1, 5, 2)
    temperature = st.slider("模型温度", 0.0, 1.0, 0.7)
    stream_mode = st.checkbox("启用实时流式传输", True)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 主界面布局
st.title("🔍 智能研究助手")
st.caption("使用LangGraph和Ollama构建的智能研究助手，支持实时网络搜索")

# 显示聊天记录
for msg in st.session_state.messages:
    role = "AI" if isinstance(msg, AIMessage) else "用户"
    with st.chat_message(role):
        st.markdown(msg.content)
        if hasattr(msg, "additional_info"):
            st.json(msg.additional_info)

# 用户输入处理
if prompt := st.chat_input("请输入您的研究问题"):
    # 添加用户消息
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 准备AI响应区域
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # 构建配置
        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id,
                "model_settings": {
                    "temperature": temperature,
                    "max_results": max_results
                }
            }
        }
        
        # 运行处理流程
        try:
            events = graph.stream(
                {"messages": [user_msg]},
                config=config,
                stream_mode="values" if stream_mode else "updates"
            )
            
            # 处理实时事件
            for event in events:
                if "messages" in event:
                    latest_msg = event["messages"][-1]
                    full_response += latest_msg.content + " "
                    
                    # 更新实时显示
                    response_placeholder.markdown(full_response + "▌")
                    
                    # 显示工具调用信息
                    if hasattr(latest_msg, "tool_calls"):
                        with st.expander("查看研究过程"):
                            for tool_call in latest_msg.tool_calls:
                                st.write(f"🔧 调用工具: {tool_call['name']}")
                                st.json(tool_call["args"])
            
            # 最终显示完整响应
            response_placeholder.markdown(full_response)
            
            # 保存AI消息
            ai_msg = AIMessage(content=full_response)
            st.session_state.messages.append(ai_msg)
            
        except Exception as e:
            st.error(f"处理请求时发生错误: {str(e)}")

# 侧边栏底部信息
with st.sidebar:
    st.divider()
    st.markdown("""
    **功能特性：**
    - 多模型支持
    - 实时网络研究
    - 对话历史管理
    - 可调节参数配置
    """)