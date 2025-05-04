import streamlit as st
from Adaptive_RAG import graph, format_output
import time

# 设置页面配置
st.set_page_config(
    page_title="Adaptive RAG Demo",
    page_icon="🤖",
    layout="wide"
)


# 侧边栏配置
st.sidebar.title("⚙️ Configuration")

# 模型配置
st.sidebar.subheader("Model Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
max_retries = st.sidebar.slider("Max Retries", 1, 5, 3)

# 检索配置
st.sidebar.subheader("Retrieval Settings")
k_docs = st.sidebar.slider("Number of Documents to Retrieve", 1, 10, 3)

# 数据源显示
st.sidebar.subheader("Available Data Sources")
st.sidebar.markdown("""
- 📚 Vector Store (Local Documents)
- 🌐 Web Search (Tavily API)
""")

# 系统状态指示器
system_status = st.sidebar.empty()
system_status.success("System Ready")

# 主界面
col1, col2 = st.columns([2, 1])

with col1:
    
    # 添加标题和描述
    st.title("🤖 Adaptive RAG System")
    st.markdown("""
    This is an advanced RAG system that can:
    - Dynamically choose between vector store and web search
    - Check for hallucinations
    - Validate answer relevance
    - Retry if necessary
    """)

    # 输入区域
    st.subheader("🔍 Ask Your Question")
    user_question = st.text_area(
        "Enter your question here:",
        height=100,
        placeholder="e.g., What are the latest developments in LLM agents?"
    )
    
    # 提交按钮
    submit = st.button("🚀 Submit", type="primary")

    if submit and user_question:
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 准备输入
        inputs = {
            "question": user_question,
            "max_retries": max_retries
        }
        
        # 创建结果容器
        result_container = st.empty()
        
        try:
            # 执行查询
            steps = 0
            for event in graph.stream(inputs, stream_mode="values"):
                steps += 1
                progress = min(steps / 5, 1.0)  # 假设最多5个步骤
                progress_bar.progress(progress)
                
                # 更新状态
                if "documents" in event:
                    status_text.info("📑 Retrieved relevant documents...")
                elif "web_search" in event:
                    status_text.info("🌐 Performing web search...")
                elif "generation" in event:
                    status_text.info("✍️ Generating response...")
                
                # 显示结果
                output = format_output(event)
                if output:
                    result_container.markdown(f"### 💡 Answer:\n{output}")
                
                time.sleep(0.5)  # 添加小延迟以展示进度
            
            # 完成
            progress_bar.progress(1.0)
            status_text.success("✅ Processing complete!")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            system_status.error("System Error")

with col2:
    # 系统信息展示
    st.subheader("📊 System Information")
    
    # 添加指标
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(label="Model", value="LLama 3.1")
    with col_b:
        st.metric(label="Embeddings", value="Nomic")
    
    # 添加处理流程展示
    st.subheader("🔄 Processing Flow")
    st.markdown("""
    1. **Question Analysis**
       - Route to appropriate data source
    2. **Document Retrieval**
       - Vector store or web search
    3. **Relevance Check**
       - Grade document relevance
    4. **Answer Generation**
       - Generate concise response
    5. **Quality Control**
       - Check for hallucinations
       - Verify answer relevance
    """)

# 添加页脚
st.markdown("---")
st.markdown("*Built with Streamlit & LangGraph 🚀*")