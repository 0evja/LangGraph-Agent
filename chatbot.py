from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from API_read import get_base_url, get_openai_key, get_tavily_api
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
import os

os.environ['TAVILY_API_KEY'] = get_tavily_api()

# ==================== 定义图的状态 ====================
class State(TypedDict):
    # Messages have the type "list". The 'add_messages' function
    # in the annotation defines how this state key should be updated.
    # (in this case, it appends new messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    name: str
    birthday: str

graph_builder = StateGraph(State)

# ==================== 定义工具 ======================
# 定义tavily搜索工具
tool = TavilySearchResults(max_results=2)

from typing import Annotated, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

# 定义人工协助工具的输入模型
class HumanAssistanceInput(BaseModel):
    name: str = Field(..., description="用户名")
    birthday: str = Field(..., description="生日")
    tool_call_id: Annotated[str, InjectedToolCallId]

# 定义人工协助工具
class HumanAssistanceTool(BaseTool):
    name: str = "human_assistance"
    description: str = "通过此工具请求人工协助进行验证"
    args_schema: type[BaseModel] = HumanAssistanceInput
    
    def _run(self, name: str, birthday: str, tool_call_id: str) -> str:
        """执行人工协助请求"""
        human_response = interrupt(
            {
                "question": "Is this correct?",
                "name": name,
                "birthday": birthday,
            }
        )
        
        if human_response.get("correct", "").lower().startswith("y"):
            verified_name = name
            verified_birthday = birthday
            response = "Correct"
        else:
            verified_name = human_response.get("name", name)
            verified_birthday = human_response.get("birthday", birthday)
            response = f"Made a correction: {human_response}"
        
        state_update = {
            "name": verified_name,
            "birthday": verified_birthday,
            "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
        }
        return Command(update=state_update)
    
    async def _arun(self, name: str, birthday: str, tool_call_id: str) -> str:
        """异步执行方法"""
        raise NotImplementedError("不支持异步调用")

# 创建工具实例
human_tool = HumanAssistanceTool()
tools = [tool, human_tool]

# ==================== 添加一个chatbot节点 ===================
llm = ChatOpenAI(model="gpt-4o-mini", base_url=get_base_url(), api_key=get_openai_key())
llm_with_tools = llm.bind_tools(tools)   # 让大模型学习工具

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # 由于我们将在工具执行过程中中断
    # 我们禁用并行工具以避免重复任何恢复时的工具调用
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

graph_builder.add_node("chatbot", chatbot)

# =================== 添加一个工具节点 ========================
"""
在让大模型学习过工具之后，我们需要创建一个函数来实际运行工具，
我们将工具添加到新节点，如果工具被调用，则会调用这个工具节点
"""
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# =================== 添加检查点 ==========================
memory = MemorySaver()

# ===================== 添加节点之间的边 ====================
# 条件边
"""
下述字典允许你指示图表将条件的输出解释为特定节点
它默认使用名称来标识节点 
你可以尝试换成其他节点.
条件边在未调用工具时会返回END, 因此我们不需要显示的设置finish_point
"""
graph_builder.add_conditional_edges("chatbot", tools_condition)  
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
# ===================== 创建一个CompileGraph =====================
graph = graph_builder.compile(checkpointer=memory)


# # ==================== 可视化我们构建的图 ======================
# from IPython.display import Image, display
# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     pass


# =================== 搭建线程 =================================
config = {"configurable": {"thread_id": "0evja"}}
user_input = (
    "Can you look up when LangGraph was released?"
    "When you have the answer, use the human_assistance tool for review."
)
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config=config,
    stream_mode="values"
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

    
graph.update_state(values={"name": "LangGraph(Lib)"}, config=config)

human_command = Command(
    resume={
        "name": "LangGraph",
        "birthday": "Jan 12, 2024",
    },
)
events = graph.stream(human_command, config=config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
 


# # ====================== 运行聊天机器人 ===========================
# # 定义处理对话流的方法
# def stream_graph_updates(user_input: str):
#     for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
#         # 遍历事件中的每个节点输出
#         for value in event.values():
#             # 消息是追加的，每次只打印最新的消息
#             print("Assistants:", value["messages"][-1].content)  # 每次只打印最新的消息


# # 主程序循环
# while True:
#     try:
#         user_input = input("User: ")
#         if user_input.lower() in ["quit", "exit", "q"]:
#             print("Goodbye!")
#             break
        
#         stream_graph_updates(user_input)
#     except:
#         # fallback if input() is not available
#         user_input ="what do you know about LangGraph?"
#         print("User: " + user_input)
#         stream_graph_updates(user_input)
#         break