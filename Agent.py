from typing import Annotated

from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from API_read import get_tavily_api
import os

os.environ['TAVILY_API_KEY'] = get_tavily_api()

# 定义图的状态
class State(TypedDict):
    """只有一个消息状态，类型为列表，更新方式为追加"""
    messages: Annotated[list, add_messages] 

# 初始化图
graph_builder = StateGraph(State) #记得初始化图要传入图的状态，不然咋初始化

# 定义工具
tool = TavilySearchResults(max_result=2) #最大生成2条答案
tools = [tool] #工具列表

# 定义模型
llm = ChatOllama(model="llama3.1:8b")
llm_with_tools = llm.bind_tools(tools) #让模型学习工具, 虽然学习了，但是还是得调用才能用，孩子。

# 定义图的节点, 传入图的状态
def chatbot(state: State): 
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 为图添加节点和边
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

# 定义记忆
memory = MemorySaver()

# 编译图
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "0evja"}}
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm learning LangGraph."
                    "Could you do some research on it for me?"
                ),
            },
        ],
    },
    config = config,
    stream_mode = "values",
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()  # -1表示只打印最新的消息，因为消息是追加的，孩子。


events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ya that's helpful. Maybe I'll build an autonomous agent with it."
                ),
            },
        ],
    },
    config = config,
    stream_mode = "values",
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print() 


# 我们可以replay 完整的状态历史记录，以查看发生的一切
to_replay = None
for state in graph.get_state_history(config = config):
    print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
    print("-" * 20)
    if len(state.values["messages"]) == 6:
        to_replay = state