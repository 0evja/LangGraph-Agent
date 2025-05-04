from langchain_ollama import ChatOllama, OllamaEmbeddings
from API_read import get_tavily_api
import os

os.environ['TAVILY_API_KEY'] = get_tavily_api()

# 定义模型和嵌入
local_llm = "llama3.1:8b"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 定义向量数据库
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# 分割数据
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 1000,
    chunk_overlap = 200,
)
docs_splits = text_splitter.split_documents(docs_list)

# 向量存储和检索器
vectorstore = SKLearnVectorStore.from_documents(
    documents = docs_splits,
    embedding = local_embeddings
)
retriever = vectorstore.as_retriever(k=3)

# 定义网路搜索工具
from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults(k=3)

# ============= 定义组件 ====================
import json
from langchain_core.messages import HumanMessage, SystemMessage

# 使用检索到的文档回答提示词
rag_prompt = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:"""

# Doc grader instructions
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

# 判断文档是否与问题相关的提示词
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

# web search or RAG
router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.

Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

# Hallucination grader instructions
hallucination_grader_instructions = """

You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader prompt
hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

# Answer grader instructions
answer_grader_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) The STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader prompt
answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""



# 后处理函数
"""
后处理函数的作用:
将检索到的多个文档片段合并为一个连贯的上下文块。
提供给LLM作为生成答案的依据
符合RAG提示词模板中{context}占位符的输入格式要求

"""
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ============== 使用LangGraph构建图 ========================
import operator
from typing_extensions import TypedDict
from typing import List, Annotated
from langchain.schema import Document
from langgraph.graph import END

# 定义图状态
class GraphState(TypedDict):
    question: str                                      # 用户问题
    generation: str                                    # LLM 生成
    web_search: str                                    # 运行web 搜索的二进制决策
    max_retries: int                                   # 生成答案的最大重试次数
    answers: int                                       # 生成的答案数
    loop_step: Annotated[int, operator.add]            
    documents: List[str]                               # 检索到的文档列表

# ============= Node ================
# 根据问题检索文档
def retrieve(state):
    """Retrieve documents from vectorstore"""

    print("--- RETRIEVE ---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}

# 只根据检索的文档生成答案
def generate(state):
    """Generate answer using RAG on retrieved documents"""

    print("--- GENERATE ---")
    question = state["question"]
    documents =  state["documents"]
    loop_step = state.get("loop_step", 0)

    # RAG 生成
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1}

# 决定是否启用网络搜索
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question,
    If any document is not relevant, we will set a flag to run web search.
    """
    print("--- CHECK DOCUMENT RELEVANT TO QUESTION ---")
    question = state["question"]
    documents = state["documents"]

    # score each doc
    filtered_doc = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        # document relevant
        if grade.lower() == "yes":
            print("--- GRADE: DOCUMENT RELEVANT ---")
            filtered_doc.append(d)
        # document not relevant
        else:
            print("--- GRADE: DOCUMENT NOT RELEVANT ---")
            """只要检索到一个不相关的文档即触发后续网络搜索"""
            web_search = "Yes"
            continue
    
    return {"documents": filtered_doc, "web_search": web_search}

# 网络搜索
def web_search(state):
    """Web search based on the question"""

    print("--- WEB SEARCH ---")
    question = state["question"]
    documents = state.get("documents", [])  # 如果documents存在则返回documents,否则返回空列表

    # web search
    docs = web_search_tool.invoke({"query": question})
    web_results = '\n'.join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents}

# ============ Edges ===============
def route_question(state):
    """Route question to web or RAG"""

    print("--- ROUTE QUESTION ---")
    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["question"])]
    )
    source = json.loads(route_question.content)["datasource"]
    if source == "websearch":
        print("--- ROUTE QUESTION TO WEB SEARCH ---")
        return "websearch"
    elif source == "vectorstore":
        print("--- ROUTE QUESTION TO RAG ---")
        return "vectorstore"
    
def decide_to_generate(state):
    """Determines whether to generate an answer or add web search"""
    
    print("--- ASSESS GRADED DOCUMENTS ---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        """All documents have been filtered check_relevance, we will re-generate a new query"""
        print("--- DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, RUN WEB SEARCH ---")
        return "websearch"
    else: 
        """we have all relevant documents, so generate answer"""
        print("--- DECISION: GENERATE ---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Detemines whether the generation is grounded in the document and answers question
    """

    print("--- CHECK HALLUCINATIONS ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3) #Default 3 if not provided

    # 幻觉检测
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=generation.content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]

    if grade == "yes":
        print("--- DECISION: GENERATION IS GROUNDED IN DOCUMENTS ---")
        # check question-answering
        print("--- GRADE GENERATION VS QUESTION ---")
        # Test using question and generation from above
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation.content
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            print("--- DECISION: GENERATION ADDRESSES QUESTION ---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("--- DECISION: GENERATION DOES NOT ADDRESS QUESTION ---")
            return "not useful"
        else:
            print("--- DECISION: MAX RETRIES REACHED ---")
            return "max retries"
    elif state["loop_step"] <= max_retries:
        print("--- DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS RE-TRY ---")
        return "not supported"
    else:
        print("--- DECISION: MAX RETRIES REACHED ---")
        return "max retries"
    
# =================== 构造图 ===================
from langgraph.graph import StateGraph
from IPython.display import Image, display
 
workflow = StateGraph(GraphState)

# 添加Node
workflow.add_node("websearch", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

# 添加边
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch" : "websearch",
        "vectorstore": "retrieve",
    }
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    },
)

# 添加历史记忆
from langgraph.checkpoint.memory import MemorySaver
import uuid

class AgentMemory:
    def __init__(self):
        self.memory = MemorySaver()
        self.context = {}            # 上下文字典
    
    def save_state(self, thread_id: str, state: GraphState):
        """保存状态到内存"""
        self.memory.save(thread_id, state)
        
        # 保存/更新上下文信息
        if thread_id not in self.context:
            self.context[thread_id] = {}
        
        # 从对话中提取信息更细上下文
        

    def load_state(self, thread_id: str) -> GraphState:
        """加载特定对话的状态"""
        try:
            return self.memory.load(thread_id)
        except KeyError:
            # 如果是新对话，返回初始状态
            return {
                "question": "",
                "generation": "",
                "web_search": "No",
                "max_retries": 3,
                "answers": 0,
                "loop_step": 0,
                "documents": [],
            }
    def get_history(self, thread_id: str):
        """获取对话历史"""
        try:
            return self.memory.list_checkpoints(thread_id)
        except KeyError:
            return []

# 创建记忆实例
agent_memory = AgentMemory()

# 编译图
graph = workflow.compile(checkpointer=agent_memory.memory)

# ====================== 对话管理函数 ======================
def create_new_conversation():
    """创建新的对话线程，生成一个唯一标识符"""
    return str(uuid.uuid4())

# 格式化输出函数
def format_output(event):
    """格式化输出结果"""
    if "generation" in event:
        return f"\nAnswer:\n{event['generation'].content}"
    return None


def process_query(thread_id: str, query: str, max_retries: int = 3):
    # 准备输入
    inputs = {
        "question": query,
        "max_retries": max_retries
    }

    # 配置
    config = {
        "configurable": {"thread_id": thread_id}
    }

    # 处理查询
    response = None
    for event in graph.stream(inputs, config=config, stream_mode="values"):
        if "generation" in event:
            response = event["generation"].content
    return response


def main():
    # 创建新对话
    thread_id = create_new_conversation()
    print(f"对话ID: {thread_id}")

    # 示例查询
    query = "Hello my name is Jim, could you tell me what is large language model?"
    response = process_query(thread_id=thread_id, query=query)
    print("\n问题: ", query)
    print("\n回答: ", response)
    query = "What's my name?"
    response = process_query(thread_id=thread_id, query=query)
    print("\n问题: ", query)
    print("\n回答: ", response)

    # # 加载历史状态
    # history = agent_memory.get_history(thread_id=thread_id)
    # print("\n对话历史状态: ", len(history))

    # # 从历史状态中恢复
    # last_state = agent_memory.load_state(thread_id=thread_id)
    # print("上一次对话状态: ", last_state.get("question"))

if __name__ == "__main__":
    main()
