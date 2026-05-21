"""
最终版 LangGraph 固定工作流（完全符合并行需求）
功能：
1. 智能检测代码意图 → 可选并行执行代码解释器
2. 所有请求 → 强制并行执行联网搜索
3. 所有请求 → 强制并行执行RAG检索
4. 所有工具完成后整合结果生成最终回答
"""
from __future__ import annotations
import asyncio
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from typing import Literal

# ===================== 导入所有核心组件 =====================
from my_agent.mylm import qianFan
from my_agent.embeddings_demos.rag_retriever import retrieve_context
from my_agent.tools.mcpTool_demo import run_code
from my_agent.tools.baseTool_demo import search_tool
from my_agent.chat_bot import chat_agent

# ===================== 状态定义（规范版） =====================
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    user_id: str
    thread_id: str
    code_result: str | None = None
    search_result: str | None = None
    rag_context: str | None = None

# ===================== 1. 核心：智能代码意图检测 =====================
async def should_run_code(state: State) -> Literal["start_with_code", "start_without_code"]:
    """
    智能判断是否需要执行代码工具
    返回两个并行启动器节点名称
    """
    user_query = state["user_query"]
    
    prompt = f"""
    请判断用户的问题是否需要执行Python代码来回答。
    只有当用户明确要求运行代码、计算结果、执行脚本时才返回"yes"。
    如果只是询问代码概念、语法解释、最佳实践，返回"no"。
    
    用户问题：{user_query}
    
    只返回"yes"或"no"，不要任何其他内容。
    """
    
    try:
        response = await qianFan.ainvoke(prompt)
        result = response.content.strip().lower()
        return "start_with_code" if result == "yes" else "start_without_code"
    except Exception as e:
        print(f"⚠️ 代码意图检测失败，默认跳过代码执行: {e}")
        return "start_without_code"

# ===================== 2. 并行启动器节点（无操作，仅用于触发并行分支） =====================
async def start_with_code(state: State):
    """有代码分支的并行启动器：同时启动三个工具"""
    return {}

async def start_without_code(state: State):
    """无代码分支的并行启动器：同时启动两个必须工具"""
    return {}

# ===================== 3. 工具执行节点 =====================
async def run_code_tool(state: State):
    """可选：执行代码工具"""
    user_query = state["user_query"]
    try:
        result = await run_code.ainvoke(user_query)
        return {"code_result": result}
    except Exception as e:
        print(f"❌ 代码执行失败: {e}")
        return {"code_result": f"代码执行失败: {str(e)}"}

async def run_web_search(state: State):
    """必须：执行联网搜索"""
    user_query = state["user_query"]
    try:
        result = await search_tool.arun(user_query)
        return {"search_result": result}
    except Exception as e:
        print(f"❌ 联网搜索失败: {e}")
        return {"search_result": "联网搜索暂时不可用"}

async def get_rag_context(state: State):
    """必须：执行RAG检索"""
    user_query = state["user_query"]
    try:
        rag_context = await retrieve_context(user_query)
        return {"rag_context": rag_context}
    except Exception as e:
        print(f"❌ RAG检索失败: {e}")
        return {"rag_context": "知识库检索暂时不可用"}

# ===================== 4. 答案生成节点 =====================
async def generate_final_answer(state: State):
    """所有工具完成后，调用子图生成最终回答"""
    input_state = {
        **state,
        "messages": [HumanMessage(content=state["user_query"])]
    }

    final_state = await chat_agent.ainvoke(
        input=input_state,
        config={"configurable": {"thread_id": state["thread_id"]}}
    )

    return {"messages": final_state["messages"]}

# ===================== 构建工作流（完全并行版） =====================
graph_builder = StateGraph(State)

# 添加所有节点
graph_builder.add_node("start_with_code", start_with_code)
graph_builder.add_node("start_without_code", start_without_code)
graph_builder.add_node("run_code_tool", run_code_tool)
graph_builder.add_node("web_search", run_web_search)
graph_builder.add_node("rag_retrieve", get_rag_context)
graph_builder.add_node("generate_answer", generate_final_answer)

# -------------------- 核心流程定义（完全符合需求） --------------------
# 1. 起点 → 条件判断
graph_builder.add_conditional_edges(
    START,
    should_run_code,
    {
        "start_with_code": "start_with_code",
        "start_without_code": "start_without_code"
    }
)

# 2. 有代码分支：同时启动三个工具（完全并行）
graph_builder.add_edge("start_with_code", "run_code_tool")
graph_builder.add_edge("start_with_code", "web_search")
graph_builder.add_edge("start_with_code", "rag_retrieve")

# 3. 无代码分支：同时启动两个必须工具（完全并行）
graph_builder.add_edge("start_without_code", "web_search")
graph_builder.add_edge("start_without_code", "rag_retrieve")

# 4. 所有工具完成后，统一进入答案生成节点
# LangGraph会自动等待所有前置节点全部完成
graph_builder.add_edge("run_code_tool", "generate_answer")
graph_builder.add_edge("web_search", "generate_answer")
graph_builder.add_edge("rag_retrieve", "generate_answer")

# 5. 生成答案后结束
graph_builder.add_edge("generate_answer", END)

# 编译异步工作流
graph = graph_builder.compile()

# ===================== 测试调用 =====================
async def main():
    test_query = "请帮我计算1-100的和，同时查一下Python求和的方法"
    test_user_id = "user_123"
    test_thread_id = "thread_456_20260521"
    
    result = await graph.ainvoke({
        "user_query": test_query,
        "user_id": test_user_id,
        "thread_id": test_thread_id,
        "messages": []
    })
    
    print("="*50)
    print("最终回答：")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())