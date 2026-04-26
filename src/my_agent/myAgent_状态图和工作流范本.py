from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# 复用你已有的工具和模型
from my_agent.tools.tool_demo1 import web_search
from my_agent.mylm import qianFan

# ===================== 核心条件1：定义工具列表 =====================
tools = [web_search]

# ===================== 核心条件2：给模型绑定工具（最容易被忽略） =====================
# 只有bind_tools后，模型才知道"我可以调用这些工具"
llm_with_tools = qianFan.bind_tools(tools)

# ===================== LangGraph标准状态定义 =====================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ===================== 图节点定义 =====================
def agent_node(state: AgentState):
    """调用绑定了工具的模型，加上严格的工具调用规则"""
    # 核心：给模型加system prompt，明确工具调用规则
    system_prompt = """
    你是一个必须优先使用联网搜索工具的智能助手，严格遵守以下规则：
    1. 只要用户的问题符合以下任意一种情况，必须先调用【联网搜索】工具，禁止直接使用自身知识库回答：
       - 涉及实时信息、天气、新闻、最新政策、时效性内容
       - 涉及需要权威来源验证的事实性内容
       - 问题中出现了「查询」「搜索」「最新」「现在」「今天」等关键词
    2. 只有当问题是纯逻辑、纯理论、完全不需要外部信息的常识内容，且你100%确定答案准确时，才能直接回答。
    3. 调用工具时，必须严格按照工具的入参要求，传入准确的搜索关键词。
    """
    
    # 把system prompt拼接到对话最前面
    messages = [
        {"role": "system", "content": system_prompt}
    ] + state["messages"]
    
    # 调用绑定了工具的模型
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 工具执行节点：必须传入tools列表
tool_node = ToolNode(tools)

# ===================== 核心条件3：定义路由逻辑（让流程能走到工具节点） =====================
def should_continue(state: AgentState) -> Literal["tools", END]:
    """判断模型是否要调用工具"""
    last_message = state["messages"][-1]
    # 如果模型生成了tool_calls，就去执行工具
    if last_message.tool_calls:
        return "tools"
    # 否则直接结束
    return END

# ===================== 构建并编译图 =====================
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# 导出langgraph dev需要的编译后对象
agent = workflow.compile()

# ===================== 本地快速测试（可选，不用langgraph dev也能跑） =====================
if __name__ == "__main__":
    print("--- 本地测试工具调用 ---")
    test_query = "嘉兴今天的天气怎么样？"
    result = agent.invoke({"messages": [HumanMessage(content=test_query)]})
    print("\n最终回答：")
    print(result["messages"][-1].content)