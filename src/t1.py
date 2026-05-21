"""
2026年5月最新版导入验证
所有导入路径均符合官方最新文档
"""
try:
    # LangChain 核心
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
    from langchain_core.tools import tool
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    print("✅ LangChain 核心模块导入成功")

    # LangGraph 核心
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    print("✅ LangGraph 核心模块导入成功")

    # 检查点后端
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.memory import MemorySaver
    print("✅ 检查点后端导入成功")

    print("\n🎉 所有最新版模块导入成功！可以开始开发了")

except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("\n请重新执行安装命令:")
    print("pip install --upgrade langchain langchain-core langchain-openai langgraph langgraph-checkpoint-postgres psycopg2-binary python-dotenv")