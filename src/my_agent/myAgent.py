from langchain.agents import create_agent
from my_agent.tools.tool_demo1 import web_search
from my_agent.mylm import qianFan
agent=create_agent(
    qianFan,
    tools=[web_search],
    system_prompt="你是一个智能助手，可以帮助用户完成各种任务。",
)