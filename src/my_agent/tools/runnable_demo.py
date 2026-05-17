from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from my_agent.mylm import qianFan
from pydantic import BaseModel, Field

prompt = (
    PromptTemplate.from_template("帮我生成一个简短的，关于{topic}的报幕词")
    +",要求： 1、内容搞笑"
    +"2、输出内容采用{language}。"
)
chain = prompt | qianFan | StrOutputParser()

class ToolArgs(BaseModel):
    topic:str = Field(description="报幕词的主题")
    language:str = Field(description="报幕词的语言")

runnable_tool=chain.as_tool(
    name='runnable0_tool',
    description='这是一个专门生成报幕词的工具',
    args_schema=ToolArgs
)