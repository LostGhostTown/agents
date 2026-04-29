from my_agent.mylm import qianFan
from pydantic import BaseModel, Field

# 1. 定义Pydantic数据结构
class ResponseFormatter(BaseModel):
    """始终使用此工具来结构化输出。"""
    answer: str = Field(description="回答内容，必须是一个简洁的文本，不要包含任何工具调用相关的信息")
    follow_up: str = Field(description="后续跟进问题，必须是一个简洁的文本，如果没有后续问题，请返回空字符串")

runnable = qianFan.bind_tools(
    [ResponseFormatter], 
    tool_choice="ResponseFormatter"  # 核心：强制模型必须调用这个工具
)
resp=runnable.invoke("细胞的动力源是什么？")
print(resp)
resp.pretty_print()

