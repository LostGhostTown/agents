from langchain_core.prompts import PromptTemplate
from my_agent.mylm import qianFan
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# 1. 定义Pydantic数据结构
class Joke(BaseModel):
    setup: str = Field(description="笑话的铺垫，前半段背景描述")
    punchline: str = Field(description="笑话的笑点，反转的核心句子")
    score: float = Field(description="搞笑程度评分，范围0-5，保留1位小数")

# 2. 初始化解析器，自动生成格式约束指令
parser = PydanticOutputParser(pydantic_object=Joke)

# 3. 构建带强格式约束的提示词（关键！彻底锁死输出格式）
prompt = PromptTemplate(
    template="""
    你是一个专业的段子手。
    请严格遵守以下规则：
    1. 完成用户的笑话写作需求
    2. 必须严格按照下方指定的JSON格式输出
    3. 禁止输出任何JSON以外的内容、解释、注释
    4. 禁止输出markdown代码块标记，不要用```包裹内容
    5. 字段名必须严格使用英文：setup、punchline、score

    格式要求：
    {format_instructions}

    用户需求：{query}
    """,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. 构建链式调用：提示词 -> 大模型 -> 结构化解析
chain = prompt | qianFan | parser

# 5. 执行调用
resp = chain.invoke({"query": "写一个关于程序员的笑话"})
print(resp)
print(type(resp))  # 输出 <class '__main__.Joke'>