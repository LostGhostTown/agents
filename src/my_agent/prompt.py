from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from my_agent.mylm import qianFan
example=[
    {
        "question":"学习agent的计划",
        "answer":"""
1. 了解Agent的基本概念和原理
2. 学习Agent的常见架构和设计模式
3. 了解Agent的使用场景和适用性
"""
    }
]
base_prompt = PromptTemplate.from_template("""
问题：{question}\n{answer}
""")#这里是告诉程序如何把 examples 列表里的每个字典（如 {"question":..., "answer":...}）转换成一段可读的文本

#指令大模型学习示例的模板
final_template = FewShotPromptTemplate(
    examples=example,
    example_prompt=base_prompt,
    prefix="你是一个智能助手，学习以下示例的格式和内容，来回答用户的问题：",#这里是前置提示
    suffix="请根据上面的示例，回答以下问题：{question}",#这里传入实际问题
)
input=final_template| qianFan
res=input.invoke({"question":"学习agent的计划"})
print(res)
