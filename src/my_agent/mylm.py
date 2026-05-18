#加载环境变量
import os
from dotenv import load_dotenv
load_dotenv(override=True)

QIANFAN_URL = os.getenv("QIANFAN_URL")
QIANFAN_API_KEY = os.getenv("QIANFAN_API_KEY")

# 调用速率限制
from langchain_core.rate_limiters import InMemoryRateLimiter
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.2,       #每10秒最多2次请求
    check_every_n_seconds=0.1,  # 每100毫秒检查一次是否超限，默认即可
    max_bucket_size=10  # 最大突发请求数，控制峰值，默认即可
)

#格式化输出   json解析器
#langchain有.with_structured_output方法，在调用大模型时使用，会返回一个字典或pydantic对象。需要传入详见结构化输出.py
from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_core.prompts import PromptTemplate
json_parser = SimpleJsonOutputParser()

prompt = PromptTemplate(
    template="""
【绝对规则】
1. 只返回纯 JSON，不要任何其他文字
2. 不要 Markdown 格式（如 **加粗**、```json 等）
3. 不要解释、不要问候、不要多余内容
4. 严格遵守 JSON 语法，引号用双引号
5. 必须包含以下字段：
   - thinking: 详细的分步推理过程
   - answer: 最终的数学答案
   - confidence: 答案的置信度，0 到 1 之间的浮点数

问题：{question}
""",
    input_variables=["question"]
)
#调大模型
from langchain_openai import ChatOpenAI
qianFan = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=QIANFAN_API_KEY,
    base_url=QIANFAN_URL,
    temperature=0.2, 
    rate_limiter=rate_limiter,
    max_retries=3,  # 失败自动重试3次
    timeout=120
    #extra_body={ 
    #    "fps":2, 
    #    "penalty_score":1, 
    #    "stop":[], 
    #    "use_audio":True, 
    #    "compression":True,
    #    "enable_thinking": True
    #}
)
#绑定解析器
chain = prompt | qianFan | json_parser