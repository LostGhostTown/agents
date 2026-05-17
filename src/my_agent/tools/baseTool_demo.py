from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import os
import requests
from dotenv import load_dotenv
load_dotenv(override=True)
QIANFAN_API_KEY = os.getenv("QIANFAN_API_KEY")
url = "https://qianfan.baidubce.com/v2/ai_search/web_search"
headers = {
    "X-Appbuilder-Authorization": f"Bearer {QIANFAN_API_KEY}",
    "Content-Type": "application/json"
}
data = {
    "messages": [
        {
            "content": query,
            "role": "user"
        }
    ],
    "search_source": "baidu_search_v2",
    # 可选：返回前5条网页结果
    "resource_type_filter": [{"type": "web", "top_k": 5}],
    # 可选：时效性过滤，支持 hour/day/week/month/year
    "search_recency_filter": "week"
}

class SearchArgs(BaseModel):
    query: str = Field(description="搜索内容的文字信息")
class MysearchTool(BaseTool):
    name = "search_tool"
    description = """使用百度搜索引擎进行联网搜索,获取互联网上的最新信息、实时数据、新闻资讯等内容。
    当用户的问题涉及以下场景时，必须使用此工具：
    1. 最新的新闻、事件、政策等时效性内容
    2. 实时数据、行情、榜单等动态信息
    3. 超出模型知识库范围的内容
    4. 需要权威来源验证的事实性内容
"""
    #直接返回内容还是返回到大模型
    return_direct = True

    args_schema = SearchArgs
    
    def _run(self,query:str):
        try:
            # 发送请求
            response = requests.post(url, headers=headers, json=data, timeout=15)
            response.raise_for_status()
            result = response.json()
            
            # ===================== 格式化搜索结果（方便大模型读取）=====================
            formatted_result = f"【搜索关键词：{query}】\n"
            # 解析官方返回的结果结构
            if "references" in result and len(result["references"]) > 0:
                for idx, item in enumerate(result["references"], 1):
                    formatted_result += f"\n{idx}. 标题：{item.get('title', '无标题')}\n"
                    formatted_result += f"   来源：{item.get('website', '未知站点')} | 发布时间：{item.get('date', '未知时间')}\n"
                    formatted_result += f"   摘要：{item.get('snippet', '无摘要')}\n"
                    formatted_result += f"   链接：{item.get('url', '无链接')}\n"
            else:
                formatted_result += "未找到相关搜索结果"
            
            return formatted_result
    
        except requests.exceptions.RequestException as e:
            return f"搜索请求失败：{str(e)}，请检查API Key或网络连接"
        except Exception as e:
            return f"搜索结果解析失败：{str(e)}"
