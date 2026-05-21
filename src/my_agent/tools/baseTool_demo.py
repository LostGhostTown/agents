from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import os
import httpx
from dotenv import load_dotenv
from typing import Type

load_dotenv(override=True)
QIANFAN_API_KEY = os.getenv("QIANFAN_API_KEY")
if not QIANFAN_API_KEY:
    raise ValueError("请在.env文件中设置QIANFAN_API_KEY环境变量")

URL = "https://qianfan.baidubce.com/v2/ai_search/web_search"
HEADERS = {
    "Authorization": f"Bearer {QIANFAN_API_KEY}",  # ✅ 修复：统一认证头
    "Content-Type": "application/json"
}

# 入参模型
class SearchArgs(BaseModel):
    query: str = Field(description="搜索关键词")

class MysearchTool(BaseTool):
    name: str = "search_tool"
    description: str = """使用百度搜索引擎进行联网搜索,获取互联网上的最新信息、实时数据、新闻资讯等内容。
    当用户的问题涉及以下场景时，必须使用此工具：
    1. 最新的新闻、事件、政策等时效性内容
    2. 实时数据、行情、榜单等动态信息
    3. 超出模型知识库范围的内容
    4. 需要权威来源验证的事实性内容
"""
    return_direct: bool = False  # ✅ 修复：关闭直接返回，回到工作流
    args_schema: Type[BaseModel] = SearchArgs
    
    def _run(self, query: str):
        """同步搜索方法"""
        data = {
            "messages": [{"content": query, "role": "user"}],
            "search_source": "baidu_search_v2",
            "resource_type_filter": [{"type": "web", "top_k": 5}],
            "search_recency_filter": "week"
        }
        try:
            import requests
            response = requests.post(URL, headers=HEADERS, json=data, timeout=15)
            response.raise_for_status()
            result = response.json()
            
            formatted_result = f"【搜索关键词：{query}】\n"
            if "references" in result and len(result["references"]) > 0:
                for idx, item in enumerate(result["references"], 1):
                    formatted_result += f"\n{idx}. 标题：{item.get('title', '无标题')}\n"
                    formatted_result += f"   来源：{item.get('website', '未知')} | 时间：{item.get('date', '未知')}\n"
                    formatted_result += f"   摘要：{item.get('snippet', '无')}\n"
            else:
                formatted_result += "未找到相关结果"
            return formatted_result
            
        except Exception as e:
            return f"搜索失败：{str(e)}"
    
    async def _arun(self, query: str):
        """✅ 新增：异步搜索方法（适配父图异步工作流）"""
        data = {
            "messages": [{"content": query, "role": "user"}],
            "search_source": "baidu_search_v2",
            "resource_type_filter": [{"type": "web", "top_k": 5}],
            "search_recency_filter": "week"
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(URL, headers=HEADERS, json=data)
                response.raise_for_status()
                result = response.json()
            
            formatted_result = f"【搜索关键词：{query}】\n"
            if "references" in result and len(result["references"]) > 0:
                for idx, item in enumerate(result["references"], 1):
                    formatted_result += f"\n{idx}. 标题：{item.get('title', '无标题')}\n"
                    formatted_result += f"   来源：{item.get('website', '未知')} | 时间：{item.get('date', '未知')}\n"
                    formatted_result += f"   摘要：{item.get('snippet', '无')}\n"
            else:
                formatted_result += "未找到相关结果"
            return formatted_result
            
        except Exception as e:
            return f"搜索失败：{str(e)}"

# 导出工具实例
search_tool = MysearchTool()