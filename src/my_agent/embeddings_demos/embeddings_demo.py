import requests
import json
import base64
import os
from typing import Optional, List, Union
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(override=True)

class QianfanMultiModalEmbeddings(Embeddings):
    """
    百度千帆多模态嵌入模型封装类
    支持：纯文本、纯图片、文本+图片 三种输入模式
    兼容：LangChain生态，可直接接入向量库、RAG链
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://qianfan.baidubce.com/v2"):
        """
        初始化千帆多模态嵌入模型
        :param api_key: 千帆API Key，不传则从环境变量QIANFAN_API_KEY读取
        :param base_url: 千帆API地址，默认v2版本
        """
        self.api_key = api_key or os.getenv("QIANFAN_API_KEY")
        if not self.api_key:
            raise ValueError("请提供千帆API Key，或设置环境变量QIANFAN_API_KEY")
        
        self.base_url = base_url.rstrip("/")
        self.embedding_url = f"{self.base_url}/embeddings"
        self.model = "gme-qwen2-vl-2b-instruct"
        
        # 鉴权头
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

    def _image_to_base64(self, image_path: str) -> str:
        """本地图片转base64"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"本地图片不存在：{image_path}")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _is_url(self, image_input: str) -> bool:
        """判断是否是网络URL"""
        return image_input.startswith(("http://", "https://"))

    def embed(self, text: Optional[str] = None, image: Optional[str] = None) -> List[float]:
        """
        【核心调用方法】生成多模态向量
        :param text: 文本内容（可选）
        :param image: 图片内容（可选），支持：
                      - 本地图片路径（如："./test.jpg"）
                      - 网络图片URL（如："https://xxx.com/xxx.jpg"）
        :return: 向量数组（List[float]）
        """
        # 校验：至少要有一个输入
        if not text and not image:
            raise ValueError("text和image至少要提供一个")
        
        # 构建input对象
        input_item = {}
        if text:
            input_item["text"] = text
        if image:
            if self._is_url(image):
                # 网络图片直接传URL
                input_item["image"] = image
            else:
                # 本地图片转base64
                input_item["image"] = self._image_to_base64(image)
        
        # 构建请求体
        payload = json.dumps({
            "model": self.model,
            "input": [input_item]  # 多模态模型必须是单元素数组
        })
        
        # 发送请求
        try:
            response = requests.post(
                self.embedding_url,
                headers=self.headers,
                data=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # 提取向量
            if "data" in result and len(result["data"]) > 0:
                return result["data"][0]["embedding"]
            else:
                raise ValueError(f"API返回格式异常：{result}")
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"网络请求失败：{str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"向量生成失败：{str(e)}") from e

    # ==================== 以下是LangChain兼容的强制方法 ====================
    def embed_query(self, text: str) -> List[float]:
        """LangChain兼容：单条查询文本向量化（仅文本）"""
        return self.embed(text=text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain兼容：批量文档向量化（仅文本）"""
        return [self.embed(text=t) for t in texts]