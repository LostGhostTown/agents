"""
百度千帆代码解释器MCP服务调用模块
自动从环境变量读取API密钥
支持同步、异步和流式输出
可直接被其他Python文件导入使用
"""
import os
import json
import requests
import httpx
from typing import Optional, Iterator, AsyncIterator, Dict, Any, Type
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# 自动加载.env文件（如果存在）
load_dotenv(override=True)

@dataclass
class CodeInterpreterResponse:
    """代码解释器响应结果"""
    content: str
    code_generated: Optional[str] = None
    code_output: Optional[str] = None
    is_success: bool = True
    error_message: Optional[str] = None

class QianfanCodeInterpreter:
    """
    百度千帆代码解释器MCP服务客户端
    用于调用"代码解释器"智能体，自动分析用户需求，生成代码并运行
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://qianfan.baidubce.com/v2/tools/code-interpreter-agent/mcp",
        timeout: int = 300,  # 代码执行可能需要较长时间
        stream_timeout: int = 600
    ):
        """
        初始化代码解释器客户端
        
        Args:
            api_key: 百度千帆API密钥（格式：bce-v3/your_api_key_here）
                     如果不传入，会自动从环境变量QIANFAN_API_KEY读取
            base_url: MCP服务地址
            timeout: 同步调用超时时间（秒）
            stream_timeout: 流式调用超时时间（秒）
        """
        # 优先使用传入的api_key，否则从环境变量读取
        self.api_key = api_key or os.getenv("QIANFAN_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "未找到百度千帆API密钥。请在.env文件中设置QIANFAN_API_KEY，"
                "或者在初始化时传入api_key参数。"
            )
        
        self.base_url = base_url
        self.timeout = timeout
        self.stream_timeout = stream_timeout
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def invoke(self, query: str) -> CodeInterpreterResponse:
        """
        同步调用代码解释器
        
        Args:
            query: 用户的问题或需求
            
        Returns:
            CodeInterpreterResponse: 包含回答、生成的代码和执行结果
        """
        try:
            # ✅ 修复：正确的请求格式（与搜索工具一致）
            payload = {"messages": [{"role": "user", "content": query}]}
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            return self._parse_response(result)
            
        except requests.exceptions.RequestException as e:
            return CodeInterpreterResponse(
                content="",
                is_success=False,
                error_message=f"请求失败: {str(e)}"
            )
        except json.JSONDecodeError:
            return CodeInterpreterResponse(
                content="",
                is_success=False,
                error_message="响应解析失败: 无效的JSON格式"
            )
        except Exception as e:
            return CodeInterpreterResponse(
                content="",
                is_success=False,
                error_message=f"未知错误: {str(e)}"
            )
    
    def stream(self, query: str) -> Iterator[CodeInterpreterResponse]:
        """
        流式调用代码解释器
        
        Args:
            query: 用户的问题或需求
            
        Yields:
            CodeInterpreterResponse: 逐步返回的响应片段
        """
        try:
            payload = {"messages": [{"role": "user", "content": query}], "stream": True}
            
            with requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=self.stream_timeout
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                yield self._parse_chunk(chunk)
                            except json.JSONDecodeError:
                                continue
                                
        except requests.exceptions.RequestException as e:
            yield CodeInterpreterResponse(
                content="",
                is_success=False,
                error_message=f"流式请求失败: {str(e)}"
            )
    
    async def ainvoke(self, query: str) -> CodeInterpreterResponse:
        """
        异步调用代码解释器
        
        Args:
            query: 用户的问题或需求
            
        Returns:
            CodeInterpreterResponse: 包含回答、生成的代码和执行结果
        """
        try:
            payload = {"messages": [{"role": "user", "content": query}]}
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload
                )
                
                response.raise_for_status()
                result = response.json()
                
                return self._parse_response(result)
                
        except httpx.HTTPError as e:
            return CodeInterpreterResponse(
                content="",
                is_success=False,
                error_message=f"异步请求失败: {str(e)}"
            )
        except json.JSONDecodeError:
            return CodeInterpreterResponse(
                content="",
                is_success=False,
                error_message="响应解析失败: 无效的JSON格式"
            )
        except Exception as e:
            return CodeInterpreterResponse(
                content="",
                is_success=False,
                error_message=f"未知错误: {str(e)}"
            )
    
    async def astream(self, query: str) -> AsyncIterator[CodeInterpreterResponse]:
        """
        异步流式调用代码解释器
        
        Args:
            query: 用户的问题或需求
            
        Yields:
            CodeInterpreterResponse: 逐步返回的响应片段
        """
        try:
            payload = {"messages": [{"role": "user", "content": query}], "stream": True}
            
            async with httpx.AsyncClient(timeout=self.stream_timeout) as client:
                async with client.stream(
                    "POST",
                    self.base_url,
                    headers=self.headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line:
                            if line.startswith('data: '):
                                data = line[6:]
                                if data == '[DONE]':
                                    break
                                try:
                                    chunk = json.loads(data)
                                    yield self._parse_chunk(chunk)
                                except json.JSONDecodeError:
                                    continue
                                    
        except httpx.HTTPError as e:
            yield CodeInterpreterResponse(
                content="",
                is_success=False,
                error_message=f"异步流式请求失败: {str(e)}"
            )
    
    def _parse_response(self, result: Dict[str, Any]) -> CodeInterpreterResponse:
        """解析完整响应"""
        content = result.get("content", "")
        code_generated = result.get("code_generated")
        code_output = result.get("code_output")
        
        return CodeInterpreterResponse(
            content=content,
            code_generated=code_generated,
            code_output=code_output,
            is_success=True
        )
    
    def _parse_chunk(self, chunk: Dict[str, Any]) -> CodeInterpreterResponse:
        """解析流式响应片段"""
        content = chunk.get("content", "")
        code_generated = chunk.get("code_generated")
        code_output = chunk.get("code_output")
        
        return CodeInterpreterResponse(
            content=content,
            code_generated=code_generated,
            code_output=code_output,
            is_success=True
        )

# 全局单例实例
code_interpreter = QianfanCodeInterpreter()

# ✅ 修复：改为BaseTool实现，支持异步调用
class RunCodeArgs(BaseModel):
    query: str = Field(description="用户的问题或需求，描述清楚需要解决的问题")

class RunCodeTool(BaseTool):
    name: str = "run_code"
    description: str = """代码解释器工具，可以执行Python代码解决数学计算、数据分析、绘图等问题。
    当用户需要进行数值计算、数据处理、生成图表或执行任何需要代码的任务时使用此工具。
"""
    return_direct: bool = False
    args_schema: Type[BaseModel] = RunCodeArgs
    
    def _run(self, query: str) -> str:
        result = code_interpreter.invoke(query)
        
        if result.is_success:
            # ✅ 修复：返回完整的执行信息，包括代码和输出
            formatted_result = "【代码执行结果】\n"
            if result.code_generated:
                formatted_result += f"\n生成的代码：\n```python\n{result.code_generated}\n```\n"
            if result.code_output:
                formatted_result += f"\n执行输出：\n{result.code_output}\n"
            formatted_result += f"\n回答：{result.content}"
            return formatted_result
        else:
            return f"代码执行失败: {result.error_message}"
    
    async def _arun(self, query: str) -> str:
        result = await code_interpreter.ainvoke(query)
        
        if result.is_success:
            formatted_result = "【代码执行结果】\n"
            if result.code_generated:
                formatted_result += f"\n生成的代码：\n```python\n{result.code_generated}\n```\n"
            if result.code_output:
                formatted_result += f"\n执行输出：\n{result.code_output}\n"
            formatted_result += f"\n回答：{result.content}"
            return formatted_result
        else:
            return f"代码执行失败: {result.error_message}"

# 导出工具实例（与原接口保持兼容）
run_code = RunCodeTool()