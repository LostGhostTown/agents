# ===================== 环境配置 =====================
import os
import base64
import asyncio
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from typing import Annotated, List, Optional, AsyncIterator, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.message import add_messages
from my_agent.mylm import qianFan

# 导入历史切片管理器
from my_agent.history_manager import AsyncChatHistoryManager

load_dotenv()
POSTGRES_URI = os.getenv("POSTGRES_URI")
if not POSTGRES_URI:
    raise ValueError("请在.env文件中设置POSTGRES_URI环境变量")

# ===================== 全局资源初始化（修复异步问题） =====================
checkpointer: Optional[AsyncPostgresSaver] = None
history_manager: Optional[AsyncChatHistoryManager] = None

async def init_global_resources():
    """统一异步初始化所有全局资源"""
    global checkpointer, history_manager
    
    # 1. 初始化LangGraph异步PG持久化
    checkpointer = AsyncPostgresSaver.from_conn_string(POSTGRES_URI)
    async with checkpointer._engine.begin() as conn:
        await checkpointer.create_tables(conn)
    
    # 2. 初始化历史切片管理器
    history_manager = AsyncChatHistoryManager(POSTGRES_URI)
    await history_manager.setup()
    
    print("✅ 所有全局资源初始化完成")

# ===================== 【核心扩展】状态定义 =====================
class ChatAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    # 👇 扩展字段：外部工作流传入工具/RAG结果（单次有效）
    code_result: Optional[str] = None
    search_result: Optional[str] = None
    rag_context: Optional[str] = None

# ===================== 系统提示词 =====================
system_prompt = SystemMessage(content="""
你是一个专业的多模态沟通助手。
- 可以理解和分析图片、视频、音频内容
- 可以读取和分析PDF/TXT等文本文件
- 可以执行数学计算、数据分析、代码编写
- 请结合聊天历史、参考资料、工具执行结果准确回答
- 如果有【全局对话摘要】或【近期对话摘要】，请优先参考理解上下文
""")

# ===================== 多模态消息预处理（新增：用于历史切片） =====================
def multimodal_message_to_text(message: BaseMessage) -> str:
    """将多模态消息转换为纯文本，用于历史摘要"""
    if isinstance(message.content, str):
        return message.content
    
    text_parts = []
    for part in message.content:
        if part["type"] == "text":
            text_parts.append(part["text"])
        elif part["type"] == "image_url":
            text_parts.append("[用户上传了一张图片]")
        elif part["type"] == "video_url":
            text_parts.append("[用户上传了一段视频]")
        elif part["type"] == "audio_url":
            text_parts.append("[用户上传了一段音频]")
    
    return " ".join(text_parts)

# ===================== 核心业务逻辑（完全重构） =====================
async def build_final_messages(state: ChatAgentState, thread_id: str) -> List[BaseMessage]:
    """
    构建最终提示词（集成历史切片）
    顺序：系统提示 → 工具/RAG结果 → 历史切片上下文
    """
    messages = [system_prompt]

    # 1. 加入单次有效的工具执行结果 + RAG内容
    tool_rag_messages = []
    if state.get("code_result"):
        tool_rag_messages.append(SystemMessage(content=f"【代码执行结果】：{state['code_result']}"))
    if state.get("search_result"):
        tool_rag_messages.append(SystemMessage(content=f"【联网搜索结果】：{state['search_result']}"))
    if state.get("rag_context"):
        tool_rag_messages.append(SystemMessage(content=f"【RAG参考资料】：{state['rag_context']}"))
    
    messages.extend(tool_rag_messages)

    # 2. 🔥 核心：调用历史切片管理器获取分层上下文
    # 替代原有的简单历史压缩和全局摘要
    sliced_context = await history_manager.build_final_context(thread_id)
    messages.extend(sliced_context)

    return messages

# ===================== 多模态文件处理（完全不变） =====================
def get_mime_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    mime_map = {
        'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png',
        'gif': 'image/gif', 'webp': 'image/webp', 'bmp': 'image/bmp',
        'mp4': 'video/mp4', 'avi': 'video/x-msvideo', 'mov': 'video/quicktime',
        'mkv': 'video/x-matroska', 'flv': 'video/x-flv', 'webm': 'video/webm',
        'mp3': 'audio/mpeg', 'wav': 'audio/wav', 'flac': 'audio/flac',
        'aac': 'audio/aac', 'm4a': 'audio/m4a'
    }
    return mime_map.get(ext, 'application/octet-stream')

def get_file_category(file_path: str, mime_type: str = "") -> str:
    if mime_type:
        if mime_type.startswith('image/'): return 'image'
        if mime_type.startswith('video/'): return 'video'
        if mime_type.startswith('audio/'): return 'audio'
    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']: return 'image'
    if ext in ['mp4', 'avi', 'mov', 'mkv', 'flv', 'webm']: return 'video'
    if ext in ['mp3', 'wav', 'flac', 'aac', 'm4a']: return 'audio'
    if ext in ['pdf', 'txt', 'md', 'docx', 'doc', 'rtf']: return 'text'
    return 'other'

def file_to_base64(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def read_text_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    content = ""
    try:
        if ext == 'pdf':
            reader = PdfReader(file_path)
            for page in reader.pages:
                content += page.extract_text() + "\n"
        elif ext in ['txt', 'md']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
    except Exception as e:
        print(f"❌ 文件读取失败: {e}")
        content = f"[文件读取失败: {str(e)}]"
    if len(content) > 10000:
        content = content[:10000] + "\n\n[文件过长，已截断...]"
    return content

def build_multimodal_message(text_part: str, files_data: List[dict]) -> List[dict]:
    content = []
    if text_part and text_part.strip():
        content.append({"type": "text", "text": text_part.strip()})
    elif len(files_data) > 0:
        content.append({"type": "text", "text": "请基于用户上传的文件内容进行回答"})
    for file_data in files_data:
        category = file_data['category']
        file_path = file_data['path']
        mime_type = file_data['mime_type']
        base64_str = file_to_base64(file_path)
        data_url = f"data:{mime_type};base64,{base64_str}"
        if category == 'image':
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        elif category == 'video':
            content.append({"type": "video_url", "video_url": {"url": data_url}})
        elif category == 'audio':
            content.append({"type": "audio_url", "audio_url": {"url": data_url}})
    return content

def get_filename_from_path(path: str) -> str:
    return os.path.basename(path)

# ===================== LangGraph 节点（重构） =====================
async def chat_node(state: ChatAgentState, config: dict):
    """核心聊天节点：调用大模型生成回答"""
    thread_id = config["configurable"]["thread_id"]
    messages = await build_final_messages(state, thread_id)
    response = await qianFan.ainvoke(messages)
    
    # 清空单次有效的工具结果字段（避免后续对话重复引用）
    return {
        "messages": [response],
        "code_result": None,
        "search_result": None,
        "rag_context": None
    }

async def slice_history_node(state: ChatAgentState, config: dict):
    """
    🔥 历史切片节点：每新增10条消息执行一次
    替代原有的简单摘要节点
    """
    thread_id = config["configurable"]["thread_id"]
    total_messages = len(state["messages"])
    
    # 严格按照需求：每新增10条记录调用一次切片
    if total_messages % 10 == 0 and total_messages > 0:
        print(f"📊 触发历史切片，当前消息总数: {total_messages}")
        try:
            # 后台执行切片，不阻塞主流程
            asyncio.create_task(history_manager.slice_history(thread_id))
        except Exception as e:
            print(f"⚠️ 历史切片执行失败: {e}")
    
    return {}

# ===================== 构建并编译智能体 =====================
def build_chat_agent():
    workflow = StateGraph(ChatAgentState)
    workflow.add_node("chat", chat_node)
    workflow.add_node("slice_history", slice_history_node)
    
    # 执行顺序：聊天 → 检查并执行切片 → 结束
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", "slice_history")
    workflow.add_edge("slice_history", END)
    
    return workflow

# 编译智能体（在初始化函数中完成）
chat_agent = None

# ===================== 导出接口（修改为异步初始化） =====================
async def ensure_agent_initialized():
    """确保智能体和所有资源已初始化"""
    global chat_agent
    if chat_agent is None:
        await init_global_resources()
        chat_agent = build_chat_agent().compile(
            checkpointer=checkpointer,
            name="chat_agent"
        )

async def call_chat_agent(
    user_input: str | list[dict], 
    user_id: str, 
    thread_id: str,
    code_result: Optional[str] = None,
    search_result: Optional[str] = None,
    rag_context: Optional[str] = None
) -> str:
    await ensure_agent_initialized()
    
    config = {"configurable": {"thread_id": thread_id}}
    user_message = HumanMessage(content=user_input)
    
    result = await chat_agent.ainvoke(
        {
            "messages": [user_message], 
            "user_id": user_id,
            "code_result": code_result,
            "search_result": search_result,
            "rag_context": rag_context
        },
        config=config
    )
    return result["messages"][-1].content

async def stream_chat_agent(
    user_input: str | list[dict], 
    user_id: str, 
    thread_id: str,
    code_result: Optional[str] = None,
    search_result: Optional[str] = None,
    rag_context: Optional[str] = None
) -> AsyncIterator[str]:
    await ensure_agent_initialized()
    
    config = {"configurable": {"thread_id": thread_id}}
    user_message = HumanMessage(content=user_input)
    
    async for event in chat_agent.astream(
        {
            "messages": [user_message], 
            "user_id": user_id,
            "code_result": code_result,
            "search_result": search_result,
            "rag_context": rag_context
        },
        config=config, 
        stream_mode="values"
    ):
        if "messages" in event:
            last_msg = event["messages"][-1]
            if last_msg.type == "ai" and last_msg.content:
                yield last_msg.content

# ===================== Gradio 界面（修改初始化逻辑） =====================
def run_gradio_interface():
    import gradio as gr
    
    # 先同步运行初始化（Gradio启动时执行一次）
    asyncio.run(ensure_agent_initialized())
    
    def store_uploaded_file(files):
        return files if files else None
    
    def add_text_and_file_message(chat_history, text_input, stored_files):
        if stored_files:
            content_list = []
            if text_input and text_input.strip():
                content_list.append({"type": "text", "text": text_input.strip()})
            for f in stored_files:
                content_list.append({"type":"file","file":{"path":f.name,"orig_name":get_filename_from_path(f.name)}})
            chat_history.append({"role": "user", "content": content_list})
        else:
            if text_input and text_input.strip():
                chat_history.append({"role": "user", "content": text_input.strip()})
        return chat_history, "", None
    
    def add_audio_message(chat_history, audio_filepath):
        if audio_filepath:
            chat_history.append({"role":"user","content":[{"type":"file","file":{"path":audio_filepath,"mime_type":"audio/wav"}}]})
        return chat_history
    
    async def execute_chain(chat_history, session_id):
        if not chat_history or chat_history[-1]["role"] != "user":
            yield chat_history
            return
        
        last_user_msg = chat_history[-1]
        input_content = last_user_msg["content"]
        model_input_content = None
        text_part = ""
        multimodal_files = []
        
        if isinstance(input_content, list) and len(input_content) > 0:
            for item in input_content:
                if item.get("type") == "text":
                    text_part += item.get("text", "")
                elif item.get("type") == "file" and "file" in item:
                    file_info = item["file"]
                    file_path = file_info.get("path")
                    file_name = file_info.get("orig_name") or get_filename_from_path(file_path)
                    mime_type = file_info.get("mime_type", "") or get_mime_type(file_path)
                    category = get_file_category(file_path, mime_type)
                    if category == 'text':
                        file_content = read_text_file(file_path)
                        text_part += f"\n\n【文件内容：{file_name}】\n{file_content}"
                    else:
                        multimodal_files.append({'path':file_path,'name':file_name,'mime_type':mime_type,'category':category})
            model_input_content = build_multimodal_message(text_part, multimodal_files)
        else:
            model_input_content = str(input_content)
        
        try:
            full_response = ""
            chat_history.append({"role": "assistant", "content": ""})
            async for chunk in stream_chat_agent(model_input_content, user_id=session_id, thread_id=session_id):
                full_response = chunk
                chat_history[-1]["content"] = full_response
                yield chat_history
            print(f"💾 对话已保存到PostgreSQL，thread_id: {session_id}")
        except Exception as e:
            print(f"❌ 调用失败: {e}")
            import traceback
            traceback.print_exc()
            chat_history.append({"role": "assistant", "content": "抱歉，我暂时无法处理这个请求，请稍后再试。"})
            yield chat_history
    
    with gr.Blocks(title="多模态聊天机器人", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 多模态聊天机器人")
        gr.Markdown("支持纯文本、图片、视频、音频和PDF/TXT文件 | 智能历史切片 | 工具与RAG集成")
        user_id = gr.State(value="user_test_001")
        stored_files = gr.State(value=None)
        chatbot = gr.Chatbot(height=600, label="聊天记录", type="messages")
        
        with gr.Row():
            with gr.Column(scale=4):
                input_box = gr.Textbox(placeholder="请输入问题或上传文件...", scale=5, container=False)
                send_btn = gr.Button("发送", min_width=120)
            with gr.Column(scale=1):
                audio_input = gr.Audio(sources=["microphone"], type="filepath", label="语音输入")
                file_upload = gr.File(label="上传文件", file_count="multiple", scale=1)
        
        file_upload.change(fn=store_uploaded_file, inputs=[file_upload], outputs=[stored_files])
        input_box.submit(fn=add_text_and_file_message, inputs=[chatbot,input_box,stored_files], outputs=[chatbot,input_box,stored_files]).then(fn=execute_chain, inputs=[chatbot,user_id], outputs=[chatbot]).then(lambda: None, outputs=[file_upload])
        send_btn.click(fn=add_text_and_file_message, inputs=[chatbot,input_box,stored_files], outputs=[chatbot,input_box,stored_files]).then(fn=execute_chain, inputs=[chatbot,user_id], outputs=[chatbot]).then(lambda: None, outputs=[file_upload])
        audio_input.stop_recording(fn=add_audio_message, inputs=[chatbot,audio_input], outputs=[chatbot]).then(fn=execute_chain, inputs=[chatbot,user_id], outputs=[chatbot]).then(lambda: None, outputs=[audio_input])
    
    demo.queue(max_size=10)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    print("🤖 多模态聊天机器人启动中...")
    run_gradio_interface()