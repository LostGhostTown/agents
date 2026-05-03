from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from pydantic import BaseModel, Field
from typing import List
from my_agent.mylm import qianFan

# ==========================================
# 1. 核心配置
# ==========================================
LATEST_KEEP_ROUNDS = 2  # 第一层：最新2轮完整保留
MIDDLE_SUMMARY_ROUNDS = 3  # 第二层：中期3轮逐轮摘要
EARLY_SUMMARY_THRESHOLD = 5  # 超过5轮触发第三层全局摘要

# ==========================================
# 2. 定义Pydantic结构（锁死输出格式）
# ==========================================
class SingleRoundSummary(BaseModel):
    """单轮对话摘要结构"""
    user_summary: str = Field(description="用户提问核心内容，1句话")
    assistant_summary: str = Field(description="助手回复核心内容，1句话")

class GlobalEarlyHistorySummary(BaseModel):
    """早期对话全局摘要结构"""
    user_core_info: str = Field(description="用户固定核心信息，1-2句话")
    dialogue_core_goal: str = Field(description="对话核心任务，1句话")
    closed_matters: List[str] = Field(description="已闭环事项，每条不超过20字")
    long_term_constraints: List[str] = Field(description="长期约束要求，每条不超过20字")

# ==========================================
# 3. 构建摘要Chain
# ==========================================
# 第二层：单轮摘要Chain
second_level_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的对话摘要助手。
请严格遵守以下规则：
1. 对给定的「用户提问+助手回答」做摘要
2. 每部分1-2句话，必须简洁
3. 100%保留关键实体、数字、约束
4. 把代词替换成具体实体
5. 严格按照JSON格式输出"""),
    ("human", """请对以下单轮对话做摘要：
【用户提问】{user_content}
【助手回答】{assistant_content}""")
])
second_level_chain = second_level_prompt | qianFan.with_structured_output(SingleRoundSummary, method="json_mode")

# 第三层：全局摘要Chain
third_level_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的对话全局摘要助手。
请严格遵守以下规则：
1. 对早期多轮对话做全局合并摘要，禁止分轮次
2. 仅保留长期有效信息，剔除临时内容
3. 100%保留用户身份、技术栈、核心目标
4. 严格按照JSON格式输出"""),
    ("human", """请对以下早期对话做全局摘要：
【完整早期对话】
{all_early_dialogue}""")
])
third_level_chain = third_level_prompt | qianFan.with_structured_output(GlobalEarlyHistorySummary, method="json_mode")

# ==========================================
# 4. 核心：分层处理函数（不修改原始历史，仅返回处理后的列表）
# ==========================================
def compress_chat_history(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    对聊天历史做分层剪辑/摘要
    :param messages: 原始历史消息列表
    :return: 处理后的消息列表（不修改原始数据）
    """
    # 1. 把消息配对成完整轮次
    rounds = []
    i = 0
    while i < len(messages):
        if isinstance(messages[i], HumanMessage) and i+1 < len(messages) and isinstance(messages[i+1], AIMessage):
            rounds.append((messages[i], messages[i+1]))
            i += 2
        else:
            i += 1
    
    total_rounds = len(rounds)
    if total_rounds <= LATEST_KEEP_ROUNDS:
        return messages  # 轮数太少，直接返回原始消息

    # --------------------------
    # 第一层：最新N轮，完整保留
    # --------------------------
    latest_rounds = rounds[-LATEST_KEEP_ROUNDS:]
    latest_messages = []
    for u, a in latest_rounds:
        latest_messages.append(u)
        latest_messages.append(a)

    remaining_rounds = rounds[:-LATEST_KEEP_ROUNDS]
    remaining_total = len(remaining_rounds)
    middle_summary_messages = []
    early_summary_messages = []

    # --------------------------
    # 第二层：中期M轮，逐轮摘要
    # --------------------------
    if remaining_total > 0:
        middle_rounds = remaining_rounds[-MIDDLE_SUMMARY_ROUNDS:] if remaining_total > MIDDLE_SUMMARY_ROUNDS else remaining_rounds
        early_rounds = remaining_rounds[:-MIDDLE_SUMMARY_ROUNDS] if remaining_total > MIDDLE_SUMMARY_ROUNDS else []

        if middle_rounds:
            middle_summaries = []
            for idx, (user_msg, ai_msg) in enumerate(middle_rounds):
                try:
                    single_summary = second_level_chain.invoke({
                        "user_content": user_msg.content,
                        "assistant_content": ai_msg.content
                    })
                    middle_summaries.append(f"【中期第{total_rounds - len(middle_rounds) + idx +1}轮】用户：{single_summary.user_summary} | 助手：{single_summary.assistant_summary}")
                except Exception as e:
                    middle_summaries.append(f"【中期第{total_rounds - len(middle_rounds) + idx +1}轮】用户：{user_msg.content[:50]}... | 助手：{ai_msg.content[:50]}...")
            
            middle_summary_text = "【中期对话摘要】\n" + "\n".join(middle_summaries)
            middle_summary_messages = [HumanMessage(content=middle_summary_text)]

        # --------------------------
        # 第三层：早期轮次，全局摘要
        # --------------------------
        if early_rounds and len(early_rounds) >= EARLY_SUMMARY_THRESHOLD:
            early_dialogue_text = ""
            for idx, (user_msg, ai_msg) in enumerate(early_rounds):
                early_dialogue_text += f"第{idx+1}轮\n用户：{user_msg.content}\n助手：{ai_msg.content}\n\n"
            
            try:
                global_summary = third_level_chain.invoke({"all_early_dialogue": early_dialogue_text})
                early_summary_text = f"""【早期对话全局摘要】
1. 用户核心信息：{global_summary.user_core_info}
2. 对话核心目标：{global_summary.dialogue_core_goal}
3. 已闭环事项：{'; '.join(global_summary.closed_matters) if global_summary.closed_matters else '无'}
4. 长期约束要求：{'; '.join(global_summary.long_term_constraints) if global_summary.long_term_constraints else '无'}
"""
                early_summary_messages = [SystemMessage(content=early_summary_text)]
            except Exception as e:
                early_summary_messages = []

    # 最终拼接
    return early_summary_messages + middle_summary_messages + latest_messages

# ==========================================
# 5. 构建主聊天Chain（整合摘要功能）
# ==========================================
system_prompt = SystemMessage(content="你是一个沟通助手，根据用户输入、聊天历史，调用合适的工具，并给出相应的答案。如果有【早期/中期对话摘要】，请参考摘要理解上下文。")

def build_final_messages(input_data: dict) -> List[BaseMessage]:
    """手动构建最终消息列表，彻底避免模板格式化破坏多模态结构"""
    messages = [system_prompt]
    # 加入压缩后的历史
    messages.extend(input_data["chat_history"])
    # 加入当前用户输入（直接保留原始多模态结构，不做任何格式化）
    messages.append(HumanMessage(content=input_data["input"]))
    return messages

main_chain = (
    RunnablePassthrough.assign(
        chat_history=lambda x: compress_chat_history(x["chat_history"])
    )
    | RunnableLambda(build_final_messages)
    | qianFan
)

# ==========================================
# 6. 封装带历史的Chain
# ==========================================
def get_chat_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection='sqlite:///chat_history.db'
    )

chain_with_history = RunnableWithMessageHistory(
    main_chain,
    get_chat_history,
    input_messages_key='input',
    history_messages_key='chat_history'
)
#7. 可视化页面
import os
import base64
from PyPDF2 import PdfReader
def get_mime_type(file_path: str) -> str:
    """兼容图片/视频/音频全格式MIME类型，和OpenAI规范对齐"""
    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    # 图片格式
    if ext in ['jpg', 'jpeg']:
        return 'image/jpeg'
    elif ext == 'png':
        return 'image/png'
    elif ext == 'gif':
        return 'image/gif'
    elif ext == 'webp':
        return 'image/webp'
    elif ext == 'bmp':
        return 'image/bmp'
    # 视频格式（核心补充，和模型支持对齐）
    elif ext == 'mp4':
        return 'video/mp4'
    elif ext == 'avi':
        return 'video/x-msvideo'
    elif ext == 'mov':
        return 'video/quicktime'
    elif ext == 'mkv':
        return 'video/x-matroska'
    elif ext == 'flv':
        return 'video/x-flv'
    elif ext == 'webm':
        return 'video/webm'
    # 音频格式（和视频格式规范对齐）
    elif ext == 'mp3':
        return 'audio/mpeg'
    elif ext == 'wav':
        return 'audio/wav'
    elif ext == 'flac':
        return 'audio/flac'
    elif ext == 'aac':
        return 'audio/aac'
    elif ext == 'm4a':
        return 'audio/m4a'
    # 兜底
    return 'application/octet-stream'

def get_file_category(file_path: str, mime_type: str = "") -> str:
    """
    判断文件类型：
    - image: 图片
    - video: 视频
    - audio: 音频
    - text: 文字文件 (pdf, txt, docx等)
    - other: 其他
    """
    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    
    # 先通过 MIME 类型判断
    if mime_type:
        if mime_type.startswith('image/'):
            return 'image'
        elif mime_type.startswith('video/'):
            return 'video'
        elif mime_type.startswith('audio/'):
            return 'audio'
    
    # 通过扩展名判断
    if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']:
        return 'image'
    elif ext in ['mp4', 'avi', 'mov', 'mkv', 'flv']:
        return 'video'
    elif ext in ['mp3', 'wav', 'flac', 'aac', 'm4a']:
        return 'audio'
    elif ext in ['pdf', 'txt', 'md', 'docx', 'doc', 'rtf']:
        return 'text'
    return 'other'

def file_to_base64(file_path: str) -> str:
    """文件转 Base64"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def read_text_file(file_path: str) -> str:
    """读取文字文件内容 (PDF/TXT)"""
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
        # 可以扩展 docx 等其他格式
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        content = f"[文件读取失败: {str(e)}]"
    
    # 限制长度，避免过长
    if len(content) > 10000:
        content = content[:10000] + "\n\n[文件过长，已截断...]"
    
    return content

def build_multimodal_message(text_part: str, files_data: List[dict]) -> List[dict]:
    """
    构建标准OpenAI多模态格式，和图片/视频/音频格式完全对齐
    规则：
    1. 文本在前，多媒体在后（千帆强制要求）
    2. 图片/视频/音频 格式完全对齐，仅修改type和对应key
    3. 100%保留原生多模态结构，不做任何降级处理
    4. 返回的content列表可直接传入HumanMessage，适配你修改后的主Chain
    """
    content = []
    # 第一步：先加文本（必须在前，千帆多模态接口强制要求）
    if text_part and text_part.strip():
        content.append({"type": "text", "text": text_part.strip()})
    # 兜底：无文本时给默认提示词，避免纯多媒体输入报错
    elif not text_part.strip() and len(files_data) > 0:
        content.append({"type": "text", "text": "请基于用户上传的文件内容进行回答"})
    
    # 第二步：按格式添加多媒体文件，和图片格式完全对齐
    for file_data in files_data:
        category = file_data['category']
        file_path = file_data['path']
        mime_type = file_data['mime_type']
        file_name = file_data['name']
        
        # 通用base64转码，和图片逻辑完全一致
        base64_str = file_to_base64(file_path)
        data_url = f"data:{mime_type};base64,{base64_str}"
        
        # 图片：标准OpenAI格式
        if category == 'image':
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })
            print(f"✅ 成功加载图片：{file_name}，base64长度：{len(base64_str)}")
        
        # 视频：和图片格式完全对齐，仅修改type和key
        elif category == 'video':
            content.append({
                "type": "video_url",
                "video_url": {"url": data_url}
            })
            print(f"✅ 成功加载视频：{file_name}，base64长度：{len(base64_str)}")
        
        # 音频：按照你的要求，和视频传入方式完全一致
        elif category == 'audio':
            content.append({
                "type": "audio_url",
                "audio_url": {"url": data_url}
            })
            print(f"✅ 成功加载音频：{file_name}，base64长度：{len(base64_str)}")
    
    return content

def get_filename_from_path(path: str) -> str:
    return os.path.basename(path)

# ==========================================
# 7. 文件暂存 + 发送逻辑
# ==========================================
def store_uploaded_file(files):
    """文件上传只暂存，不发送"""
    if not files:
        return None
    return files

def add_text_and_file_message(chat_history, text_input, stored_files):
    """点击发送：组合 文字 + 暂存文件 加入聊天框"""
    if stored_files:
        content_list = []
        if text_input and text_input.strip():
            content_list.append({"type":"text", "text":text_input.strip()})
        for f in stored_files:
            content_list.append({
                "type": "file",
                "file": {
                    "path": f.name,
                    "orig_name": get_filename_from_path(f.name)
                }
            })
        chat_history.append({"role":"user", "content":content_list})
    else:
        if text_input and text_input.strip():
            chat_history.append({'role':'user','content': text_input.strip()})
    
    return chat_history, "", None

def add_audio_message(chat_history, audio_filepath):
    """音频保持原有逻辑不变"""
    if audio_filepath:
        chat_history.append({
            "role": "user",
            "content": [{
                "type": "file",
                "file": {
                    "path": audio_filepath,
                    "mime_type": "audio/wav"
                }
            }]
        })
    return chat_history

# ==========================================
# 8. 核心执行链（支持智能文件处理）
# ==========================================
def execute_chain(chat_history, session_id):
    if not chat_history or chat_history[-1]["role"] != "user":
        return chat_history
    
    last_user_msg = chat_history[-1]
    input_content = last_user_msg["content"]
    db_history = get_chat_history(session_id)

    model_input_content = None  # 现在直接是 content 列表/字符串
    history_save_content = ""
    text_part = ""
    multimodal_files = []

    # ==========================================
    # 解析输入
    # ==========================================
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
                    print(f"📄 读取文字文件: {file_name}")
                    file_content = read_text_file(file_path)
                    text_part += f"\n\n【文件内容：{file_name}】\n{file_content}"
                else:
                    multimodal_files.append({
                        'path': file_path,
                        'name': file_name,
                        'mime_type': mime_type,
                        'category': category
                    })
        
        # 构建多模态输入（现在直接返回 List[dict]）
        model_input_content = build_multimodal_message(text_part, multimodal_files)
        
        # 构建历史保存内容
        history_parts = []
        if text_part:
            history_parts.append(text_part)
        for f in multimodal_files:
            history_parts.append(f"[上传{ f['category'] }文件：{ f['name'] }]")
        history_save_content = "\n".join(history_parts)
        
    else:
        # 纯文本输入
        input_text = str(input_content)
        model_input_content = input_text
        history_save_content = input_text
        print(f"✅ 纯文本输入，内容：{input_text[:50]}...")

    # ==========================================
    # 调用模型（核心修复）
    # ==========================================
    try:
        raw_history_messages = db_history.messages
        
        # 🔥 修复：直接打印 model_input_content，它已经是 content 了
        print(f"🔍 最终传给模型的输入：{model_input_content}")
        
        # 🔥 直接传入 model_input_content，主Chain会处理好
        resp = main_chain.invoke({
            "input": model_input_content,
            "chat_history": raw_history_messages
        })
        ai_response = resp.content
    except Exception as e:
        print(f"❌ 模型调用失败：{e}")
        import traceback
        traceback.print_exc()
        ai_response = "抱歉，我暂时无法处理这个请求，请稍后再试。"

    # ==========================================
    # 保存历史
    # ==========================================
    db_history.add_user_message(history_save_content)
    db_history.add_ai_message(ai_response)
    print(f"💾 历史记录已保存：{history_save_content}")

    # 前端更新
    chat_history.append({'role': 'assistant', 'content': ai_response})
    return chat_history

import gradio as gr
with gr.Blocks(title="多模态聊天机器人") as demo:
    user_id = gr.State(value='user_test_001')
    stored_files = gr.State(value=None)
    chatbot = gr.Chatbot(height=500,label='聊天记录')
    with gr.Row():
        with gr.Column(scale=4):
            input=gr.Textbox(placeholder='请输入...',scale=5,container=False)
            send_btn=gr.Button('send', min_width=120)
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=['microphone'],type= 'filepath')
            file_upload = gr.File(
            label="上传文件",  # 组件说明文字
            file_count="multiple",  # "single"单文件 / "multiple"多文件
            scale=1  # 宽度占比
        )
    
    file_upload.change(
        fn=store_uploaded_file,
        inputs=[file_upload],
        outputs=[stored_files]
    )

    # 文字回车 / 点发送：组合文字+暂存文件发送
    submit_flow = (
        input.submit(
            fn=add_text_and_file_message,
            inputs=[chatbot, input, stored_files],
            outputs=[chatbot, input, stored_files]
        ).then(
            fn=execute_chain,
            inputs=[chatbot, user_id],
            outputs=[chatbot]
        ).then(
            fn=lambda: None,
            outputs=[file_upload]
        )
    )

    send_btn.click(
        fn=add_text_and_file_message,
        inputs=[chatbot, input, stored_files],
        outputs=[chatbot, input, stored_files]
    ).then(
        fn=execute_chain,
        inputs=[chatbot, user_id],
        outputs=[chatbot]
    ).then(
        fn=lambda: None,
        outputs=[file_upload]
    )

    # 音频录音逻辑不变
    audio_input.stop_recording(
        fn=add_audio_message,
        inputs=[chatbot, audio_input],
        outputs=[chatbot]
    ).then(
        fn=execute_chain,
        inputs=[chatbot, user_id],
        outputs=[chatbot]
    ).then(
        fn=lambda: None,
        outputs=[audio_input]
    )


if __name__ == "__main__":
    #print("🤖 带上下文摘要的聊天机器人已启动（输入 '退出' 结束）")
    #session_id = "user_test_001"
    
    # 先清空之前的测试数据
    #test_history = get_chat_history(session_id)
    #test_history.clear()
    
    """while True:
        user_input = input("\n你: ")
        if user_input == "退出":
            print("🤖 再见！")
            break
        
        # 打印原始历史长度，方便对比
        raw_history = get_chat_history(session_id).messages
        print(f"📊 原始历史轮数：{len(raw_history)//2} 轮")
        
        resp = chain_with_history.invoke(
            {'input': user_input},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"🤖: {resp.content}")"""
    demo.queue()
    demo.launch(theme=gr.themes.Soft())