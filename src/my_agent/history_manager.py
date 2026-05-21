import os
import json
from dotenv import load_dotenv
from typing import List, Optional, Set
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Text, Integer, DateTime
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage
)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel, Field
from my_agent.mylm import qianFan

# 加载环境变量
load_dotenv(override=True)
POSTGRES_URI = os.getenv("POSTGRES_URI")
if not POSTGRES_URI:
    raise ValueError("请设置POSTGRES_URI环境变量")

# 转换为异步连接字符串（如果需要）
if POSTGRES_URI.startswith("postgresql://"):
    ASYNC_POSTGRES_URI = POSTGRES_URI.replace("postgresql://", "postgresql+asyncpg://")
else:
    ASYNC_POSTGRES_URI = POSTGRES_URI

# ==========================================
# 1. 数据模型定义
# ==========================================
class SingleRoundSummary(BaseModel):
    """单轮对话摘要结构"""
    user_summary: str = Field(description="用户提问核心内容，1句话")
    assistant_summary: str = Field(description="助手回复核心内容，1句话")

# ==========================================
# 2. 独立数据库表定义
# ==========================================
Base = declarative_base()

class ConversationSlice(Base):
    """存储切片结果的独立表"""
    __tablename__ = "conversation_slices"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), index=True, nullable=False)  # 与LangGraph的thread_id完全一致
    last_processed_round = Column(Integer, default=0)  # 上次处理到的全局轮次编号
    processed_rounds = Column(Text, nullable=False, default="[]")  # 已处理的全局轮次集合
    global_summary = Column(Text, nullable=False, default="[]")  # JSON数组，永远只追加
    middle_summaries = Column(Text, nullable=False, default="[]")  # JSON数组，每次覆盖
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

# 初始化异步数据库引擎和会话工厂
async_engine = create_async_engine(ASYNC_POSTGRES_URI, echo=False)
AsyncSessionLocal = sessionmaker(
    bind=async_engine, 
    class_=AsyncSession, 
    expire_on_commit=False,
    autocommit=False, 
    autoflush=False
)

# 异步创建表
async def create_tables():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# ==========================================
# 3. 核心管理器类
# ==========================================
class AsyncChatHistoryManager:
    def __init__(self, postgres_uri: Optional[str] = None):
        self.postgres_uri = postgres_uri or ASYNC_POSTGRES_URI
        
        # LangGraph 异步Checkpoint读取器（只读）
        self.checkpointer = AsyncPostgresSaver.from_conn_string(self.postgres_uri)
        
        # 初始化摘要链
        self._init_summary_chains()
        
        # 严格按照需求配置
        self.FIRST_CALL_READ_ALL = True  # 第一次调用读取全部
        self.NORMAL_CALL_READ_LATEST = 15  # 后续调用只读取最新15条消息
        self.KEEP_LATEST_FULL_MESSAGES = 2  # 最新2条完整保留
        self.MIDDLE_SLICE_MESSAGES = 3  # 第3-5条（共3条）做第二层摘要
        self.GLOBAL_SLICE_START_MESSAGE = 6  # 第6条开始做全局摘要

    def _init_summary_chains(self):
        """初始化摘要生成链"""
        # 第二层：单轮摘要链
        single_round_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的对话摘要助手。
请严格遵守以下规则：
1. 对给定的「用户提问+助手回答」做摘要
2. 每部分1句话，必须简洁
3. 100%保留关键实体、数字、约束
4. 把代词替换成具体实体
5. 严格按照JSON格式输出"""),
            ("human", """请对以下单轮对话做摘要：
【用户提问】{user_content}
【助手回答】{assistant_content}""")
        ])
        self.single_round_chain = single_round_prompt | qianFan.with_structured_output(
            SingleRoundSummary, method="json_mode"
        )

        # 全局增量摘要链
        global_increment_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的对话摘要助手。
请严格遵守以下规则：
1. 只对给定的新对话内容做摘要
2. 生成1-2句话的简洁摘要
3. 100%保留关键信息
4. 不要修改或引用已有的全局摘要
5. 直接返回摘要文本，不要任何格式"""),
            ("human", """请对以下新的对话内容做摘要：
【新对话内容】
{new_dialogue}""")
        ])
        self.global_increment_chain = global_increment_prompt | qianFan

    # ==========================================
    # 基础操作
    # ==========================================
    async def setup(self):
    # 初始化LangGraph checkpoint表
        async with self.checkpointer._engine.begin() as conn:
            await self.checkpointer.create_tables(conn)
        # 初始化切片管理器自己的表
        await create_tables()

    async def get_messages_for_slicing(self, session_id: str, is_first_call: bool) -> List[BaseMessage]:
        """
        根据调用次数获取需要处理的消息
        :param session_id: 会话ID
        :param is_first_call: 是否是第一次调用
        :return: 需要处理的消息列表（按时间正序，最早的在前）
        """
        config = {"configurable": {"thread_id": session_id}}
        checkpoint = await self.checkpointer.get(config)
        
        if not checkpoint or "channel_values" not in checkpoint:
            return []
        
        all_messages = checkpoint["channel_values"].get("messages", [])
        
        if is_first_call:
            return all_messages
        else:
            return all_messages[-self.NORMAL_CALL_READ_LATEST:] if all_messages else []

    async def get_or_create_slice(self, session_id: str, db: AsyncSession) -> ConversationSlice:
        """获取或创建指定会话的切片记录"""
        result = await db.execute(
            ConversationSlice.__table__.select().where(
                ConversationSlice.session_id == session_id
            )
        )
        slice_record = result.scalar_one_or_none()
        
        if not slice_record:
            slice_record = ConversationSlice(session_id=session_id)
            db.add(slice_record)
            await db.commit()
            await db.refresh(slice_record)
        
        return slice_record

    # ==========================================
    # 核心切片逻辑（修复后）
    # ==========================================
    async def slice_history(self, session_id: str) -> dict:
        """
        执行历史切片操作
        :param session_id: 会话ID（与LangGraph的thread_id完全一致）
        :return: 切片结果
        """
        async with AsyncSessionLocal() as db:
            # 1. 判断是否是第一次调用
            slice_record = await self.get_or_create_slice(session_id, db)
            is_first_call = len(json.loads(slice_record.global_summary)) == 0
            
            # 2. 根据调用次数获取需要处理的消息
            messages = await self.get_messages_for_slicing(session_id, is_first_call)
            if len(messages) < 2:
                return {"status": "skipped", "reason": "消息数量不足2条"}
            
            # 3. 获取已处理的全局轮次集合
            processed_rounds: Set[int] = set(json.loads(slice_record.processed_rounds))
            
            # 4. 计算全局消息索引（关键修复！）
            # 先获取全部消息的总数，以计算正确的全局索引
            config = {"configurable": {"thread_id": session_id}}
            full_checkpoint = await self.checkpointer.get(config)
            all_messages = full_checkpoint["channel_values"].get("messages", []) if full_checkpoint else []
            total_messages = len(all_messages)
            start_global_index = total_messages - len(messages)
            
            # 5. 将消息配对成完整轮次（带全局索引和轮次编号）
            rounds = []
            i = 0
            while i < len(messages):
                global_index = start_global_index + i
                if (isinstance(messages[i], HumanMessage) 
                    and i+1 < len(messages) 
                    and isinstance(messages[i+1], AIMessage)):
                    # 全局轮次编号 = (全局消息索引 // 2) + 1
                    global_round_num = (global_index // 2) + 1
                    rounds.append({
                        "user_msg": messages[i],
                        "ai_msg": messages[i+1],
                        "global_round_num": global_round_num,
                        "start_global_index": global_index,
                        "end_global_index": global_index + 1
                    })
                    i += 2
                else:
                    i += 1
            
            if not rounds:
                return {"status": "skipped", "reason": "没有完整的对话轮次"}
            
            # 6. 按全局轮次倒序处理（最新的在前）
            rounds_sorted = sorted(rounds, key=lambda x: x["global_round_num"], reverse=True)
            
            new_middle_summaries = []
            new_global_entries = []
            newly_processed_rounds = []
            
            for round_data in rounds_sorted:
                round_num = round_data["global_round_num"]
                user_msg = round_data["user_msg"]
                ai_msg = round_data["ai_msg"]
                start_idx = round_data["start_global_index"]
                
                # 跳过已经处理过的轮次
                if round_num in processed_rounds:
                    continue
                
                # --------------------------
                # 第一层：最新2条消息：完整保留，不处理
                # --------------------------
                if start_idx >= total_messages - self.KEEP_LATEST_FULL_MESSAGES:
                    continue
                
                # --------------------------
                # 第二层：第3-5条消息：逐轮摘要
                # --------------------------
                elif start_idx >= total_messages - self.GLOBAL_SLICE_START_MESSAGE:
                    try:
                        summary = await self.single_round_chain.ainvoke({
                            "user_content": user_msg.content,
                            "assistant_content": ai_msg.content
                        })
                        new_middle_summaries.append({
                            "round_num": round_num,
                            "user_summary": summary.user_summary,
                            "assistant_summary": summary.assistant_summary
                        })
                        newly_processed_rounds.append(round_num)
                    except Exception as e:
                        print(f"⚠️ 第{round_num}轮第二层摘要失败: {e}")
                        new_middle_summaries.append({
                            "round_num": round_num,
                            "user_summary": user_msg.content[:100] + "...",
                            "assistant_summary": ai_msg.content[:100] + "..."
                        })
                        newly_processed_rounds.append(round_num)
                
                # --------------------------
                # 第三层：第6条及以后：全局增量摘要
                # --------------------------
                else:
                    try:
                        dialogue_text = f"用户：{user_msg.content}\n助手：{ai_msg.content}"
                        result = await self.global_increment_chain.ainvoke({"new_dialogue": dialogue_text})
                        summary = result.content.strip()
                        new_global_entries.append(f"【第{round_num}轮】{summary}")
                        newly_processed_rounds.append(round_num)
                    except Exception as e:
                        print(f"⚠️ 第{round_num}轮全局摘要失败: {e}")
                        new_global_entries.append(f"【第{round_num}轮】{user_msg.content[:50]}... | {ai_msg.content[:50]}...")
                        newly_processed_rounds.append(round_num)
            
            # 7. 更新切片记录
            # 第二层：每次覆盖，只保留最新的
            if new_middle_summaries:
                # 按轮次正序排列，方便阅读
                new_middle_summaries_sorted = sorted(new_middle_summaries, key=lambda x: x["round_num"])
                slice_record.middle_summaries = json.dumps(new_middle_summaries_sorted, ensure_ascii=False)
            
            # 第三层：只增量追加，不修改已有内容
            if new_global_entries:
                existing_global = json.loads(slice_record.global_summary)
                existing_global.extend(new_global_entries)
                slice_record.global_summary = json.dumps(existing_global, ensure_ascii=False)
            
            # 更新已处理的轮次集合
            processed_rounds.update(newly_processed_rounds)
            slice_record.processed_rounds = json.dumps(list(processed_rounds), ensure_ascii=False)
            
            # 更新最后处理的轮次
            if rounds:
                slice_record.last_processed_round = max(r["global_round_num"] for r in rounds)
            
            slice_record.updated_at = datetime.now()
            await db.commit()
            
            print(f"✅ 切片完成，会话ID: {session_id}")
            print(f"📊 调用类型: {'第一次调用' if is_first_call else '后续调用'}")
            print(f"📊 读取消息数: {len(messages)}")
            print(f"📊 处理了 {len(new_middle_summaries)} 轮第二层摘要")
            print(f"📊 新增了 {len(new_global_entries)} 条全局摘要")
            print(f"📍 最后处理轮次: {slice_record.last_processed_round}")
            
            return {
                "status": "success",
                "session_id": session_id,
                "is_first_call": is_first_call,
                "messages_read": len(messages),
                "last_processed_round": slice_record.last_processed_round,
                "new_middle_summaries": new_middle_summaries,
                "new_global_entries": new_global_entries,
                "total_global_entries": len(json.loads(slice_record.global_summary)),
                "total_middle_summaries": len(json.loads(slice_record.middle_summaries))
            }

    # ==========================================
    # 切片结果读取接口
    # ==========================================
    async def get_current_slices(self, session_id: str) -> dict:
        """获取当前会话的所有切片内容"""
        async with AsyncSessionLocal() as db:
            slice_record = await self.get_or_create_slice(session_id, db)
            
            return {
                "session_id": session_id,
                "last_processed_round": slice_record.last_processed_round,
                "global_summary": json.loads(slice_record.global_summary),
                "middle_summaries": json.loads(slice_record.middle_summaries),
                "updated_at": slice_record.updated_at
            }

    async def build_final_context(self, session_id: str) -> List[BaseMessage]:
        """构建最终发送给模型的上下文"""
        slices = await self.get_current_slices(session_id)
        
        # 只读取最新的2条完整消息（性能优化）
        config = {"configurable": {"thread_id": session_id}}
        checkpoint = await self.checkpointer.get(config)
        all_messages = checkpoint["channel_values"].get("messages", []) if checkpoint else []
        latest_messages = all_messages[-2:] if all_messages else []
        
        context = []
        
        # 1. 添加全局摘要（所有历史的增量摘要）
        if slices["global_summary"]:
            global_text = "【全局对话摘要】\n" + "\n".join(slices["global_summary"])
            context.append(SystemMessage(content=global_text))
        
        # 2. 添加第二层摘要（最新的3-5条）
        if slices["middle_summaries"]:
            middle_text = "【近期对话摘要】\n"
            for s in slices["middle_summaries"]:
                middle_text += f"第{s['round_num']}轮：用户-{s['user_summary']} | 助手-{s['assistant_summary']}\n"
            context.append(SystemMessage(content=middle_text))
        
        # 3. 添加最新的2条完整消息
        context.extend(latest_messages)
        
        return context

    # ==========================================
    # 工具方法
    # ==========================================
    async def clear_slices(self, session_id: str):
        """清空指定会话的所有切片记录"""
        async with AsyncSessionLocal() as db:
            await db.execute(
                ConversationSlice.__table__.delete().where(
                    ConversationSlice.session_id == session_id
                )
            )
            await db.commit()
            print(f"🗑️ 已清空会话 {session_id} 的所有切片记录")

    async def close(self):
        """关闭所有资源"""
        await self.checkpointer.close()
        await async_engine.dispose()