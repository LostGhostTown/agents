from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from pydantic import BaseModel, Field
from typing import List
from my_agent.mylm import qianFan  # 导入你的模型实例



#可以考虑替换为postgresql并使用checkpoint



# ==========================================
# 1. 历史压缩配置（可根据需求调整）
# ==========================================
LATEST_KEEP_ROUNDS = 2  # 第一层：最新N轮完整保留
MIDDLE_SUMMARY_ROUNDS = 3  # 第二层：中期M轮逐轮摘要
EARLY_SUMMARY_THRESHOLD = 5  # 超过K轮触发第三层全局摘要

# ==========================================
# 2. 摘要输出格式定义（锁死JSON结构）
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
# 3. 核心历史管理器类
# ==========================================
class ChatHistoryManager:
    def __init__(self, db_path: str = "sqlite:///chat_history.db"):
        """
        初始化聊天历史管理器
        :param db_path: SQLite数据库路径
        """
        self.db_path = db_path
        self._init_summary_chains()

    def _init_summary_chains(self):
        """初始化两个摘要Chain"""
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
        self.second_level_chain = second_level_prompt | qianFan.with_structured_output(
            SingleRoundSummary, method="json_mode"
        )

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
        self.third_level_chain = third_level_prompt | qianFan.with_structured_output(
            GlobalEarlyHistorySummary, method="json_mode"
        )

    def get_chat_history(self, session_id: str) -> SQLChatMessageHistory:
        """获取指定会话的历史记录对象"""
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=self.db_path
        )

    def compress_chat_history(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        对聊天历史做分层剪辑/摘要（核心算法）
        :param messages: 原始历史消息列表
        :return: 处理后的压缩消息列表
        """
        # 1. 把消息配对成完整轮次
        rounds = []
        i = 0
        while i < len(messages):
            if (isinstance(messages[i], HumanMessage) 
                and i+1 < len(messages) 
                and isinstance(messages[i+1], AIMessage)):
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
            middle_rounds = (remaining_rounds[-MIDDLE_SUMMARY_ROUNDS:] 
                            if remaining_total > MIDDLE_SUMMARY_ROUNDS 
                            else remaining_rounds)
            early_rounds = (remaining_rounds[:-MIDDLE_SUMMARY_ROUNDS] 
                           if remaining_total > MIDDLE_SUMMARY_ROUNDS 
                           else [])

            if middle_rounds:
                middle_summaries = []
                for idx, (user_msg, ai_msg) in enumerate(middle_rounds):
                    try:
                        single_summary = self.second_level_chain.invoke({
                            "user_content": user_msg.content,
                            "assistant_content": ai_msg.content
                        })
                        middle_summaries.append(
                            f"【中期第{total_rounds - len(middle_rounds) + idx +1}轮】"
                            f"用户：{single_summary.user_summary} | 助手：{single_summary.assistant_summary}"
                        )
                    except Exception as e:
                        middle_summaries.append(
                            f"【中期第{total_rounds - len(middle_rounds) + idx +1}轮】"
                            f"用户：{user_msg.content[:50]}... | 助手：{ai_msg.content[:50]}..."
                        )
                
                middle_summary_text = "【中期对话摘要】\n" + "\n".join(middle_summaries)
                middle_summary_messages = [HumanMessage(content=middle_summary_text)]

            # --------------------------
            # 第三层：早期轮次，全局摘要
            # --------------------------
            if early_rounds and len(early_rounds) >= EARLY_SUMMARY_THRESHOLD:
                early_dialogue_text = ""
                for idx, (user_msg, ai_msg) in enumerate(early_rounds):
                    early_dialogue_text += (
                        f"第{idx+1}轮\n"
                        f"用户：{user_msg.content}\n"
                        f"助手：{ai_msg.content}\n\n"
                    )
                
                try:
                    global_summary = self.third_level_chain.invoke({
                        "all_early_dialogue": early_dialogue_text
                    })
                    early_summary_text = f"""【早期对话全局摘要】
1. 用户核心信息：{global_summary.user_core_info}
2. 对话核心目标：{global_summary.dialogue_core_goal}
3. 已闭环事项：{'; '.join(global_summary.closed_matters) if global_summary.closed_matters else '无'}
4. 长期约束要求：{'; '.join(global_summary.long_term_constraints) if global_summary.long_term_constraints else '无'}
"""
                    early_summary_messages = [SystemMessage(content=early_summary_text)]
                except Exception as e:
                    early_summary_messages = []

        # 最终拼接：早期摘要 → 中期摘要 → 最新完整对话
        return early_summary_messages + middle_summary_messages + latest_messages