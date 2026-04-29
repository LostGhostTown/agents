from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from my_agent.mylm import qianFan

prompt = ChatPromptTemplate.from_messages(#聊天提示结构
    [('system', '你是一个沟通助手，你需要根据用户的输入，提供的聊天历史记录包含用户和助手的对话，进行智能处理，调用合适的工具，并给出相应的答案。'),
    MessagesPlaceholder(variable_name="chat_history", optional=True),#聊天历史记录占位符，optional=True表示可选
    ('human', '{input}'),#用户输入占位符
    MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),#agent_scratchpad(智能体草稿本,储存思考过程)占位符，optional=True表示可选
    ]
)

chain = prompt | qianFan

#存储聊天记录 内存或者数据库
from langchain_core.chat_history  import InMemoryChatMessageHistory
history = {}#key为会话id，value为聊天历史记录
def get_chat_history(session_id:str):
    """从内存中的历史记录中获取指定会话id的聊天历史记录"""
    if session_id not in history:
        history[session_id] = InMemoryChatMessageHistory()
    return history[session_id]


# “自动对话历史记忆” 
from langchain_core.runnables.history import RunnableWithMessageHistory
chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key='input',
    history_messages_key='chat_history'
)

result = chain_with_message_history.invoke({'input': '你好，我是Y'}, config={"configurable":{"session_id": "1234567890"}})
print(result)

result = chain_with_message_history.invoke({'input': '你好，我是谁？'}, config={"configurable":{"session_id": "1234567890"}})
print(result)