import bs4

# --- 文档加载与切割 ---
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # 如果版本较旧，可替换为: from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 向量模型与数据库 ---

from langchain_community.vectorstores import FAISS

# --- 链式调用 (Chains) 相关 ---、
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

# --- 运行时与历史记录管理 ---
from langchain_core.runnables.history import RunnableWithMessageHistory

# (以下为你代码中已经包含的本地及核心模块，为了规范建议一并整理到顶部)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from my_agent.mylm import qianFan
from my_agent.chat_history_manager import ChatHistoryManager
from embeddings_demo import QianfanMultiModalEmbeddings
def create_dense_db_basic():
    """基础优化版：调整切割参数，优化加载配置"""
    # 1. 优化Web加载器，添加反爬头，更精准过滤内容
    loader = WebBaseLoader(
        web_path=('https://lilianweng.github.io/posts/2023-06-23-agent/'),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title")  # 只保留核心内容，去掉header/footer
            )
        ),
        requests_kwargs={
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
            "timeout": 10
        }
    )
    docs_list = loader.load()

    # 2. 优化切割参数：技术文章适合更大的chunk，增加重叠比例
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # 技术文章单chunk建议1500-2000字符
        chunk_overlap=300,  # 重叠比例20%，避免上下文断裂
        separators=["\n\n", "\n", ". ", " ", ""],  # 优先按段落、句子切割
        keep_separator=True  # 保留分隔符，保持语义完整
    )
    splits = text_splitter.split_documents(docs_list)

    # 3. 简单内容清洗
    for doc in splits:
        doc.page_content = doc.page_content.strip().replace("\n\n\n", "\n\n")

    # 4. 构建向量库
    embeddings = QianfanMultiModalEmbeddings()
    vector_store = FAISS.from_documents(splits, embeddings)
    vector_store.save_local("./agent_db_basic")
    print(f"✅ 基础版向量库构建完成，共 {len(splits)} 个chunk")
    return vector_store
#create_dense_db_basic() #创建向量数据库
#将带有聊天历史的问题转化为独立问题
from my_agent.mylm import qianFan
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
contextualize_q_system_prompt=(
    "给定聊天历史和最新的用户问题（可能引用聊天历史中的上下文，"
    "将其重新表述为一个独立的问题(不需要聊天历史记录也能理解)."
    "不要回答问题，只需在需要时重新表述问题，否则保持原样。"
)

contextualize_q_prompt=ChatPromptTemplate.from_messages(
    [
        ("system",contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ]
)
embeddings = QianfanMultiModalEmbeddings()
local_vector_store = FAISS.load_local(
    folder_path="./agent_db_basic", 
    embeddings=embeddings, 
    allow_dangerous_deserialization=True  
)
retriever = local_vector_store.as_retriever(search_kwargs={'k': 3})
#创建上下文感知的检索器
history_aware_retriever=create_history_aware_retriever(
    qianFan,
    retriever,
    contextualize_q_prompt
)

#RAG
system_prompt=(
    "你是一个问答任务助手。"
    "使用以下检索到的上下文来回答问题。"
    "如果不知道答案，就说不知道。"
    "回答最多三句话，保持简洁。"
    "\n\n"
    "{context}"
)

qa_prompt=ChatPromptTemplate.from_messages(
[
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
]
)
question_chain=create_stuff_documents_chain(qianFan,qa_prompt)
rag_chain=create_retrieval_chain(history_aware_retriever,question_chain)

from my_agent.chat_history_manager import ChatHistoryManager
chat_history_manager = ChatHistoryManager()

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    chat_history_manager.get_chat_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)
resp1=conversational_rag_chain.invoke(
    {"input":"我是谁?"},
    config={"configurable":{"session_id":"user_test_001"}}
)
resp2=conversational_rag_chain.invoke(
    {"input":"What is Task Decomposition?"},
    config={"configurable":{"session_id":"user_test_001"}}
)
resp3=conversational_rag_chain.invoke(
    {"input":"What are common ways of doing it?"},
    config={"configurable":{"session_id":"user_test_001"}}
)
print(resp1['answer'])
print(resp2['answer'])
print(resp3['answer'])