"""
纯向量检索工具
功能：接收用户问题 → 文本向量化 → FAISS库检索相似内容 → 返回上下文文本
输出：干净的文本内容，直接用于主程序拼接提示词
"""
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 导入你已实现的核心组件
from .embeddings_demo import QianfanMultiModalEmbeddings
from .cos_distance import cosin_distance

# ===================== 固定配置 =====================
# 向量数据库本地路径
VECTOR_DB_PATH = "./agent_db_basic"
# 检索返回的最大条数
TOP_K = 3
# 初始化嵌入模型（全局单例，避免重复初始化）
embeddings = QianfanMultiModalEmbeddings()

# ===================== 向量库加载 =====================
def load_vector_db() -> FAISS:
    """
    加载本地FAISS向量数据库
    :return: FAISS向量库实例
    """
    try:
        db = FAISS.load_local(
            folder_path=VECTOR_DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        raise RuntimeError(f"向量库加载失败，请先执行create_dense_db_basic()创建库：{str(e)}")

# ===================== 核心检索函数（主程序调用这个） =====================
def retrieve_context(user_query: str) -> str:
    """
    【核心接口】根据用户问题检索相关上下文
    :param user_query: 用户输入的问题（纯文本）
    :return: 拼接好的上下文文本（直接用于提示词）
    """
    # 1. 加载向量库
    db = load_vector_db()
    
    # 2. 检索最相似的TOP-K个文档
    # FAISS内部自动完成：问题embedding + 余弦相似度检索
    retrieved_docs: list[Document] = db.similarity_search(
        query=user_query,
        k=TOP_K
    )

    # 3. 清洗并拼接检索结果（去除空行、多余换行）
    context_content = []
    for i, doc in enumerate(retrieved_docs):
        content = doc.page_content.strip().replace("\n\n", "\n")
        context_content.append(f"[参考资料{i+1}]\n{content}")

    # 4. 合并为最终文本，返回给主程序
    final_context = "\n\n".join(context_content)
    return final_context

# ===================== 测试代码 =====================
if __name__ == "__main__":
    # 测试检索功能
    test_query = "What is Task Decomposition?"
    context = retrieve_context(test_query)
    print("=" * 50)
    print("检索到的上下文：")
    print("=" * 50)
    print(context)