from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from embeddings_demo import QianfanMultiModalEmbeddings
import faiss
embeddings = QianfanMultiModalEmbeddings()
sample_vector = embeddings.embed(text="test")
dimension = len(sample_vector)
index = faiss.IndexFlatL2(dimension)

#数据需要是Document对象
test_documents = [
    # 1. 纯文本知识类
    Document(
        page_content="2025年全球人工智能市场规模预计将突破5000亿美元，其中多模态大模型的增长率最高，达到了45%。",
        metadata={
            "source": "AI行业报告2025",
            "type": "text",
            "category": "科技"
        }
    ),
    
    # 2. 图片关联类（模拟你上传的红色风景图）
    Document(
        page_content="这是一张展示红色地貌的自然风景图，主要特征是红色的土壤和远处的深色山脉，天空部分有云层覆盖，整体色调偏暖，给人一种荒凉而壮丽的视觉感受。",
        metadata={
            "source": "用户上传图片mmexport1711688494306.jpg",
            "type": "image_description",
            "category": "自然风景",
            "image_path": "./mmexport1711688494306.jpg"
        }
    ),
    
    # 3. 教育知识类
    Document(
        page_content="教师职业在国内的前景整体向好，随着国家对教育的重视程度不断提高，教师的薪资待遇和社会地位都在稳步提升，尤其是在一线城市和重点学校，优秀教师的需求非常大。",
        metadata={
            "source": "教育行业白皮书",
            "type": "text",
            "category": "教育"
        }
    ),
    
    # 4. 多模态产品描述类
    Document(
        page_content="这是一款白色的智能手表，屏幕尺寸为1.78英寸，支持心率监测、血氧检测、睡眠分析等健康功能，续航时间可达7天，防水等级为IP68。",
        metadata={
            "source": "产品图片描述",
            "type": "product_description",
            "category": "消费电子",
            "image_url": "https://example.com/smartwatch.jpg"
        }
    ),
    
    # 5. 生活常识类
    Document(
        page_content="夏季养生需要注意多喝水，每天的饮水量建议在1500-2000毫升左右，避免在高温时段长时间户外活动，饮食宜清淡，多吃蔬菜水果，保证充足的睡眠。",
        metadata={
            "source": "健康养生指南",
            "type": "text",
            "category": "生活健康"
        }
    )
]

vector_store = FAISS.from_documents(
    documents=test_documents,
    embedding=embeddings
)

#vector_store.save_local("./faiss_db")
#FAISS.load_local("./faiss_db", embeddings)


results =vector_store.similarity_search("今天的政治新闻", k=2)
for res in results:
    print(type(res))
    print(f"* {res.page_content}[{res.metadata}]")