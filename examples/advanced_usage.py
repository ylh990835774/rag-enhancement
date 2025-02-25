import asyncio
from pathlib import Path

from src.rag import RAGConfig, RAGSystem
from src.rag.retrieval.retriever import HybridRetriever


async def main():
    # 自定义配置
    config = RAGConfig(
        embedding_model="BAAI/bge-large-zh",
        llm_model="deepseek-ai/deepseek-llm-7b-chat",
        chunk_size=500,
        chunk_overlap=50,
        device="cuda:0",
        batch_size=16,
        cache_dir=Path("./custom_cache"),
    )

    # 初始化系统
    rag = RAGSystem(config)

    # 自定义检索器参数
    rag.retriever = HybridRetriever(
        vector_store=rag.vector_store,
        metrics=rag.metrics,
        keyword_weight=0.4,  # 增加关键词检索权重
    )

    # 批量处理文档
    doc_dir = Path("data/documents")
    docs = list(doc_dir.glob("**/*.*"))  # 递归获取所有文件
    await rag.process_documents(docs)

    # 多轮对话示例
    conversations = ["什么是机器学习？", "它和深度学习有什么区别？", "请举一个具体的应用例子"]

    for i, query in enumerate(conversations):
        print(f"\n问题 {i + 1}: {query}")
        result = await rag.query(query)
        print(f"答案: {result['answer']}")
        print(f"相关文档数: {len(result['sources'])}")
        print(f"处理时间: {result['metrics']['processing_time']:.2f}秒")

    # 获取系统指标
    metrics = rag.metrics.get_average_metrics()
    print("\n系统性能指标:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
