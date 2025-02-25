import asyncio
from pathlib import Path

from src.rag import RAGConfig, RAGSystem


async def main():
    # 初始化配置
    config = RAGConfig(
        embedding_model="BAAI/bge-large-zh",
        llm_model="deepseek-ai/deepseek-llm-7b-chat",
        chunk_size=500,
        chunk_overlap=50,
    )

    # 初始化系统
    rag = RAGSystem(config)

    # 处理文档
    docs = [Path("data/sample/doc1.pdf"), Path("data/sample/doc2.txt"), Path("data/sample/doc3.md")]
    await rag.process_documents(docs)

    # 普通查询
    query = "什么是机器学习？"
    result = await rag.query(query)
    print(f"问题: {query}")
    print(f"答案: {result['answer']}")
    print(f"来源: {[doc['metadata']['source'] for doc in result['sources']]}")
    print(f"置信度: {result['confidence']}")

    # 流式查询
    query = "深度学习和机器学习的区别是什么？"
    print(f"\n问题: {query}")
    print("答案: ", end="", flush=True)
    async for token in rag.query_stream(query):
        print(token, end="", flush=True)
    print()

    # 保存系统状态
    rag.save(Path("data/saved_system"))


if __name__ == "__main__":
    asyncio.run(main())
