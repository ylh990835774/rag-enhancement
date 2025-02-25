import pytest

from src.rag import RAGConfig, RAGSystem


@pytest.fixture
def rag_system():
    config = RAGConfig(
        embedding_model="BAAI/bge-large-zh",
        device="cpu",  # 测试时使用CPU
    )
    return RAGSystem(config)


@pytest.fixture
def sample_docs():
    return [
        {"page_content": "机器学习是人工智能的一个子领域", "metadata": {"source": "test_doc_1", "type": "text"}},
        {"page_content": "深度学习是机器学习的一个分支", "metadata": {"source": "test_doc_2", "type": "text"}},
    ]


@pytest.mark.asyncio
async def test_document_processing(rag_system, tmp_path):
    # 创建测试文件
    test_file = tmp_path / "test.txt"
    test_file.write_text("这是一个测试文档")

    # 处理文档
    await rag_system.process_documents([test_file])

    # 验证文档已被添加到向量存储
    assert len(rag_system.vector_store.doc_ids) > 0


@pytest.mark.asyncio
async def test_query(rag_system, sample_docs):
    # 添加测试文档
    rag_system.vector_store.add_documents(sample_docs)

    # 测试查询
    result = await rag_system.query("什么是机器学习？")

    assert "answer" in result
    assert "sources" in result
    assert "confidence" in result
    assert result["confidence"] > 0


@pytest.mark.asyncio
async def test_streaming_query(rag_system, sample_docs):
    # 添加测试文档
    rag_system.vector_store.add_documents(sample_docs)

    # 测试流式查询
    tokens = []
    async for token in rag_system.query_stream("什么是机器学习？"):
        tokens.append(token)

    assert len(tokens) > 0


def test_save_load(rag_system, sample_docs, tmp_path):
    # 添加测试文档
    rag_system.vector_store.add_documents(sample_docs)

    # 保存系统
    save_path = tmp_path / "test_system"
    rag_system.save(save_path)

    # 加载系统
    new_system = RAGSystem(load_path=save_path)

    # 验证文档已被正确加载
    assert len(new_system.vector_store.doc_ids) == len(sample_docs)


def test_metrics(rag_system, sample_docs):
    # 添加测试文档并执行查询
    rag_system.vector_store.add_documents(sample_docs)

    # 记录指标
    rag_system.metrics.start_query()
    metrics = rag_system.metrics.get_average_metrics()

    # 验证指标
    assert isinstance(metrics, dict)
    assert "query_time" in metrics
