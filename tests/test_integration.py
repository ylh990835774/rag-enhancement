import time
from pathlib import Path

import pytest
import torch

from src.rag import RAGConfig, RAGSystem


@pytest.mark.integration
class TestRAGIntegration:
    @pytest.fixture(scope="class")
    async def setup_system(self):
        config = RAGConfig(
            embedding_model="BAAI/bge-large-zh",
            llm_model="deepseek-ai/deepseek-llm-7b-chat",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        system = RAGSystem(config)

        # 准备测试数据
        test_docs = [Path("tests/data/test_doc_1.pdf"), Path("tests/data/test_doc_2.txt")]

        # 处理文档
        await system.process_documents(test_docs)

        return system

    @pytest.mark.asyncio
    async def test_end_to_end_query(self, setup_system):
        # 执行查询
        result = await setup_system.query("什么是机器学习？")

        # 验证结果
        assert result["answer"]
        assert len(result["sources"]) > 0
        assert result["confidence"] > 0.5
        assert result["metrics"]["processing_time"] < 5.0  # 处理时间应小于5秒

    @pytest.mark.asyncio
    async def test_streaming_performance(self, setup_system):
        # 测试流式响应性能
        start_time = time.time()
        tokens = []

        async for token in setup_system.query_stream("深度学习的应用"):
            tokens.append(token)
            # 确保每个token的生成时间合理
            assert time.time() - start_time < 10.0  # 总时间应小于10秒

        assert len(tokens) > 50  # 确保生成了足够的内容

    def test_system_persistence(self, setup_system, tmp_path):
        # 测试系统状态保存和加载
        save_path = tmp_path / "test_system"

        # 保存系统
        setup_system.save(save_path)

        # 加载系统
        loaded_system = RAGSystem(load_path=save_path)

        # 验证加载的系统状态
        assert len(loaded_system.vector_store.doc_ids) == len(setup_system.vector_store.doc_ids)

        # 验证向量存储的一致性
        original_vectors = setup_system.vector_store.index.reconstruct_n(0, 10)
        loaded_vectors = loaded_system.vector_store.index.reconstruct_n(0, 10)
        assert torch.allclose(torch.tensor(original_vectors), torch.tensor(loaded_vectors))
