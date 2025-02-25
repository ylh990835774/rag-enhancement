import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import RAGConfig
from .document.embeddings import DocumentEmbeddings
from .document.processor import DocumentProcessor
from .query.processor import StreamingQueryProcessor
from .retrieval.retriever import HybridRetriever
from .utils.metrics import MetricsCollector
from .vector_store.faiss_store import FAISSVectorStore


class RAGSystem:
    """RAG系统"""

    def __init__(self, config: Optional[RAGConfig] = None, load_path: Optional[Path] = None):
        self.config = config or RAGConfig()
        self.logger = logging.getLogger(__name__)
        self.metrics = MetricsCollector()

        # 初始化组件
        self._init_components()

        # 如果指定了加载路径，则加载现有系统
        if load_path:
            self.load(load_path)

    def _init_components(self):
        """初始化系统组件"""
        try:
            # 文档处理
            self.doc_processor = DocumentProcessor(self.config)

            # 向量化
            self.embeddings = DocumentEmbeddings(self.config)

            # 向量存储
            self.vector_store = FAISSVectorStore(config=self.config, embeddings=self.embeddings, metrics=self.metrics)

            # 检索器
            self.retriever = HybridRetriever(vector_store=self.vector_store, metrics=self.metrics)

            # 查询处理器
            self.query_processor = StreamingQueryProcessor(
                config=self.config, retriever=self.retriever, llm_model=self._init_llm()
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize RAG system: {e}")
            raise

    def _init_llm(self):
        """初始化LLM模型"""
        # 实现具体的LLM初始化逻辑
        pass

    async def process_documents(self, file_paths: List[Path]) -> None:
        """处理文档"""
        try:
            for path in file_paths:
                # 处理文档
                documents = self.doc_processor.process_file(path)

                # 添加到向量存储
                self.vector_store.add_documents(documents)

            self.logger.info(f"Processed {len(file_paths)} documents")

        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            raise

    async def query(self, query: str, query_id: Optional[str] = None) -> Dict[str, Any]:
        """处理查询"""
        try:
            # 生成查询ID
            if query_id is None:
                query_id = f"query_{len(self.metrics.metrics)}"

            # 处理查询
            result = await self.query_processor.process_query(query=query, query_id=query_id)

            return {
                "answer": result.answer,
                "sources": result.sources,
                "confidence": result.confidence,
                "metrics": result.metrics.__dict__,
            }

        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            raise

    async def query_stream(self, query: str, query_id: Optional[str] = None):
        """流式处理查询"""
        try:
            if query_id is None:
                query_id = f"query_{len(self.metrics.metrics)}"

            async for token in self.query_processor.process_query_stream(query=query, query_id=query_id):
                yield token

        except Exception as e:
            self.logger.error(f"Streaming query failed: {e}")
            raise

    def save(self, path: Path) -> None:
        """保存系统状态"""
        try:
            path.mkdir(parents=True, exist_ok=True)

            # 保存向量存储
            self.vector_store.save(path / "vector_store")

            # 保存配置
            self.config.save(path / "config.yaml")

            self.logger.info(f"System saved to {path}")

        except Exception as e:
            self.logger.error(f"Failed to save system: {e}")
            raise

    def load(self, path: Path) -> None:
        """加载系统状态"""
        try:
            if not path.exists():
                raise FileNotFoundError(f"System not found at {path}")

            # 加载配置
            self.config = RAGConfig.from_yaml(path / "config.yaml")

            # 重新初始化组件
            self._init_components()

            # 加载向量存储
            self.vector_store = FAISSVectorStore.load(path / "vector_store", self.config, self.embeddings)

            self.logger.info(f"System loaded from {path}")

        except Exception as e:
            self.logger.error(f"Failed to load system: {e}")
            raise
