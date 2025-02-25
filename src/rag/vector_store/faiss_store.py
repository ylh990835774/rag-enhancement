import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss

from ..config import RAGConfig
from ..document.embeddings import DocumentEmbeddings
from ..utils.metrics import MetricsCollector


class FAISSVectorStore:
    """基于FAISS的向量存储"""

    def __init__(self, config: RAGConfig, embeddings: DocumentEmbeddings, metrics: Optional[MetricsCollector] = None):
        self.config = config
        self.embeddings = embeddings
        self.metrics = metrics or MetricsCollector()
        self.logger = logging.getLogger(__name__)

        # 初始化FAISS索引
        self.dimension = 768  # 根据embedding模型调整
        self.index = faiss.IndexFlatIP(self.dimension)  # 内积相似度

        # 文档ID映射
        self.doc_ids: List[str] = []
        self.doc_mapping: Dict[str, Dict[str, Any]] = {}

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """添加文档到向量存储"""
        try:
            # 生成文档向量
            doc_vectors = self.embeddings.embed_documents(documents)

            # 添加到FAISS索引
            self.index.add(doc_vectors.cpu().numpy())

            # 更新文档映射
            for i, doc in enumerate(documents):
                doc_id = f"doc_{len(self.doc_ids) + i}"
                self.doc_ids.append(doc_id)
                self.doc_mapping[doc_id] = doc

            self.logger.info(f"Added {len(documents)} documents to vector store")

        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            raise

    def similarity_search(self, query: str, k: int = 5, threshold: float = 0.6) -> List[Tuple[Dict[str, Any], float]]:
        """相似度搜索"""
        try:
            # 生成查询向量
            query_vector = self.embeddings.embed_query(query)

            # 搜索最相似的文档
            scores, indices = self.index.search(query_vector.cpu().numpy().reshape(1, -1), k)

            # 过滤低相似度的结果
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score < threshold:
                    continue
                if idx >= len(self.doc_ids):
                    continue

                doc_id = self.doc_ids[idx]
                doc = self.doc_mapping[doc_id]
                results.append((doc, float(score)))

            return results

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise

    def save(self, path: Path) -> None:
        """保存向量存储"""
        path.mkdir(parents=True, exist_ok=True)

        # 保存FAISS索引
        faiss.write_index(self.index, str(path / "index.faiss"))

        # 保存文档映射
        with open(path / "mapping.pkl", "wb") as f:
            pickle.dump({"doc_ids": self.doc_ids, "doc_mapping": self.doc_mapping}, f)

    @classmethod
    def load(cls, path: Path, config: RAGConfig, embeddings: DocumentEmbeddings) -> "FAISSVectorStore":
        """加载向量存储"""
        if not path.exists():
            raise FileNotFoundError(f"Vector store not found at {path}")

        store = cls(config, embeddings)

        # 加载FAISS索引
        store.index = faiss.read_index(str(path / "index.faiss"))

        # 加载文档映射
        with open(path / "mapping.pkl", "rb") as f:
            mapping = pickle.load(f)
            store.doc_ids = mapping["doc_ids"]
            store.doc_mapping = mapping["doc_mapping"]

        return store
