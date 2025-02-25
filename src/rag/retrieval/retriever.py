import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.metrics import MetricsCollector, QueryMetrics
from ..vector_store.faiss_store import FAISSVectorStore


@dataclass
class RetrievalResult:
    """检索结果"""

    documents: List[Dict[str, Any]]
    scores: List[float]
    query_id: str
    metrics: QueryMetrics


class Retriever:
    """文档检索器"""

    def __init__(self, vector_store: FAISSVectorStore, metrics: Optional[MetricsCollector] = None):
        self.vector_store = vector_store
        self.metrics = metrics or MetricsCollector()
        self.logger = logging.getLogger(__name__)

    def retrieve(self, query: str, query_id: str, top_k: int = 5, threshold: float = 0.6) -> RetrievalResult:
        """检索相关文档"""
        try:
            start_time = time.time()

            # 执行检索
            results = self.vector_store.similarity_search(query, k=top_k, threshold=threshold)

            retrieval_time = time.time() - start_time

            # 分离文档和分数
            documents = [r[0] for r in results]
            scores = [r[1] for r in results]

            # 收集指标
            metrics = QueryMetrics(
                query_time=retrieval_time,
                retrieval_time=retrieval_time,
                processing_time=0.0,  # 将在后续处理中更新
                total_docs=len(self.vector_store.doc_ids),
                retrieved_docs=len(documents),
            )

            return RetrievalResult(documents=documents, scores=scores, query_id=query_id, metrics=metrics)

        except Exception as e:
            self.logger.error(f"Retrieval failed for query {query_id}: {e}")
            raise


class HybridRetriever(Retriever):
    """混合检索器 - 支持关键词和语义检索"""

    def __init__(
        self, vector_store: FAISSVectorStore, metrics: Optional[MetricsCollector] = None, keyword_weight: float = 0.3
    ):
        super().__init__(vector_store, metrics)
        self.keyword_weight = keyword_weight

    def retrieve(self, query: str, query_id: str, top_k: int = 5, threshold: float = 0.6) -> RetrievalResult:
        """混合检索"""
        try:
            start_time = time.time()

            # 语义检索
            semantic_results = self.vector_store.similarity_search(
                query,
                k=top_k * 2,  # 检索更多候选
                threshold=threshold,
            )

            # 关键词检索
            keyword_results = self._keyword_search(query, k=top_k * 2)

            # 融合结果
            merged_results = self._merge_results(semantic_results, keyword_results, top_k)

            retrieval_time = time.time() - start_time

            # 分离文档和分数
            documents = [r[0] for r in merged_results]
            scores = [r[1] for r in merged_results]

            metrics = QueryMetrics(
                query_time=retrieval_time,
                retrieval_time=retrieval_time,
                processing_time=0.0,
                total_docs=len(self.vector_store.doc_ids),
                retrieved_docs=len(documents),
            )

            return RetrievalResult(documents=documents, scores=scores, query_id=query_id, metrics=metrics)

        except Exception as e:
            self.logger.error(f"Hybrid retrieval failed for query {query_id}: {e}")
            raise

    def _keyword_search(self, query: str, k: int) -> List[Tuple[Dict[str, Any], float]]:
        """关键词检索"""
        # 实现BM25或其他关键词检索算法
        # 这里是一个简单的实现
        from rank_bm25 import BM25Okapi

        # 准备文档
        docs = list(self.vector_store.doc_mapping.values())
        texts = [doc["page_content"] for doc in docs]

        # 创建BM25模型
        tokenized_texts = [text.split() for text in texts]
        bm25 = BM25Okapi(tokenized_texts)

        # 搜索
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)

        # 获取top-k结果
        top_indices = np.argsort(scores)[-k:][::-1]
        results = [(docs[i], float(scores[i])) for i in top_indices if scores[i] > 0]

        return results

    def _merge_results(
        self,
        semantic_results: List[Tuple[Dict[str, Any], float]],
        keyword_results: List[Tuple[Dict[str, Any], float]],
        top_k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """融合检索结果"""
        # 归一化分数
        semantic_max = max(s for _, s in semantic_results) if semantic_results else 1
        keyword_max = max(s for _, s in keyword_results) if keyword_results else 1

        normalized_semantic = [(doc, score / semantic_max) for doc, score in semantic_results]

        normalized_keyword = [(doc, score / keyword_max) for doc, score in keyword_results]

        # 合并结果
        doc_scores = {}
        for doc, score in normalized_semantic:
            doc_id = doc.get("metadata", {}).get("id", "")
            doc_scores[doc_id] = score * (1 - self.keyword_weight)

        for doc, score in normalized_keyword:
            doc_id = doc.get("metadata", {}).get("id", "")
            if doc_id in doc_scores:
                doc_scores[doc_id] += score * self.keyword_weight
            else:
                doc_scores[doc_id] = score * self.keyword_weight

        # 排序并返回top-k结果
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [(self.vector_store.doc_mapping[doc_id], score) for doc_id, score in sorted_results]
