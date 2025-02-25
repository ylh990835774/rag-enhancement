import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class QueryMetrics:
    """查询性能指标"""

    query_time: float
    retrieval_time: float
    processing_time: float
    total_docs: int
    retrieved_docs: int
    token_count: Optional[int] = None


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.metrics: Dict[str, QueryMetrics] = {}
        self._start_time: float = 0

    def start_query(self):
        """开始查询计时"""
        self._start_time = time.time()

    def end_query(self, query_id: str, metrics: QueryMetrics):
        """结束查询计时并记录指标"""
        metrics.total_time = time.time() - self._start_time
        self.metrics[query_id] = metrics

    def get_average_metrics(self) -> Dict[str, float]:
        """获取平均指标"""
        if not self.metrics:
            return {}

        total_metrics = {
            "query_time": 0,
            "retrieval_time": 0,
            "processing_time": 0,
            "total_docs": 0,
            "retrieved_docs": 0,
        }

        for m in self.metrics.values():
            total_metrics["query_time"] += m.query_time
            total_metrics["retrieval_time"] += m.retrieval_time
            total_metrics["processing_time"] += m.processing_time
            total_metrics["total_docs"] += m.total_docs
            total_metrics["retrieved_docs"] += m.retrieved_docs

        count = len(self.metrics)
        return {k: v / count for k, v in total_metrics.items()}
