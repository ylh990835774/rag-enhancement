import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from ..config import RAGConfig
from ..retrieval.retriever import Retriever
from ..utils.metrics import QueryMetrics


@dataclass
class QueryResult:
    """查询结果"""

    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    metrics: QueryMetrics


class QueryProcessor:
    """查询处理器"""

    def __init__(
        self,
        config: RAGConfig,
        retriever: Retriever,
        llm_model: Any,  # 实际使用时替换为具体的LLM类型
    ):
        self.config = config
        self.retriever = retriever
        self.llm = llm_model
        self.logger = logging.getLogger(__name__)

    async def process_query(self, query: str, query_id: str, context_size: int = 2048) -> QueryResult:
        """处理查询"""
        try:
            start_time = time.time()

            # 检索相关文档
            retrieval_result = self.retriever.retrieve(query=query, query_id=query_id)

            # 构建提示
            prompt = self._build_prompt(query=query, documents=retrieval_result.documents, context_size=context_size)

            # 生成答案
            answer, confidence = await self._generate_answer(prompt)

            # 计算处理时间
            processing_time = time.time() - start_time

            # 更新指标
            metrics = retrieval_result.metrics
            metrics.processing_time = processing_time

            return QueryResult(
                answer=answer, sources=retrieval_result.documents, confidence=confidence, metrics=metrics
            )

        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            raise

    def _build_prompt(self, query: str, documents: List[Dict[str, Any]], context_size: int) -> str:
        """构建提示"""
        # 合并文档内容
        context = "\n\n".join(doc["page_content"] for doc in documents)

        # 截断上下文以适应最大长度
        if len(context) > context_size:
            context = context[:context_size] + "..."

        # 构建提示模板
        prompt = f"""请基于以下信息回答问题。如果无法从提供的信息中找到答案，请明确说明。

信息来源:
{context}

问题: {query}

请提供详细的答案，并尽可能引用具体的信息来源。回答时要确保:
1. 答案准确且完整
2. 清晰说明信息来源
3. 如果信息不足，明确指出

回答:"""

        return prompt

    async def _generate_answer(self, prompt: str) -> tuple[str, float]:
        """生成答案"""
        try:
            # 调用LLM生成答案
            response = await self.llm.generate(prompt)

            # 提取答案和置信度
            answer = response.text
            confidence = response.confidence

            return answer, confidence

        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            raise


class StreamingQueryProcessor(QueryProcessor):
    """流式查询处理器"""

    async def process_query_stream(self, query: str, query_id: str, context_size: int = 2048):
        """流式处理查询"""
        try:
            # 检索文档
            retrieval_result = self.retriever.retrieve(query=query, query_id=query_id)

            # 构建提示
            prompt = self._build_prompt(query=query, documents=retrieval_result.documents, context_size=context_size)

            # 流式生成
            async for token in self._generate_answer_stream(prompt):
                yield token

        except Exception as e:
            self.logger.error(f"Streaming query processing failed: {e}")
            raise

    async def _generate_answer_stream(self, prompt: str):
        """流式生成答案"""
        try:
            async for token in self.llm.generate_stream(prompt):
                yield token
        except Exception as e:
            self.logger.error(f"Streaming generation failed: {e}")
            raise
