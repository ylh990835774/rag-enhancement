from typing import Any, Dict, List

import torch
from transformers import AutoModel, AutoTokenizer

from ..config import RAGConfig


class DocumentEmbeddings:
    """文档向量化处理"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.device = torch.device(config.device)

        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)
        self.model = AutoModel.from_pretrained(config.embedding_model).to(self.device)

    def embed_documents(self, documents: List[Dict[str, Any]]) -> torch.Tensor:
        """批量生成文档向量"""
        texts = [doc["page_content"] for doc in documents]

        # 分批处理
        batch_size = self.config.batch_size
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            embeddings.append(batch_embeddings)

        return torch.cat(embeddings, dim=0)

    def embed_query(self, query: str) -> torch.Tensor:
        """生成查询向量"""
        return self._embed_batch([query])[0]

    def _embed_batch(self, texts: List[str]) -> torch.Tensor:
        """处理单批次文本"""
        # 编码文本
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(
            self.device
        )

        # 生成向量
        with torch.no_grad():
            outputs = self.model(**encoded)
            # 使用[CLS]标记的输出作为文档向量
            embeddings = outputs.last_hidden_state[:, 0, :]

        # 归一化
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings
