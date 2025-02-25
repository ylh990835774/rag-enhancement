from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


@dataclass
class RAGConfig:
    """RAG系统配置"""

    # 模型配置
    embedding_model: str = "BAAI/bge-large-zh"
    llm_model: str = "deepseek-ai/deepseek-llm-7b-chat"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 文档处理配置
    chunk_size: int = 1000
    chunk_overlap: int = 100
    batch_size: int = 32

    # 适配PDF图片
    process_pdf_images: bool = True

    # 系统配置
    max_retries: int = 3
    cache_dir: Path = Path("./cache")
    log_level: str = "INFO"

    # OCR配置
    ocr_languages: str = "chi_sim+eng"

    @classmethod
    def from_yaml(cls, config_path: Path) -> "RAGConfig":
        """从YAML文件加载配置"""
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def validate(self):
        """验证配置有效性"""
        if self.chunk_size <= self.chunk_overlap:
            raise ValueError("chunk_size must be greater than chunk_overlap")
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)
