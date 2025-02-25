from pathlib import Path
from typing import Any, Dict, List, Optional

import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from pdf2image import convert_from_path
from PIL import Image, UnidentifiedImageError

from ..config import RAGConfig
from ..utils.logging import setup_logging


class DocumentProcessor:
    """文档处理器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = setup_logging(config.log_level)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )

    def process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """处理单个文件"""
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            processor_map = {
                ".txt": self._process_text,
                ".md": self._process_text,
                ".pdf": self._process_pdf,
                ".png": self._process_image,
                ".jpg": self._process_image,
                ".jpeg": self._process_image,
            }

            processor = processor_map.get(file_path.suffix.lower())
            if not processor:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

            documents = processor(file_path)
            return self.text_splitter.split_documents(documents)

        except Exception as e:
            self.logger.exception(f"Error processing file {file_path}: {e}")
            raise

    def _process_text(self, file_path: Path) -> List[Dict[str, Any]]:
        """处理文本文件"""
        loader = TextLoader(str(file_path))
        return loader.load()

    def _process_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """处理PDF文件"""
        documents = []

        # 提取文本
        loader = PyPDFLoader(str(file_path))
        documents.extend(loader.load())

        # 处理图片
        try:
            images = convert_from_path(str(file_path))
            for i, img in enumerate(images):
                text = self._extract_text_from_image(img)
                if text:
                    documents.append(
                        {
                            "page_content": text,
                            "metadata": {"source": f"{file_path}:image_{i}", "page": i, "type": "pdf_image"},
                        }
                    )
        except Exception as e:
            self.logger.warning(f"Failed to process PDF images in {file_path}: {e}")

        return documents

    def _process_image(self, file_path: Path) -> List[Dict[str, Any]]:
        """处理图片文件"""
        try:
            with Image.open(file_path) as img:
                text = self._extract_text_from_image(img)
                if not text:
                    return []

                return [
                    {
                        "page_content": text,
                        "metadata": {"source": str(file_path), "type": "image", "size": img.size, "format": img.format},
                    }
                ]
        except UnidentifiedImageError:
            self.logger.error(f"Invalid image file: {file_path}")
            raise
        except Exception as e:
            self.logger.exception(f"Error processing image {file_path}: {e}")
            raise

    def _extract_text_from_image(self, img: Image.Image) -> Optional[str]:
        """从图片中提取文本"""
        try:
            # 图像预处理
            img = self._preprocess_image(img)

            # OCR处理
            text = pytesseract.image_to_string(img, lang=self.config.ocr_languages)

            return text.strip()
        except Exception as e:
            self.logger.error(f"OCR failed: {e}")
            return None

    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """图像预处理"""
        # 转换为RGB
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")

        # 调整大小
        img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)

        # 转换为灰度图
        img = img.convert("L")

        # 二值化
        img = img.point(lambda x: 0 if x < 128 else 255, "1")

        return img
