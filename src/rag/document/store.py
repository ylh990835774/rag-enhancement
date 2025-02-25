import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logging import logging


class ThreadSafeDocumentStore:
    """线程安全的文档存储"""

    def __init__(self, storage_path: Optional[Path] = None):
        self._documents: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._storage_path = storage_path

    def add(self, docs: List[Dict[str, Any]]) -> None:
        """添加文档"""
        with self._lock:
            self._documents.extend(docs)
            self._save_to_disk()

    def get_all(self) -> List[Dict[str, Any]]:
        """获取所有文档"""
        with self._lock:
            return self._documents.copy()

    def clear(self) -> None:
        """清空文档"""
        with self._lock:
            self._documents.clear()
            self._save_to_disk()

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取文档"""
        with self._lock:
            for doc in self._documents:
                if doc.get("metadata", {}).get("id") == doc_id:
                    return doc.copy()
        return None

    def remove_by_id(self, doc_id: str) -> bool:
        """根据ID删除文档"""
        with self._lock:
            for i, doc in enumerate(self._documents):
                if doc.get("metadata", {}).get("id") == doc_id:
                    self._documents.pop(i)
                    self._save_to_disk()
                    return True
        return False

    def _save_to_disk(self) -> None:
        """保存到磁盘"""
        if self._storage_path:
            try:
                with open(self._storage_path, "w", encoding="utf-8") as f:
                    json.dump(self._documents, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logging.error(f"Failed to save documents to disk: {e}")

    def _load_from_disk(self) -> None:
        """从磁盘加载"""
        if self._storage_path and self._storage_path.exists():
            try:
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    self._documents = json.load(f)
            except Exception as e:
                logging.error(f"Failed to load documents from disk: {e}")
