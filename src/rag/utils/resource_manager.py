import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict


class ResourceManager:
    """资源管理器"""

    def __init__(self):
        self._resources: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._ref_counts: Dict[str, int] = {}

    @contextmanager
    def acquire(self, name: str, factory: Callable[[], Any]):
        """获取资源的上下文管理器"""
        with self._lock:
            if name not in self._resources:
                self._resources[name] = factory()
                self._ref_counts[name] = 0
            self._ref_counts[name] += 1
            resource = self._resources[name]

        try:
            yield resource
        finally:
            with self._lock:
                self._ref_counts[name] -= 1
                if self._ref_counts[name] == 0:
                    self.release(name)

    def release(self, name: str):
        """释放指定资源"""
        with self._lock:
            if name in self._resources:
                resource = self._resources[name]
                if hasattr(resource, "close"):
                    resource.close()
                elif hasattr(resource, "cleanup"):
                    resource.cleanup()
                del self._resources[name]
                del self._ref_counts[name]

    def release_all(self):
        """释放所有资源"""
        with self._lock:
            for name in list(self._resources.keys()):
                self.release(name)
