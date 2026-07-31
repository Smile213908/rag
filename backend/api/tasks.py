"""异步任务跟踪(单进程内存版):索引/重建类耗时操作的任务注册与进度查询。

约束:单 worker 部署(见 server.py 模块头),任务表为进程内字典;
迁多实例时按路线图 M2 换 Redis/数据库任务队列。
"""
from __future__ import annotations

import threading
import time
import uuid


class TaskStore:
    """线程安全的任务登记表。状态机:indexing → done | failed。"""

    def __init__(self) -> None:
        self._tasks: dict[str, dict] = {}
        self._by_doc: dict[str, str] = {}  # doc_id -> 最新 task_id
        self._lock = threading.Lock()

    def create(self, doc_id: str, kind: str) -> str:
        task_id = uuid.uuid4().hex
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id, "doc_id": doc_id, "kind": kind,
                "status": "indexing", "progress": 0.0,
                "stage": "queued",           # 分段进度:queued/uploaded/parsing/chunked/encoding/finalizing/done
                "chunks_done": 0, "chunks_total": 0,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "error": "",
            }
            self._by_doc[doc_id] = task_id
        return task_id

    def update(self, task_id: str, **fields) -> None:
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].update(fields)

    def get(self, task_id: str) -> dict | None:
        with self._lock:
            t = self._tasks.get(task_id)
            return dict(t) if t else None

    def latest_for_doc(self, doc_id: str) -> dict | None:
        with self._lock:
            tid = self._by_doc.get(doc_id)
            return dict(self._tasks[tid]) if tid else None

    def run_async(self, task_id: str, fn, *args) -> None:
        """后台线程执行 fn,异常落 task.error 并置 failed。"""
        def _run() -> None:
            try:
                fn(*args)
                self.update(task_id, status="done", progress=1.0)
            except Exception as exc:
                self.update(task_id, status="failed", error=str(exc)[:300])
        threading.Thread(target=_run, daemon=True).start()
