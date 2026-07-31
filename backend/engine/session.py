"""会话存储(PRD F4 / docs/03 §2.4):单进程内存版多轮上下文。

职责:session_id → 对话轮次(问/答),供指代消解改写取上下文。
约束(对齐 tasks.py):单 worker 进程内字典,迁多实例时换 Redis/业务库;
容量双封顶——会话数 MAX_SESSIONS(超了淘汰最旧)、每会话轮次 MAX_TURNS。
"""
from __future__ import annotations

import threading
import time
import uuid

MAX_SESSIONS = 500
MAX_TURNS = 10  # 每会话最多保留轮次(改写只取最近几轮,见 rewriter)


class SessionStore:
    """线程安全的会话表。turns: [{"q": 用户原话, "a": 答案}];title 缺省取首条问题。"""

    def __init__(self) -> None:
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None) -> tuple[str, list[dict]]:
        """取会话;不存在(或未给 id)则新建。返回 (session_id, turns 副本)。"""
        with self._lock:
            sid = session_id or uuid.uuid4().hex
            s = self._sessions.get(sid)
            if s is None:
                s = {"turns": [], "title": "",
                     "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                self._sessions[sid] = s
                self._evict_if_full()
            return sid, list(s["turns"])

    def append(self, session_id: str, q: str, a: str) -> None:
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return
            s["turns"].append({"q": q, "a": a})
            s["turns"] = s["turns"][-MAX_TURNS:]
            if not s.get("title"):  # 首条问题自动成标题
                s["title"] = q[:20]
            s["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def rename(self, session_id: str, title: str) -> bool:
        """修改会话标题(前端会话管理)。"""
        title = title.strip()[:50]
        if not title:
            return False
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return False
            s["title"] = title
            return True

    def history(self, session_id: str) -> dict | None:
        """会话历史(切换会话回填前端);不存在返回 None。"""
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return None
            return {"session_id": session_id, "title": s.get("title", ""),
                    "turns": list(s["turns"])}

    def list(self) -> list[dict]:
        with self._lock:
            return [{
                "session_id": sid,
                "title": s.get("title") or (s["turns"][-1]["q"][:20] if s["turns"] else "新会话"),
                "turns": len(s["turns"]),
                "updated_at": s["updated_at"],
                "last_question": s["turns"][-1]["q"][:40] if s["turns"] else "",
            } for sid, s in sorted(self._sessions.items(),
                                   key=lambda kv: kv[1]["updated_at"],
                                   reverse=True)]

    def _evict_if_full(self) -> None:
        if len(self._sessions) <= MAX_SESSIONS:
            return
        oldest = min(self._sessions, key=lambda k: self._sessions[k]["updated_at"])
        self._sessions.pop(oldest, None)


# 模块级单例(pipeline 与服务层共享)
SESSIONS = SessionStore()
