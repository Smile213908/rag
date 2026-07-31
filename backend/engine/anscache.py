"""答案缓存(M2 性能优化 P0):高频重复问题直接返回答案,跳过检索+生成。

- 精确匹配:问题归一化(去全部空白 + 去结尾句读)做键;
  一期不做相似问题模糊缓存(防张冠李戴,见设计方案)。
- 只缓存非拒答答案;知识库任何变更由 retriever._after_mutation 调 clear() 全清。
- 存储 backend/.cache/answers.db(sqlite,可再生,.cache 已 gitignore)。
- 读写失败一律放行(get 返回 None / put、clear 静默),缓存是优化不是功能。
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time

from engine.paths import CACHE_DIR

DB_PATH = os.path.join(CACHE_DIR, "answers.db")


def _norm(query: str) -> str:
    """归一化:「住宿补贴多少?」「住宿补贴多少」「住宿补贴 多少」同键。"""
    q = re.sub(r"\s+", "", query)
    return re.sub(r"[。?!！?,、;:\.…]+$", "", q)


def _conn() -> sqlite3.Connection:
    os.makedirs(CACHE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS answers ("
        "qkey TEXT PRIMARY KEY, "
        "question TEXT NOT NULL, "
        "answer TEXT NOT NULL, "
        "sources TEXT NOT NULL, "
        "model TEXT, "
        "created_at REAL NOT NULL)"
    )
    return conn


def get(query: str) -> dict | None:
    """命中返回 {answer, sources, model, created_at};未命中或异常返回 None。"""
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT answer, sources, model, created_at "
                "FROM answers WHERE qkey=?", (_norm(query),),
            ).fetchone()
        if not row:
            return None
        return {"answer": row[0], "sources": json.loads(row[1]),
                "model": row[2], "created_at": row[3]}
    except Exception:
        return None


def put(query: str, answer: str, sources: list[dict],
        model: str | None) -> None:
    """写入/覆盖缓存;失败静默。"""
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO answers VALUES (?,?,?,?,?,?)",
                (_norm(query), query, answer,
                 json.dumps(sources, ensure_ascii=False), model, time.time()),
            )
    except Exception:
        pass


def clear() -> int:
    """知识库变更后全清,返回清除条数;失败返回 0。"""
    try:
        with _conn() as conn:
            cur = conn.execute("DELETE FROM answers")
        return cur.rowcount
    except Exception:
        return 0
