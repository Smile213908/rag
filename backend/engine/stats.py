"""运营统计(docs/03 §4):扫描 logs/*.jsonl 计算看板指标。

数据源:asks.jsonl(提问)/ feedback.jsonl(👍👎)/ bad_cases.jsonl(bad case 池)。
单进程 M2 阶段直接读文件全量扫描;数据量大后按路线图迁业务库聚合。
指标口径(与 PRD §2.2 / docs/03 §4 对齐):
- 拒答率 = 拒答条数 / 总提问;命中率 = 1 - 拒答率(作答占比);
- 👍率 = 👍 / (👍+👎);P95 延迟取 asks.latency_ms 分位;
- bad case 状态折叠:同一 qa_id 以最新一条记录为准(resolve 追加写)。
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter

from engine.paths import LOG_DIR


def _read(name: str, log_dir: str = LOG_DIR) -> list[dict]:
    path = os.path.join(log_dir, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _p95(vals: list[int]) -> int:
    if not vals:
        return 0
    vals = sorted(vals)
    k = (len(vals) - 1) * 0.95
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (k - lo))


def overview(log_dir: str = LOG_DIR) -> dict:
    asks = _read("asks.jsonl", log_dir)
    fb = _read("feedback.jsonl", log_dir)
    today = time.strftime("%Y-%m-%d")
    total = len(asks)
    refused = sum(1 for a in asks if a.get("refused"))
    ups = sum(1 for f in fb if f.get("rating") == 1)
    downs = sum(1 for f in fb if f.get("rating") == -1)
    return {
        "total_queries": total,
        "today_queries": sum(1 for a in asks if a.get("ts", "").startswith(today)),
        "hit_rate": round(1 - refused / total, 4) if total else None,
        "refuse_rate": round(refused / total, 4) if total else None,
        "thumbs_up_rate": round(ups / (ups + downs), 4) if ups + downs else None,
        "p95_latency_ms": _p95([a.get("latency_ms", 0) for a in asks]),
        "tokens_total": sum(a.get("tokens") or 0 for a in asks),
        "open_bad_cases": len(bad_cases(status="open", log_dir=log_dir)),
    }


def hot_questions(n: int = 10, log_dir: str = LOG_DIR) -> list[dict]:
    """高频问题 Top-N(按原问题归一化计数;聚类归并留 M2 后续)。"""
    asks = _read("asks.jsonl", log_dir)
    cnt = Counter(a.get("question", "").strip() for a in asks if a.get("question"))
    return [{"question": q, "count": c} for q, c in cnt.most_common(n)]


def bad_cases(status: str | None = None, log_dir: str = LOG_DIR) -> list[dict]:
    """bad case 池(按 qa_id 折叠到最新状态),默认只看 open。"""
    latest: dict[str, dict] = {}
    for rec in _read("bad_cases.jsonl", log_dir):
        latest[rec.get("qa_id", "")] = rec  # 后写覆盖先写 = 最新状态
    items = sorted(latest.values(), key=lambda r: r.get("ts", ""), reverse=True)
    if status:
        items = [r for r in items if r.get("status") == status]
    return items


def resolve_bad_case(qa_id: str, action: str, log_dir: str = LOG_DIR) -> bool:
    """标记 bad case 已处理(追加 resolved 记录,附处理动作)。"""
    pool = {r.get("qa_id"): r for r in _read("bad_cases.jsonl", log_dir)}
    if qa_id not in pool:
        return False
    rec = dict(pool[qa_id])
    rec.update({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "resolved",
        "resolve_action": action,
    })
    try:
        with open(os.path.join(log_dir, "bad_cases.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass
    return True
