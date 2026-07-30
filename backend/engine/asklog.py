"""提问日志(docs/03 §4 运营统计的数据基础):每次问答落 logs/asks.jsonl。

M1 遗留:M1 阶段问答未落盘(见 feedback.py 注释),运营指标无从算起;
M2 补齐——一条一行 JSON,字段对齐统计口径(拒答率/延迟/👍率/token)。
写失败不阻断主流程(对齐 refusals/feedback 日志原则)。
"""
from __future__ import annotations

import json
import os
import time

from engine.paths import LOG_DIR


def log_ask(
    qa_id: str,
    session_id: str,
    question: str,
    standalone: str,
    refused: bool,
    latency_ms: int,
    tokens: int | None,
    model: str | None,
    layer: str | None = None,
    log_dir: str = LOG_DIR,
) -> None:
    """落一条问答日志。layer 仅拒答时有值(threshold/crag)。"""
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "qa_id": qa_id,
        "session_id": session_id,
        "question": question,
        "standalone": standalone if standalone != question else None,
        "refused": refused,
        "layer": layer,
        "latency_ms": latency_ms,
        "tokens": tokens,
        "model": model,
    }
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "asks.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass
