"""答案反馈收集(PRD F5 / docs/03 §2.3):👍👎 落盘,👎 进 bad case 池。

设计(对齐 refuser.log_refusal 的日志模式):
- 全部反馈追加 logs/feedback.jsonl(一条一行 JSON);
- rating=-1(👎)同时追加 logs/bad_cases.jsonl——即 bad case 池,
  供运营标注归因(检索/生成/知识盲区,PRD §6.3 评估闭环);
- 日志写失败不影响主流程,接口仍返回成功;
- qa_id 来自 ask 响应 meta 帧(uuid)。M1 阶段问答日志未落盘,
  不做 qa_id 存在性校验;M2 迁业务库后再关联 question/answer 上下文。

issue_type 枚举(👎 时可选,PRD F5):
  not_found(没查到) / wrong_answer(答错了) /
  wrong_source(引用错) / bad_refuse(拒答不当) / other(其他)
"""
from __future__ import annotations

import json
import os
import time

from engine.paths import LOG_DIR

ISSUE_TYPES = ("not_found", "wrong_answer", "wrong_source", "bad_refuse", "other")


def record_feedback(
    qa_id: str,
    rating: int,
    issue_type: str | None = None,
    comment: str | None = None,
    log_dir: str = LOG_DIR,
) -> dict:
    """记录一条反馈;👎 额外写入 bad case 池。返回落盘的记录。

    写盘失败只吞掉 OSError(对齐 refusals 日志的"不阻断"原则)。
    """
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "qa_id": qa_id,
        "rating": rating,  # 1=👍 / -1=👎
        "issue_type": issue_type,
        "comment": comment or None,
        "status": "open" if rating < 0 else None,  # bad case 处理状态(供 M2 闭环)
    }
    try:
        os.makedirs(log_dir, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with open(os.path.join(log_dir, "feedback.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(line)
        if rating < 0:
            with open(os.path.join(log_dir, "bad_cases.jsonl"), "a",
                      encoding="utf-8") as f:
                f.write(line)
    except OSError:
        pass  # 日志失败不阻断反馈
    return record
