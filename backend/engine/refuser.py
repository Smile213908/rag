"""拒答器(信任基石):reranker 最高分低于阈值 → 判定"库中无答案",拒绝生成。

原理(见大脑笔记「RAG 生成与幻觉抑制」§2):
检索缺失型幻觉(库里没有、模型硬答)的药方是**拒答机制**——
对 Rerank 分数设阈值,低于阈值直接「资料不足」,宁可拒答不可硬编。
阈值过严误杀、过松虚设:先用经验值上线,再用评估集标定(PRD §6.3)。

设计(PRD F3 / 架构④ / API `refused` 帧):
- 判据:精排后最高分 < REFUSE_THRESHOLD(或零命中)→ 拒答;
- 拒答时**不调用 LLM**,直接返回标准话术 + 建议(省 token、零幻觉风险);
- 拒答问题落盘 logs/refusals.jsonl,供运营分析是否为知识盲区。

分数量纲:bge-reranker-base 经 sentence-transformers CrossEncoder 输出
sigmoid 分数 ∈ (0,1)。v1 评估集(50 条,eval/)标定:库内 Top-1 ≥ 0.73,
库外纯噪声 ≤ 0.33,阈值取 0.7(库内零误杀、库外拒获 72.7%)。
注意:**域邻近对抗问题**(如"国际出差标准")分数高达 0.95+,落在库内
分布区间——单靠分数阈值达不到 PRD 的拒答率 ≥95%,需后续加第二层
校验(CRAG 质检 / 生成侧忠实度判断,见大脑笔记「RAG 生成与幻觉抑制」)。
"""
from __future__ import annotations

import json
import os
import time

from engine.chunking import Chunk
from engine.paths import LOG_DIR

# 拒答阈值:reranker 最高分低于此值即拒答(.env 可覆盖;评估集标定后调整)
# v1 评估集(50 条)标定结果:0.7 为平衡准确率最优且库内零误杀(见 eval/)
REFUSE_THRESHOLD = float(os.environ.get("REFUSE_THRESHOLD", "0.7"))

# 标准拒答话术(与 docs/03 API「拒答场景」的 answer 文案一致)
REFUSAL_MESSAGE = (
    "根据现有资料无法回答该问题(检索置信度不足)。"
    "建议换个问法补充关键词,或联系 HR/行政确认。"
)

# 拒答日志目录(LOG_DIR,供运营分析知识盲区,PRD F3)


def should_refuse(
    ranked: list[tuple[Chunk, float]],
    threshold: float = REFUSE_THRESHOLD,
) -> bool:
    """精排结果为零命中,或最高分低于阈值 → 应拒答。"""
    if not ranked:
        return True
    top_score = ranked[0][1]
    return top_score < threshold


def log_refusal(
    query: str,
    ranked: list[tuple[Chunk, float]],
    threshold: float = REFUSE_THRESHOLD,
    log_dir: str = LOG_DIR,
    layer: str = "threshold",
) -> None:
    """拒答问题落盘 logs/refusals.jsonl(追加),供运营聚类分析知识盲区。

    layer 标记触发层:threshold(第一层分数)/crag(第二层质检)。
    日志只记查询与分数,不记答案;写失败不影响主流程。
    """
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "query": query,
        "top_score": round(ranked[0][1], 4) if ranked else None,
        "threshold": threshold,
        "layer": layer,
    }
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "refusals.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # 日志失败不阻断拒答
