"""指代消解改写(PRD F4):多轮对话时,把依赖上下文的追问改写为独立问题。

原理:「那它的价格呢?」这类追问直接检索会丢失主体——先用 LLM 结合
最近几轮对话把它改写成自包含问题(「出差住宿补贴是多少?」),再走
标准检索/质检/生成链路;检索层零改动。

设计(对齐 checker 的失败方向原则):
- 改写走快模型(FAST_LLM_MODEL,2~4s),只对追问调用,首轮直传;
- **失败一律返回原问题**(改写挂 ≠ 问答挂,退化为单轮效果);
- 上下文只取最近 REWRITE_TURNS 轮,防 prompt 膨胀。
"""
from __future__ import annotations

import os

REWRITE_TURNS = 4  # 改写参考的最近轮次

_SYSTEM = (
    "你是查询改写器。根据对话历史,把用户的最后一个问题改写为独立完整的"
    "检索问题:补全指代(它/那/这个)与省略的主语,保留原意,不要回答问题。"
    "若问题本身已独立完整,原样输出。只输出改写后的问题,不要任何解释。"
)


def rewrite_query(query: str, turns: list[dict],
                  model: str | None = None) -> str:
    """结合会话历史改写 query 为独立问题;无历史或改写失败返回原问题。"""
    if not turns:
        return query

    history = "\n".join(
        f"用户:{t['q']}\n助手:{t['a'][:150]}" for t in turns[-REWRITE_TURNS:]
    )
    user = f"【对话历史】\n{history}\n\n【最后一个问题】\n{query}"

    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("LLM_API_KEY", ""),
        base_url=os.environ.get("LLM_BASE_URL") or None,
    )
    try:
        resp = client.chat.completions.create(
            model=model or os.environ.get("FAST_LLM_MODEL", "kimi-for-coding"),
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": user}],
            max_tokens=1000,  # 快模型也有隐藏思考开销,给足防截断空串
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        #  sanity:空串/异常长(改写不应比原问题长 3 倍)都视为失败
        if not rewritten or len(rewritten) > max(60, len(query) * 3):
            return query
        return rewritten
    except Exception:
        return query  # 改写失败:退回原问题,退化为单轮
