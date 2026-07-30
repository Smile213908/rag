"""CRAG 质检器(拒答第二层):LLM 判断「检索资料是否真的支撑问题」。

原理(见大脑笔记「RAG 生成与幻觉抑制」§3 CRAG + [[RAG评估]] §3):
分数阈值能挡住纯无关问题(分数低),但**域邻近对抗问题**——问题贴着
知识库主题、资料却只覆盖邻接内容——reranker 分数会落在库内分布区间,
单阈值无解。CRAG 在生成前插一个轻量质检评估器,直接问 LLM
「这些资料能否回答这个问题」,不能即拒答。

v1 为二值质检(支撑/不支撑);CRAG 完整三档(正确/模糊→知识精炼/错误)
留待 v2。

成本:M1 每次问答多一次 k3 推理调用(10~50s);M2 起质检走快模型
kimi-for-coding(2~4s,few-shot 准确率实测 4/4),经 CRAG_MODEL 可覆盖。
分级模型策略见大脑笔记「延迟工程」。
"""
from __future__ import annotations

import json
import os

from engine.chunking import Chunk

# CRAG 质检开关(.env 可关,关闭后退化为单层阈值拒答)
CRAG_CHECK = os.environ.get("CRAG_CHECK", "1") not in ("0", "false", "False")
# 质检模型(M2 起默认走快模型 kimi-for-coding:k3 推理模型质检要 10~50s,
# 快模型 2~4s 且 few-shot 后准确率 4/4;可用 CRAG_MODEL 覆盖回 k3)
CRAG_MODEL = (os.environ.get("CRAG_MODEL")
              or os.environ.get("FAST_LLM_MODEL")
              or "kimi-for-coding")

# few-shot 系统提示(M2 修订):明确「相关规定即支撑」与「邻接不算支撑」的边界,
# 修复 F06 类误杀(资料含汽车自驾补贴条款,问私车公用油费标准被判不支撑)
_SYSTEM = (
    "你是检索质检员。判断给定的资料片段是否包含回答用户问题所需的规定。\n"
    "判定标准:\n"
    "- 资料中有与问题相关的明确规定(哪怕是间接条款、部分覆盖)即算支撑;\n"
    "- 资料完全没有问题所问事项的规定(只覆盖邻接主题)才算不支撑。\n"
    "示例1:资料「汽车自驾按1.0元/公里计交通费补贴」,问「私车公用油费补贴标准」→ 支撑(有相关规定)。\n"
    "示例2:资料「国内差旅住宿费标准380元/天」,问「国际出差住宿标准」→ 不支撑(无国际出差规定)。\n"
    "示例3:资料「年休假折算公式」,问「离职补偿金怎么算」→ 不支撑(无离职补偿规定)。\n"
    "只输出 JSON:{\"support\": true} 或 {\"support\": false},不要输出其他内容。"
)


def materials_support(query: str, ranked: list[tuple[Chunk, float]],
                      model: str | None = None, max_snippets: int = 3) -> bool:
    """LLM 质检:Top 片段是否支撑回答 query。失败(网络/解析)时默认放行,
    质检层宁可漏拒不可误杀(误杀代价 > 漏拒,见路线图 M1 风险)。"""
    if not ranked:
        return False
    snippets = "\n\n".join(
        f"[资料{i}] 《{c.meta.get('source', c.doc_id)}》\n{c.text[:500]}"
        for i, (c, _) in enumerate(ranked[:max_snippets], 1)
    )
    user = f"【资料片段】\n{snippets}\n\n【用户问题】\n{query}"

    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("LLM_API_KEY", ""),
        base_url=os.environ.get("LLM_BASE_URL") or None,
    )
    try:
        # max_tokens 要给足:k3/快模型都会先烧隐藏思考 token,限额太小会
        # 输出空串(finish_reason=length)造成误杀;实际 JSON 判定仅十余 token
        resp = client.chat.completions.create(
            model=model or CRAG_MODEL or "deepseek-chat",
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": user}],
            max_tokens=2000,
        )
        text = (resp.choices[0].message.content or "").strip()
        # 宽容解析:模型可能在 JSON 外裹文字,找第一个 { 到最后一个 }
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return bool(json.loads(text[start:end + 1]).get("support"))
            except json.JSONDecodeError:
                pass
        if "false" in text.lower():
            return False
        if "true" in text.lower():
            return True
        return True  # 输出异常(空串/截断):放行,退化为单层阈值结果
    except Exception:
        return True  # 质检调用失败:放行,退化为单层阈值结果
