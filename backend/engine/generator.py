"""云端 LLM 生成:读精排后的片段,生成带引用的答案。

用 OpenAI 兼容接口(DeepSeek / Qwen / 月之暗面等都兼容)。
配置走环境变量(.env):
    LLM_API_KEY   云端服务的 key
    LLM_BASE_URL  如 https://api.deepseek.com
    LLM_MODEL     如 deepseek-chat / qwen-plus

设计(见大脑笔记「RAG 生成与幻觉抑制」):
- 系统层约束:基于资料作答、不足明说、标注引用;
- 来源标注前置:每块带编号和来源;
- 结构化输出:分点 + 引用编号。
"""
from __future__ import annotations

import os

from engine.chunking import Chunk

# M2 模型路由(验收挂账治理):简单问题走快模型,复杂问题留推理模型。
# k3 生成 40~100s,kimi-for-coding 10~16s(M1 验收实测);路由判据透明可配。
FAST_LLM_MODEL = os.environ.get("FAST_LLM_MODEL", "kimi-for-coding")
ROUTE_MIN_SCORE = float(os.environ.get("ROUTE_MIN_SCORE", "0.85"))
ROUTE_MAX_QLEN = int(os.environ.get("ROUTE_MAX_QLEN", "30"))
# P1 生成长度约束:快模型 max_tokens 封顶,压端到端 token 硬下限(验收报告建议);
# 注意 kimi-for-coding 也是思考型模型(隐藏推理烧 completion 预算)——512 曾把
# 预算烧光导致空答案(M2 验收实测),故封顶放到 2048;k3 这类更重的思考型模型禁用封顶
FAST_MAX_TOKENS = int(os.environ.get("FAST_MAX_TOKENS", "2048"))


def route_model(query: str, top_score: float) -> str:
    """简单事实型问题(高置信 + 短问题)→ 快模型;否则 → 默认推理模型。

    判据刻意简单透明:事实型问题特征是高 top_score 的短问题;
    综合/多跳通常更长或分数分散,留给推理模型保质量。
    """
    if top_score >= ROUTE_MIN_SCORE and len(query.strip()) <= ROUTE_MAX_QLEN:
        return FAST_LLM_MODEL
    return os.environ.get("LLM_MODEL", "deepseek-chat")

_SYSTEM = (
    "你是知识库问答助手。请严格基于给定的资料回答问题;"
    "资料不足以回答时明确说明「根据现有资料无法回答」,不要编造;"
    "在答案的每个关键论断后用 [来源N] 标注引用编号。"
)


def _build_context(ranked: list[tuple[Chunk, float]]) -> str:
    """把精排片段组装成带编号的资料区,来源标注前置。"""
    blocks = []
    for i, (chunk, _) in enumerate(ranked, 1):
        source = chunk.meta.get("source", chunk.doc_id)
        blocks.append(f"[来源{i}] 《{source}》\n{chunk.text}")
    return "\n\n".join(blocks)


def generate_answer(
    query: str,
    ranked: list[tuple[Chunk, float]],
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """根据精排片段生成答案。ranked 为空时直接走拒答。"""
    if not ranked:
        return "根据现有资料无法回答该问题(未检索到相关内容)。"

    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("LLM_API_KEY", ""),
        base_url=os.environ.get("LLM_BASE_URL") or None,
    )
    model = model or os.environ.get("LLM_MODEL", "deepseek-chat")

    context = _build_context(ranked)
    user = f"【背景资料】\n{context}\n\n【用户问题】\n{query}\n\n【输出要求】分点作答;关键论断标注引用编号;结尾列出引用来源。"

    # 部分模型(如 Kimi k3)只允许特定 temperature,默认不传用模型自身默认值
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    capped = model == FAST_LLM_MODEL
    if capped:
        kwargs["max_tokens"] = FAST_MAX_TOKENS  # P1:仅快模型封顶

    # 空输出重试:kimi-for-coding 是思考型,封顶可能把 completion 预算烧光
    # 返回空串(M2 验收实测)——空输出时去封顶重试一次,宁慢不空
    for attempt in range(2 if capped else 1):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        content = resp.choices[0].message.content.strip()
        if content:
            return content
        kwargs.pop("max_tokens", None)
    return ""


def generate_answer_stream(
    query: str,
    ranked: list[tuple[Chunk, float]],
    model: str | None = None,
):
    """流式生成(SSE 用):逐段 yield 增量文本;结束返回 usage 统计。

    见大脑笔记「延迟工程」:生成必须流式,TTFT 决定体感延迟。
    ranked 为空时不产出(拒答分流在 pipeline 层,不走这里)。
    """
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("LLM_API_KEY", ""),
        base_url=os.environ.get("LLM_BASE_URL") or None,
    )
    model = model or os.environ.get("LLM_MODEL", "deepseek-chat")

    context = _build_context(ranked)
    user = f"【背景资料】\n{context}\n\n【用户问题】\n{query}\n\n【输出要求】分点作答;关键论断标注引用编号;结尾列出引用来源。"

    kwargs = {}
    capped = model == FAST_LLM_MODEL
    if capped:
        kwargs["max_tokens"] = FAST_MAX_TOKENS  # P1:仅快模型封顶
    # 空输出重试(同 generate_answer):封顶截断思维链会导致整段空输出,
    # 此时尚未 yield 任何内容,可安全去封顶重试一次
    for attempt in range(2 if capped else 1):
        usage = None
        produced = False
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )
        for event in stream:
            if event.usage is not None:  # 末帧 usage 统计
                usage = event.usage.total_tokens
                continue
            delta = event.choices[0].delta.content if event.choices else None
            if delta:
                produced = True
                yield delta
        if produced:
            return usage
        kwargs.pop("max_tokens", None)
    return usage
