"""传统 RAG 完整链路主流程。

链路(对应四讲知识):
  文档 → 分块(代码) → bge-m3 混合索引 → 检索召回 Top-N
       → bge-reranker 精排 Top-K → 拒答第一层(分数阈值) → CRAG 质检(第二层)
       → 云端 LLM 生成带引用答案

用法:
  1. cp .env.example .env,填入云端 LLM 的 key/base_url/model
  2. 把 .md/.txt/.pdf 文档放进 docs_data/
  3. python -m engine.pipeline            # 建索引(项目根目录下运行)
  4. python -m engine.pipeline "你的问题"  # 问答
"""
from __future__ import annotations

import glob
import os
import sys

from dotenv import load_dotenv

# 先加载 .env 再 import 引擎模块:refuser/checker 的阈值等常量在 import 时读环境
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), ".env"))

from engine.paths import DATA_DIR, DEVICE
from engine.chunking import chunk_markdown_file, chunk_pdf_file, Chunk
from engine.checker import CRAG_CHECK, materials_support
from engine.asklog import log_ask
from engine.generator import generate_answer, route_model
from engine.refuser import REFUSAL_MESSAGE, REFUSE_THRESHOLD, log_refusal, should_refuse
from engine.rewriter import rewrite_query
from engine.session import SESSIONS
from engine.reranker import BGEReranker
from engine.retriever import BGERetriever

RECALL_TOP_N = int(os.environ.get("RECALL_TOP_N", "50"))   # 召回放大
RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", "5"))    # 精排后给 LLM 的块数
DENSE_WEIGHT = float(os.environ.get("DENSE_WEIGHT", "0.7"))


def load_chunks(data_dir: str = DATA_DIR) -> list[Chunk]:
    """加载目录下所有 md/txt/pdf 并分块。"""
    chunks: list[Chunk] = []
    paths = sorted(
        glob.glob(os.path.join(data_dir, "*.md"))
        + glob.glob(os.path.join(data_dir, "*.txt"))
    )
    for path in paths:
        chunks.extend(chunk_markdown_file(path))
    for path in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        chunks.extend(chunk_pdf_file(path))
    return chunks


class RAGPipeline:
    def __init__(self):
        print("连接 Chroma + 加载 bge-m3(首次会从 ModelScope 下载模型)...")
        self.retriever = BGERetriever(device=DEVICE)
        print("加载 bge-reranker-base...")
        self.reranker = BGEReranker(device=DEVICE)

    def build_index(self, data_dir: str = DATA_DIR, rebuild: bool = False) -> None:
        """建索引。Chroma 已有数据且 rebuild=False 时直接复用,跳过 bge 编码。"""
        if not rebuild and self.retriever.load_existing():
            print(f"复用 Chroma 已有索引({self.retriever.col.count()} 块),跳过编码。")
            return
        chunks = load_chunks(data_dir)
        if not chunks:
            raise SystemExit(f"{data_dir}/ 下没有 .md/.txt/.pdf 文档")
        print(f"共 {len(chunks)} 个块,开始建索引(CPU,较慢)...")
        self.retriever.index(chunks)
        print(f"索引完成,已写入 Chroma({self.retriever.col.count()} 块)。")

    def _retrieve_rank(self, query: str, top_k: int | None = None,
                       doc_filter: list[str] | None = None):
        """检索召回 → (可选文档过滤) → 精排,返回 ranked。"""
        recalled = self.retriever.search(query, top_k=RECALL_TOP_N,
                                         dense_weight=DENSE_WEIGHT)
        candidates = [c for c, _ in recalled]
        if doc_filter:  # 限定文档范围(两路统一在候选集上过滤)
            allow = set(doc_filter)
            candidates = [c for c in candidates if c.doc_id in allow]
        return self.reranker.rerank(query, candidates,
                                    top_k=top_k or RERANK_TOP_K)

    def ask(self, query: str) -> tuple[str, list[tuple[Chunk, float]]]:
        ranked = self._retrieve_rank(query)
        # 拒答第一层(分数阈值,F3):最高分 < 阈值 → 不调 LLM 直接拒答
        if should_refuse(ranked, REFUSE_THRESHOLD):
            log_refusal(query, ranked, REFUSE_THRESHOLD, layer="threshold")
            return REFUSAL_MESSAGE, []
        # 拒答第二层(CRAG 质检):分数再高也要确认资料真的支撑问题,
        #    挡住域邻近对抗题(见 checker.py);质检失败默认放行
        if CRAG_CHECK and not materials_support(query, ranked):
            log_refusal(query, ranked, REFUSE_THRESHOLD, layer="crag")
            return REFUSAL_MESSAGE, []
        # 生成(M2 模型路由:简单问题走快模型,见 generator.route_model)
        answer = generate_answer(query, ranked,
                                 model=route_model(query, ranked[0][1]))
        return answer, ranked

    def ask_stream(self, query: str, top_k: int | None = None,
                   doc_filter: list[str] | None = None,
                   session_id: str | None = None):
        """SSE 流式问答(docs/03 §2.1):yield (event, data) 事件对。

        事件序:meta(来源/拒答标记) → delta*(增量答案) → done(统计)。
        两层拒答与 ask() 完全一致:拒答时 meta refused=true 后直接 done。
        """
        import time
        import uuid

        from engine.generator import generate_answer_stream

        t0 = time.time()
        qa_id = uuid.uuid4().hex
        # 多轮(PRD F4):取会话 → 追问先改写为独立问题再走标准链路
        sid, turns = SESSIONS.get_or_create(session_id)
        standalone = rewrite_query(query, turns)

        # 答案缓存(M2 P0):高频重复问题直接回放,跳过检索+生成;
        # 键为改写后的独立问题;知识库变更时 retriever._after_mutation 已全清
        from engine.anscache import get as cache_get, put as cache_put
        cached = cache_get(standalone)
        if cached:
            latency = int((time.time() - t0) * 1000)
            yield "meta", {"sources": cached["sources"], "refused": False,
                           "model": cached["model"], "cache_hit": True,
                           "standalone_question": standalone if standalone != query else None,
                           "qa_id": qa_id, "session_id": sid}
            yield "delta", {"text": cached["answer"]}
            yield "done", {"finish": True, "answer": cached["answer"],
                           "latency_ms": latency, "tokens": 0}
            SESSIONS.append(sid, query, cached["answer"])
            log_ask(qa_id, sid, query, standalone, False, latency, 0,
                    cached["model"], layer="cache")
            return
        ranked = self._retrieve_rank(standalone, top_k=top_k, doc_filter=doc_filter)

        refused = should_refuse(ranked, REFUSE_THRESHOLD)
        layer = "threshold"
        if not refused and CRAG_CHECK:
            refused = not materials_support(standalone, ranked)
            layer = "crag"
        if refused:
            log_refusal(query, ranked, REFUSE_THRESHOLD, layer=layer)
            latency = int((time.time() - t0) * 1000)
            yield "meta", {"sources": [], "refused": True,
                           "refuse_reason": "根据现有资料无法回答该问题",
                           "qa_id": qa_id, "session_id": sid}
            yield "done", {"finish": True, "answer": REFUSAL_MESSAGE,
                           "latency_ms": latency,
                           "tokens": 0}
            SESSIONS.append(sid, query, REFUSAL_MESSAGE)
            log_ask(qa_id, sid, query, standalone, True, latency, 0,
                    None, layer=layer)
            return

        sources = [{
            "n": i,
            "doc": c.meta.get("source", c.doc_id),
            "chunk_id": c.cid.split("#")[-1],
            "score": round(s, 3),
            "snippet": c.text[:80],
        } for i, (c, s) in enumerate(ranked, 1)]
        model = route_model(standalone, ranked[0][1])  # M2 模型路由
        yield "meta", {"sources": sources, "refused": False,
                       "model": model,
                       # 改写可观测:与原文不同时下发,前端可展示"理解为:..."
                       "standalone_question": standalone if standalone != query else None,
                       "qa_id": qa_id, "session_id": sid}

        full, tokens = [], None
        stream = generate_answer_stream(standalone, ranked, model=model)
        while True:
            try:
                delta = next(stream)
            except StopIteration as end:
                tokens = end.value  # 末帧 usage(模型不支持统计时为 None)
                break
            full.append(delta)
            yield "delta", {"text": delta}
        answer = "".join(full)
        cache_put(standalone, answer, sources, model)  # P0:写答案缓存
        latency = int((time.time() - t0) * 1000)
        yield "done", {"finish": True, "answer": answer,
                       "latency_ms": latency,
                       "tokens": tokens}
        SESSIONS.append(sid, query, answer)
        log_ask(qa_id, sid, query, standalone, False, latency, tokens, model)


def main():
    rebuild = "--rebuild" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--rebuild"]
    query = " ".join(argv) if argv else None
    pipe = RAGPipeline()
    pipe.build_index(rebuild=rebuild)
    if not query:
        print("索引已建好。带上问题参数即可问答,如: python pipeline.py \"...\"")
        print("加 --rebuild 强制重建索引。")
        return
    answer, ranked = pipe.ask(query)
    print("\n===== 答案 =====")
    print(answer)
    if not ranked:
        print("\n(已触发拒答,未调用 LLM 生成;详情见 logs/refusals.jsonl)")
        return
    print("\n===== 命中的来源 =====")
    for i, (c, s) in enumerate(ranked, 1):
        print(f"[{i}] (score={s:.3f}) {c.cid} | {c.text[:60].replace(chr(10),' ')}...")


if __name__ == "__main__":
    main()
