"""知识库管理(docs/03 §3):文档级的增删查重建。

职责边界:这里只做**文档生命周期编排**(分块 → 增量索引 → 级联删除),
向量/BM25/缓存的维护在 retriever 的增量方法里(add_chunks/remove_doc)。

关键约束:
- doc_id = 文件名去扩展名(与 chunking 一致),全库唯一;
- 上传即异步入索引(调用方在任务层做线程包装,见 api/server.py);
- 删除会同时移除 docs_data/ 下的源文件——否则下次 pipeline 全量重建
  (--rebuild 扫描整个目录)会把"已删除"的文档加回来,状态不一致;
- 重建 = 删 + 增,保证内容更新后 cid 稳定(chunk_0..N 按新内容重排)。
"""
from __future__ import annotations

import os
import time

from engine.chunking import Chunk, chunk_markdown_file, chunk_pdf_file
from engine.paths import DATA_DIR
from engine.retriever import BGERetriever

SUPPORTED_EXT = (".md", ".txt", ".pdf")


def _chunk_file(path: str) -> list[Chunk]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return chunk_pdf_file(path)
    return chunk_markdown_file(path)


def _stamp(chunks: list[Chunk]) -> None:
    """给块元数据补上传时间(列表接口展示用;老块没有该字段则为 None)。"""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    for c in chunks:
        c.meta.setdefault("uploaded_at", ts)


def add_document(retriever: BGERetriever, path: str,
                 on_stage=None) -> tuple[str, int]:
    """解析一个文档并增量入库,返回 (doc_id, 块数)。

    on_stage(stage, progress, **kw):分段进度回调(知识库上传流程展示),
    阶段序列:parsing → chunked → encoding → finalizing。
    """
    def _stage(stage: str, progress: float, **kw) -> None:
        if on_stage:
            on_stage(stage, progress, **kw)

    _stage("parsing", 0.2)
    chunks = _chunk_file(path)
    if not chunks:
        raise ValueError(f"文档解析后无内容: {os.path.basename(path)}")
    _stamp(chunks)
    total = len(chunks)
    _stage("chunked", 0.4, chunks_total=total)

    def _enc_progress(done: int, total_: int) -> None:
        # 编码段占 0.4~0.85 的进度区间
        _stage("encoding", 0.4 + 0.45 * done / total_,
               chunks_done=done, chunks_total=total_)

    retriever.add_chunks(chunks, on_progress=_enc_progress)
    _stage("finalizing", 0.9, chunks_done=total, chunks_total=total)
    return chunks[0].doc_id, total


def delete_document(retriever: BGERetriever, doc_id: str) -> int:
    """级联删除文档(向量/BM25/缓存)+ 源文件,返回删除块数。"""
    removed = retriever.remove_doc(doc_id)
    for ext in SUPPORTED_EXT:  # 见模块 docstring:源文件必须同步删除
        p = os.path.join(DATA_DIR, doc_id + ext)
        if os.path.exists(p):
            os.remove(p)
    return removed


def rebuild_document(retriever: BGERetriever, doc_id: str,
                     on_stage=None) -> tuple[str, int]:
    """重建单文档索引(内容更新后)。源文件不存在时报错。"""
    for ext in SUPPORTED_EXT:
        p = os.path.join(DATA_DIR, doc_id + ext)
        if os.path.exists(p):
            if on_stage:
                on_stage("parsing", 0.2)
            retriever.remove_doc(doc_id)
            return add_document(retriever, p, on_stage=on_stage)
    raise FileNotFoundError(f"源文件不存在: {doc_id}(支持 {SUPPORTED_EXT})")


def list_documents(retriever: BGERetriever) -> list[dict]:
    """按 doc_id 聚合当前库内文档(docs/03 §3.2 的 items)。"""
    docs: dict[str, dict] = {}
    for c in retriever.chunks:
        d = docs.setdefault(c.doc_id, {
            "doc_id": c.doc_id,
            "filename": c.meta.get("source", c.doc_id),
            "chunks": 0,
            "status": "done",
            "uploaded_at": c.meta.get("uploaded_at"),
        })
        d["chunks"] += 1
    return sorted(docs.values(), key=lambda d: d["doc_id"])
