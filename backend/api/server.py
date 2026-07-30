"""FastAPI 服务层(docs/03):把引擎层 RAGPipeline 包装为 HTTP API。

当前实现(M1 第一刀):
  POST /api/v1/chat/ask       SSE 流式问答(meta → delta* → done;拒答两帧)
  POST /api/v1/chat/ask/sync  非流式一次性返回(脚本/测试用)
  POST /api/v1/chat/feedback  答案反馈(👍👎,👎 进 bad case 池)
  GET  /api/v1/health         健康检查(Chroma 连通/块数/模型/LLM 配置)

认证(§1.2):环境变量 AUTH_TOKEN 设置后,除 /health 外需
`Authorization: Bearer <token>`;未设置则开放(开发态)。

运行:  uvicorn api.server:app --host 0.0.0.0 --port 8080   # 项目根目录下
启动即加载 bge 模型并连接 Chroma(约 30~60s),就绪后再接流量。
多 worker 注意:注册表 JSON 与 BM25 缓存为单进程设计,
生产按路线图 M2 迁业务库前请单 worker 运行(--workers 1)。
"""
from __future__ import annotations

import json
import os
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from engine import feedback, kb, stats
from engine.paths import DATA_DIR
from engine.pipeline import RAGPipeline
from engine.refuser import REFUSE_THRESHOLD
from engine.session import SESSIONS
from api.tasks import TaskStore

app = FastAPI(title="企业知识库智能问答系统", version="0.3.0")

# 引擎内核:启动时装载(模型加载慢,避免首请求卡顿)
pipe: RAGPipeline | None = None
# 异步任务表(索引/重建,单进程内存版)
tasks = TaskStore()


@app.on_event("startup")
def _startup() -> None:
    global pipe
    pipe = RAGPipeline()
    pipe.build_index()  # 有索引则复用,无则从 docs_data/ 建
    if pipe.retriever.col.count() == 0:
        raise RuntimeError("Chroma 索引为空,请先 python -m engine.pipeline 建索引")


def _engine() -> RAGPipeline:
    if pipe is None:
        raise HTTPException(503, "引擎尚未就绪")
    return pipe


# ---------- 认证(简单 token,一期) ----------
def _auth(request: Request) -> None:
    token = os.environ.get("AUTH_TOKEN")
    if not token:
        return  # 未配置:开发态开放
    if request.headers.get("Authorization") != f"Bearer {token}":
        raise HTTPException(401, "未认证或 token 失效")


# ---------- 统一响应(§1.3) ----------
def _ok(data: dict) -> dict:
    return {"code": 0, "message": "ok", "data": data}


# ---------- 问答 ----------
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    doc_filter: list[str] | None = None


@app.post("/api/v1/chat/ask")
def chat_ask(req: AskRequest, _: None = Depends(_auth)):
    """SSE 流式问答:event: meta / delta / done(docs/03 §2.1)。"""
    engine = _engine()

    def events():
        try:
            for event, data in engine.ask_stream(
                req.question, top_k=req.top_k,
                doc_filter=req.doc_filter, session_id=req.session_id,
            ):
                payload = json.dumps(data, ensure_ascii=False)
                yield f"event: {event}\ndata: {payload}\n\n"
        except Exception as exc:  # 流中断:给前端可读的 error 帧
            yield f"event: error\ndata: {json.dumps({'code': 500, 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/api/v1/chat/ask/sync")
def chat_ask_sync(req: AskRequest, _: None = Depends(_auth)):
    """非流式问答(docs/03 §2.2),一次性返回完整 JSON。

    上游 LLM 异常(如 429 过载)映射为 503,与统一错误码表一致,
    调用方据此前退重试;不再裸抛 500。
    """
    engine = _engine()
    meta, done = None, None
    try:
        for event, data in engine.ask_stream(
            req.question, top_k=req.top_k,
            doc_filter=req.doc_filter, session_id=req.session_id,
        ):
            if event == "meta":
                meta = data
            elif event == "done":
                done = data
    except Exception as exc:
        raise HTTPException(503, f"生成服务暂不可用,请稍后重试({exc})")
    return _ok({**meta, **done})


# ---------- 答案反馈(docs/03 §2.3) ----------
class FeedbackRequest(BaseModel):
    qa_id: str = Field(min_length=1, max_length=64)
    rating: Literal[1, -1]  # 1=👍 / -1=👎
    issue_type: Literal["not_found", "wrong_answer", "wrong_source",
                        "bad_refuse", "other"] | None = None  # 👎 时可选
    comment: str | None = Field(default=None, max_length=1000)


@app.post("/api/v1/chat/feedback")
def chat_feedback(req: FeedbackRequest, _: None = Depends(_auth)):
    """答案反馈(§2.3):全部落 logs/feedback.jsonl,👎 进 bad case 池。"""
    record = feedback.record_feedback(
        req.qa_id, req.rating, req.issue_type, req.comment)
    return _ok({"received": True, "bad_case": req.rating < 0,
                "ts": record["ts"]})


# ---------- 会话管理(docs/03 §2.4) ----------
@app.get("/api/v1/chat/sessions")
def list_sessions(_: None = Depends(_auth)):
    """会话列表(最近更新在前)。"""
    return _ok({"items": SESSIONS.list()})


@app.delete("/api/v1/chat/sessions/{session_id}")
def clear_session(session_id: str, _: None = Depends(_auth)):
    """清空会话上下文(开启新话题)。"""
    if not SESSIONS.clear(session_id):
        raise HTTPException(404, f"会话不存在: {session_id}")
    return _ok({"cleared": True})


# ---------- 知识库管理(docs/03 §3) ----------
def _safe_filename(name: str) -> str:
    """取纯文件名并校验扩展名,防路径穿越。"""
    base = os.path.basename(name or "")
    if not base or base != name.replace("\\", "/").split("/")[-1]:
        raise HTTPException(400, "非法文件名")
    if os.path.splitext(base)[1].lower() not in kb.SUPPORTED_EXT:
        raise HTTPException(400, f"仅支持 {kb.SUPPORTED_EXT}")
    return base


@app.post("/api/v1/documents")
def upload_document(file: UploadFile = File(...), _: None = Depends(_auth)):
    """上传文档并异步建索引(§3.1)。"""
    engine = _engine()
    base = _safe_filename(file.filename or "")
    path = os.path.join(DATA_DIR, base)
    with open(path, "wb") as f:
        f.write(file.file.read())
    doc_id = os.path.splitext(base)[0]
    task_id = tasks.create(doc_id, "upload")

    def _job() -> None:
        tasks.update(task_id, progress=0.3)  # 已落盘,开始解析+索引
        _, n = kb.add_document(engine.retriever, path)
        tasks.update(task_id, progress=0.9, chunks_total=n)
    tasks.run_async(task_id, _job)
    return _ok({"doc_id": doc_id, "status": "indexing", "task_id": task_id})


@app.get("/api/v1/documents")
def list_documents(page: int = 1, size: int = 20, _: None = Depends(_auth)):
    """文档列表(§3.2),索引中的任务覆盖为 indexing 状态。"""
    engine = _engine()
    items = kb.list_documents(engine.retriever)
    for d in items:
        t = tasks.latest_for_doc(d["doc_id"])
        if t and t["status"] == "indexing":
            d["status"] = "indexing"
    start = (page - 1) * size
    return _ok({"total": len(items), "items": items[start:start + size]})


@app.get("/api/v1/documents/{doc_id}/status")
def document_status(doc_id: str, _: None = Depends(_auth)):
    """索引进度查询(§3.3)。"""
    engine = _engine()
    t = tasks.latest_for_doc(doc_id)
    if t:
        return _ok({"doc_id": doc_id, "status": t["status"],
                    "progress": t["progress"], "error": t["error"],
                    "chunks_total": t.get("chunks_total")})
    for d in kb.list_documents(engine.retriever):
        if d["doc_id"] == doc_id:
            return _ok({"doc_id": doc_id, "status": "done", "progress": 1.0,
                        "chunks_total": d["chunks"]})
    raise HTTPException(404, f"文档不存在: {doc_id}")


@app.delete("/api/v1/documents/{doc_id}")
def delete_document(doc_id: str, _: None = Depends(_auth)):
    """级联删除文档(§3.4):向量 + BM25 + 缓存 + 源文件。"""
    engine = _engine()
    removed = kb.delete_document(engine.retriever, doc_id)
    if not removed:
        raise HTTPException(404, f"文档不存在: {doc_id}")
    return _ok({"deleted_chunks": removed})


@app.post("/api/v1/documents/{doc_id}/rebuild")
def rebuild_document(doc_id: str, _: None = Depends(_auth)):
    """重建单文档索引(§3.5),异步。"""
    engine = _engine()
    if not any(d["doc_id"] == doc_id
               for d in kb.list_documents(engine.retriever)) and \
            not tasks.latest_for_doc(doc_id):
        raise HTTPException(404, f"文档不存在: {doc_id}")
    task_id = tasks.create(doc_id, "rebuild")

    def _job() -> None:
        tasks.update(task_id, progress=0.3)
        _, n = kb.rebuild_document(engine.retriever, doc_id)
        tasks.update(task_id, progress=0.9, chunks_total=n)
    tasks.run_async(task_id, _job)
    return _ok({"doc_id": doc_id, "status": "indexing", "task_id": task_id})


# ---------- 运营统计(docs/03 §4) ----------
@app.get("/api/v1/admin/stats/overview")
def admin_overview(_: None = Depends(_auth)):
    """核心指标:总提问/命中率/拒答率/👍率/P95 延迟/token。"""
    return _ok(stats.overview())


@app.get("/api/v1/admin/stats/hot-questions")
def admin_hot_questions(n: int = 10, _: None = Depends(_auth)):
    """高频问题 Top-N(发现制度盲区)。"""
    return _ok({"items": stats.hot_questions(n)})


@app.get("/api/v1/admin/bad-cases")
def admin_bad_cases(status: str = "open", _: None = Depends(_auth)):
    """bad case 列表(👎样本),按 qa_id 折叠到最新状态。"""
    return _ok({"items": stats.bad_cases(status=status or None)})


class ResolveRequest(BaseModel):
    action: str = Field(min_length=1, max_length=300)  # 处理动作(补文档/调参数…)


@app.post("/api/v1/admin/bad-cases/{qa_id}/resolve")
def admin_resolve_bad_case(qa_id: str, req: ResolveRequest,
                           _: None = Depends(_auth)):
    """标记 bad case 已处理,附处理动作。"""
    if not stats.resolve_bad_case(qa_id, req.action):
        raise HTTPException(404, f"bad case 不存在: {qa_id}")
    return _ok({"resolved": True})


# ---------- 健康检查 ----------
@app.get("/api/v1/health")
def health():
    engine = _engine()
    return _ok({
        "chroma": "ok",
        "chroma_chunks": engine.retriever.col.count(),
        "embed_model": "bge-m3(active)",
        "rerank_model": "bge-reranker-base(active)",
        "llm": f"configured({os.environ.get('LLM_MODEL', 'unset')})",
        "refuse_threshold": REFUSE_THRESHOLD,
    })


@app.exception_handler(HTTPException)
def http_exc_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code,
                        content={"code": exc.status_code,
                                 "message": str(exc.detail), "data": None})
