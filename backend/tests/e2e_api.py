# -*- coding: utf-8 -*-
"""端到端实测:rebuild 接口 + /chat/feedback(2026-07-30)。
起服 → 重建单文档(块数幂等) → 真实问答取 qa_id → 👍/👎/非法rating → 校验日志 → 停服。
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "http://localhost:8080/api/v1"
LOG = os.path.join(ROOT, "tmp_test", "server.log")
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def req(method, path, body=None, timeout=180):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    url = BASE + urllib.parse.quote(path, safe="/:?&=")  # doc_id 含中文需百分号编码
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw[:200]}


proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api.server:app", "--port", "8080"],
    stdout=open(LOG, "w", encoding="utf-8"), stderr=subprocess.STDOUT, cwd=ROOT)
try:
    # 1. 等就绪(模型加载慢)
    up = False
    for _ in range(75):
        try:
            s, h = req("GET", "/health", timeout=5)
            if s == 200:
                up = True
                break
        except Exception:
            pass
        time.sleep(2)
    check("服务就绪", up)
    if not up:
        sys.exit(1)
    s, h = req("GET", "/health")
    chunks_before_total = h["data"]["chroma_chunks"]
    print("health:", json.dumps(h["data"], ensure_ascii=False))

    # 2. rebuild:取列表第一个文档,记录块数
    s, lst = req("GET", "/documents")
    doc = lst["data"]["items"][0]
    doc_id, chunks_before = doc["doc_id"], doc["chunks"]
    print(f"重建目标: {doc_id} 块数={chunks_before}")
    s, rb = req("POST", f"/documents/{doc_id}/rebuild")
    check("rebuild 受理", s == 200 and rb["data"]["status"] == "indexing", str(rb)[:120])
    task_id = rb["data"]["task_id"]
    final = None
    for _ in range(60):
        s, st = req("GET", f"/documents/{doc_id}/status")
        if st["data"]["status"] != "indexing":
            final = st["data"]
            break
        time.sleep(2)
    check("rebuild 完成", final and final["status"] == "done", str(final))
    s, lst2 = req("GET", "/documents")
    doc2 = next(d for d in lst2["data"]["items"] if d["doc_id"] == doc_id)
    check("rebuild 块数幂等", doc2["chunks"] == chunks_before,
          f"{chunks_before} -> {doc2['chunks']}")
    s, h2 = req("GET", "/health")
    check("rebuild 总块数不变", h2["data"]["chroma_chunks"] == chunks_before_total,
          f"{chunks_before_total} -> {h2['data']['chroma_chunks']}")

    # 3. 真实问答拿 qa_id
    s, ans = req("POST", "/chat/ask/sync",
                 {"question": "国内出差住宿标准是多少?"}, timeout=180)
    qa_id = ans["data"].get("qa_id") if s == 200 else None
    check("问答拿到 qa_id", bool(qa_id), f"refused={ans['data'].get('refused')}")

    # 4. 👍 反馈(真实 qa_id)
    s, r = req("POST", "/chat/feedback", {"qa_id": qa_id, "rating": 1})
    check("👍 反馈", s == 200 and r["data"]["bad_case"] is False, str(r)[:100])

    # 5. 👎 反馈(带 issue_type + 中文 comment)
    s, r = req("POST", "/chat/feedback", {
        "qa_id": qa_id, "rating": -1,
        "issue_type": "wrong_source", "comment": "引用的段落不是最新标准"})
    check("👎 反馈进 bad case 池", s == 200 and r["data"]["bad_case"] is True, str(r)[:100])

    # 6. 非法 rating → 422
    s, r = req("POST", "/chat/feedback", {"qa_id": qa_id, "rating": 0})
    check("非法 rating 被拒(422)", s == 422, f"got {s}")

    # 7. 日志校验
    fb = open(os.path.join(ROOT, "logs", "feedback.jsonl"), encoding="utf-8") \
        .read().strip().splitlines()
    bc = open(os.path.join(ROOT, "logs", "bad_cases.jsonl"), encoding="utf-8") \
        .read().strip().splitlines()
    check("feedback.jsonl 两条", len(fb) >= 2, f"{len(fb)} 行")
    check("bad_cases.jsonl 一条且字段完整",
          len(bc) >= 1 and json.loads(bc[-1])["issue_type"] == "wrong_source"
          and json.loads(bc[-1])["rating"] == -1,
          bc[-1][:120] if bc else "空")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("\n== 结果 ==", sum(1 for _, ok, _ in RESULTS if ok), "/",
          len(RESULTS), "通过")
